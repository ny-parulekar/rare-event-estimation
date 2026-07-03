"""
Shared helpers: ground-truth loading, target selection, activation generation
and whitening.
"""

import os
import pickle

import numpy as np
import torch as th
from tqdm import tqdm

from .utils import distribution_registry

MODEL_NAMES = ["gelu-1l", "gelu-2l", "gelu-4l"]
DISTRIB_NAMES = ["hex", "camel", "indent", "ifelse", "caps", "english", "spanish", "icl"]
N_REPS_DICT = {"hex": 32, "indent": 32, "ifelse": 32, "camel": 32, "caps": 16, "english": 24, "spanish": 24, "icl": 1}

# Temperatures recommended by Xu et al. (2024) for the importance-sampling baselines.
RECOMMENDED_TEMPS = {
    "gelu-1l": {
        "ITGIS": {"hex": 1.0, "camel": 1.0, "indent": 1.0, "ifelse": 1.0, "caps": 1.5, "english": 0.45, "spanish": 0.67, "icl": 0.45},
        "MHIS": {"hex": 0.67, "camel": 2.24, "indent": 1.0, "ifelse": 2.24, "caps": 3.34, "english": 1.5, "spanish": 2.24, "icl": 1.0},
    },
    "gelu-2l": {
        "ITGIS": {"hex": 1.5, "camel": 1.5, "indent": 1.0, "ifelse": 0.45, "caps": 0.45, "english": 0.67, "spanish": 0.67, "icl": 0.3},
        "MHIS": {"hex": 0.67, "camel": 2.24, "indent": 1.5, "ifelse": 1.5, "caps": 1.5, "english": 2.24, "spanish": 2.24, "icl": 0.67},
    },
    "gelu-4l": {
        "ITGIS": {"hex": 5.0, "camel": 1.0, "indent": 0.67, "ifelse": 1.0, "caps": 0.67, "english": 0.45, "spanish": 1.0, "icl": 3.34},
        "MHIS": {"hex": 0.67, "camel": 2.24, "indent": 1.0, "ifelse": 1.0, "caps": 1.0, "english": 1.5, "spanish": 2.24, "icl": 0.67},
    },
}

# Default location of the ground-truth files shipped with this repo:
# <repo_root>/data/ground_truth/{dist}_{model}/gt_probs.pkl
DEFAULT_GT_ROOT = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "ground_truth"
)


def set_seed(seed: int):
    np.random.seed(seed)
    th.manual_seed(seed)
    if th.cuda.is_available():
        th.cuda.manual_seed_all(seed)


def load_ground_truth_probs(model_name: str, dist_name: str, data_root: str = None, device: str = "cpu") -> th.Tensor:
    """
    Loads the ground-truth output-token probabilities saved in this repo
    (a tensor of length `vocab_size` that sums to 1, obtained from the
    2^32-sample ground-truth frequencies of Xu et al. 2024).
    """
    assert model_name in MODEL_NAMES
    assert dist_name in DISTRIB_NAMES
    data_root = data_root or DEFAULT_GT_ROOT
    path = os.path.join(data_root, f"{dist_name}_{model_name}", "gt_probs.pkl")
    with open(path, "rb") as f:
        gt_probs = pickle.load(f)
    return gt_probs.to(device)


def pick_random_tokens(gt: th.Tensor, count, p_min=1e-9, p_max=1e-5):
    """
    Returns `count` random tokens whose ground-truth probabilities are in the range [p_min, p_max].
    """
    gt_probs = gt / gt.sum()
    valid_idx = th.logical_and(gt_probs >= p_min, gt_probs <= p_max).nonzero().squeeze()
    sampled_idx = th.randperm(valid_idx.size(0))[:count]
    return valid_idx[sampled_idx].tolist()


def gen_acts_pre_and_post_ln(model, dist_name: str, n_samples: int, batch_size: int = 64, show_progress: bool = True):
    """
    Generates `n_samples` samples of the final-position residual-stream activations,
    both before the final layer norm (used by GA-AMLS) and after it, i.e. right
    before the unembedding (used by QLD).

    Returns:
    - acts_pre_ln: (n_samples, d_model)
    - acts_post_ln: (n_samples, d_model)
    """
    assert dist_name in DISTRIB_NAMES
    dist = distribution_registry[dist_name](model.tokenizer)
    samples = dist.sample(n_reps=N_REPS_DICT[dist_name], n_samples=n_samples)
    acts_pre, acts_post = [], []
    device = model.device
    with th.no_grad():
        for batch in tqdm(samples.split(batch_size, dim=0), disable=not show_progress):
            batch = batch.to(device)
            x = model.embed(batch) + model.pos_embed(batch)
            for block in model.blocks:
                x = block(x)
            acts_pre.append(x[:, -1, :].clone())
            x = model.ln_final(x)
            acts_post.append(x[:, -1, :].clone())
    return th.cat(acts_pre), th.cat(acts_post)


def whiten_acts(act_samps: th.Tensor):
    """
    Whitens activation samples. Returns (whitened_samples, mean, A) where
    cov + EPS*I == A @ A.T (Cholesky factor) and
    whitened = (acts - mean) @ inv(A).T.
    """
    d_model = act_samps.shape[1]
    n = act_samps.shape[0]
    mean = act_samps.mean(dim=0)
    cov = (act_samps - mean).T @ (act_samps - mean) / n
    EPS = 1e-4
    A = th.linalg.cholesky(cov + EPS * th.eye(d_model, device=cov.device))
    u_samps = (act_samps - mean) @ th.inverse(A).T
    return u_samps, mean, A


def extract_W_U_oriented(model, hidden_dim: int) -> th.Tensor:
    """Return the unembedding matrix oriented as [hidden, vocab]."""
    W = model.unembed.W_U.detach()
    if W.shape[0] == hidden_dim:
        return W
    if W.shape[1] == hidden_dim:
        return W.t()
    raise RuntimeError(f"W_U shape {W.shape} does not match hidden_dim={hidden_dim}.")


def auto_batch_size(vocab: int, dtype: th.dtype, mem_gb: float = None) -> int:
    """Heuristic chunk size for scoring batches that fills the GPU without OOM."""
    if mem_gb is None:
        mem_gb = float(os.environ.get("PROP_BATCH_GB", "2.0"))
    bytes_per = 2 if dtype in (th.float16, th.bfloat16) else 4
    denom = max(1, vocab * bytes_per * 3)
    B = int((mem_gb * (1024**3)) // denom)
    return max(256, min(B, 16384))
