"""
High-level entry point for GA-AMLS, mirroring the pipeline used for the paper:
whiten pre-LN activations -> Student-T prior in whitened space -> pre-LN score
function -> MALA-within-AMLS.
"""

import math
from typing import Optional

import torch

from ..method_utils import whiten_acts
from .amls import run_amls
from .config import AMLSConfig
from .priors import make_default_prior
from .samplers import MALASampler
from .scoring import PreLNScoreFunction


def GA_AMLS(
    model,
    acts_pre_ln: torch.Tensor,
    target: int,
    config: Optional[AMLSConfig] = None,
    logger=None,
    return_full_result: bool = False,
):
    """
    Estimates the probability that `model` outputs `target` using GA-AMLS.

    Inputs:
    - model: the transformer.
    - acts_pre_ln: samples of the final-position residual-stream activations
      BEFORE the final layer norm (n_samples x d_model), e.g. from
      `gen_acts_pre_and_post_ln`.
    - target: the target token.
    - config: AMLSConfig; defaults to the paper's final tuned configuration.
    Returns:
    - The estimated probability of outputting token `target`
      (and the full AMLS result dict if `return_full_result=True`).
    """
    config = config or AMLSConfig()
    device = torch.device(config.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        device = torch.device("cpu")
    config.device = str(device)

    acts = acts_pre_ln.to(device)

    # 1. Whitening (the score function un-whitens internally using mean/A)
    whitened_acts, mean, A = whiten_acts(acts)

    # 2. Score function on pre-LN activations
    scorer = PreLNScoreFunction(model, target, mean, A).to(device)
    score_fn = lambda x: scorer(x, mode="hard")

    # 3. Prior in whitened space
    prior = make_default_prior(whitened_acts, prior_type=config.prior_type, df=5.0, whitening=True)

    # 4. MALA sampler
    sampler = MALASampler(h=config.h, target_acc=config.target_acc)

    # 5. Run AMLS, initialising particles from the data distribution
    result = run_amls(
        sampler=sampler,
        score_fn=score_fn,
        prior=prior,
        config=config,
        initial_acts=whitened_acts,
        logger=logger,
    )

    prob = math.exp(result["log_probability"]) if result["log_probability"] > -math.inf else 0.0
    if return_full_result:
        return prob, result
    return prob
