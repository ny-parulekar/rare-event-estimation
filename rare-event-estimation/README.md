# Estimating Rare Events in Language Models with Proper Evaluation

Code and ground-truth data accompanying the paper *"Estimating Rare Events in Language Models with Proper Evaluation"*.

The repo contains everything needed to reproduce the paper's pipeline on any of the 24 (model, distribution) benchmark pairs:

- **GA-AMLS** (ours): gradient-assisted adaptive multilevel splitting with a MALA sampler on whitened pre-layer-norm activations (`ree/amls/`).
- **Baselines** from [Xu et al. (2024)](https://github.com/alignment-research-center/low-probability-estimation): QLD, ITGIS and MHIS (`ree/methods.py`). The `random_constraint_projection` routine used by QLD is the **bug-corrected** version (correct tolerance sign, restored norm-minimisation loop, no crash when all constraints are satisfied at the start).
- **SPB loss**: the Shifted Power Bregman evaluation metric with the corrected Corollary-1 shift and the four (alpha, gamma) scenarios from the paper (`ree/metrics.py`).
- **Ground truth**: per-token output probabilities estimated from 2^32 samples for each of the 24 (distribution, model) pairs, in `data/ground_truth/{dist}_{model}/gt_probs.pkl` (`torch` tensors of length `d_vocab = 48262`).

## Installation

```bash
pip install -e .
```

(or `pip install -r requirements.txt` and run from the repo root). Python >= 3.10 and PyTorch >= 2.2 are required. A CUDA GPU is strongly recommended for the full paper configuration.

Model weights (`gelu-1l`, `gelu-2l`, `gelu-4l` from NeelNanda's HF repos) are downloaded automatically on first use. The input distributions stream their source datasets (`allenai/c4`, `codeparrot/codeparrot-clean`) from HuggingFace; set `HF_TOKEN` if your environment requires authentication.

## Quick start

Open **`example.ipynb`**. It walks through the full pipeline for one (model, distribution) pair:

1. load the model and the shipped ground-truth probabilities,
2. pick rare target tokens with ground truth in `[1e-9, 1e-5]`,
3. generate pre-/post-layer-norm activation samples,
4. estimate each target's probability with QLD, ITGIS, MHIS and GA-AMLS,
5. score all methods with the average SPB loss under the paper's four scenarios.

The notebook has a `QUICK_TEST` flag for a fast CPU sanity check; with `QUICK_TEST = False` it uses the exact final configurations from the paper.

## Final experiment configuration

The GA-AMLS configuration used for all paper experiments is the default of `ree.amls.AMLSConfig`:

| parameter | value |
|---|---|
| sampler | MALA |
| prior | Student-T (df=5) in whitened space |
| score function | pre-LN (final layer norm applied inside the score) |
| `n_particles` | 2000 |
| `mh_steps` | 900 |
| `rho` (keep fraction) | 0.6 |
| `h` (step size) | 1e-3 |
| `target_acc` | 0.57 |
| `burn_percentage` | 0.14 |

Baselines use `2^16` samples and the temperatures recommended by Xu et al. (2024) (`ree.method_utils.RECOMMENDED_TEMPS`).

## Repository layout

```
example.ipynb            # end-to-end demo: estimates + SPB evaluation
data/ground_truth/       # gt_probs.pkl for each of the 24 (dist, model) pairs
ree/
├── methods.py           # QLD (bug-corrected), ITGIS, MHIS baselines
├── method_utils.py      # ground-truth loading, target picking, activation sampling, whitening
├── metrics.py           # SPB loss, Corollary-1 epsilon, paper scenarios
├── amls/                # GA-AMLS: AMLS loop, MALA sampler, priors, score functions, GA_AMLS entry point
└── utils/               # transformer implementation and the eight input distributions (from Xu et al. 2024)
```
