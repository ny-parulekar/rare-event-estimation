"""
Score functions for GA-AMLS.

The score of an activation x is s(x) = logit_target(x) - max_{j != target} logit_j(x),
so the rare event {model outputs target} is exactly {s(x) >= 0}.
The score function also handles un-whitening and (for pre-LN activations) the
final layer norm.
"""

import copy
import logging
from abc import ABC, abstractmethod

import torch
import torch.nn as nn

from ..method_utils import extract_W_U_oriented, auto_batch_size

logger = logging.getLogger(__name__)


class BaseScoreFunction(nn.Module, ABC):
    def __init__(self, model, target_token: int, mean: torch.Tensor, A: torch.Tensor):
        super().__init__()
        self.target_token = int(target_token)

        # Buffer registration handles device movement automatically
        if isinstance(mean, torch.Tensor):
            self.register_buffer("mean", mean)
        else:
            self.mean = None
        if isinstance(A, torch.Tensor):
            self.register_buffer("A", A)
        else:
            self.A = None

        self._setup_unembed(model)
        self._setup_layernorm(model)

        self._batch_size = auto_batch_size(self.W_U.shape[1], self.W_U.dtype)

        if self.target_token < 0 or self.target_token >= self.W_U.shape[1]:
            raise IndexError(f"Target {self.target_token} out of vocab range.")

    def _setup_unembed(self, model):
        hidden_dim = model.cfg['d_model'] if hasattr(model, 'cfg') and 'd_model' in model.cfg \
            else model.unembed.W_U.shape[0]
        W_U = extract_W_U_oriented(model, hidden_dim=hidden_dim)
        if hasattr(model, "dtype"):
            W_U = W_U.to(model.dtype)
        self.register_buffer("W_U", W_U)

    @abstractmethod
    def _setup_layernorm(self, model):
        pass

    def _unwhiten(self, x: torch.Tensor) -> torch.Tensor:
        """Transform whitened space 'x' back to activation space."""
        if self.A is None or self.mean is None:
            return x
        # x = (act - mean) @ A^{-T}  <-- Definition of whitening
        # act = x @ A^T + mean       <-- Inverse
        return x @ self.A.T + self.mean

    @abstractmethod
    def _apply_layer_norm(self, x: torch.Tensor) -> torch.Tensor:
        """Strategy for LN application (Pre vs Post)."""
        pass

    def forward(self, x: torch.Tensor, mode: str = "hard", beta: float = 50.0) -> torch.Tensor:
        if x.dim() > 2:
            x = x.reshape(-1, x.shape[-1])

        particles = x.shape[0]

        # 1. Unwhiten
        x = self._unwhiten(x)

        # 2. Apply LN
        x = self._apply_layer_norm(x)

        if x.dim() > 2:
            x = x.reshape(-1, x.shape[-1])

        # 3. Compute scores in batches
        # Use float32 for scores (more stable than bfloat16 for small differences)
        scores = torch.empty(particles, device=x.device, dtype=torch.float32)
        use_amp = x.device.type == 'cuda'

        with torch.autocast(device_type="cuda", enabled=use_amp):
            for start in range(0, particles, self._batch_size):
                end = min(start + self._batch_size, particles)
                xb = x[start:end]

                logits = xb.to(self.W_U.dtype) @ self.W_U
                tgt_logits = logits[:, self.target_token]

                if mode == "hard":
                    top2_vals, top2_idx = logits.topk(k=2, dim=-1)
                    is_top1_target = (top2_idx[:, 0] == self.target_token)
                    max_other = torch.where(is_top1_target, top2_vals[:, 1], top2_vals[:, 0])
                    scores[start:end] = tgt_logits - max_other

                elif mode == "smooth":
                    # Smooth LogSumExp relaxation of the max over other tokens
                    inv_beta = 1.0 / beta
                    logits_f32 = logits.float()
                    logits_f32[:, self.target_token] = float('-inf')
                    lse_others = inv_beta * torch.logsumexp(beta * logits_f32, dim=-1)
                    scores[start:end] = (tgt_logits.float() - lse_others).to(scores.dtype)

        return scores


class PostLNScoreFunction(BaseScoreFunction):
    """For activations sampled after the final layer norm."""

    def _setup_layernorm(self, model):
        pass

    def _apply_layer_norm(self, x: torch.Tensor) -> torch.Tensor:
        return x


class PreLNScoreFunction(BaseScoreFunction):
    """For activations sampled before the final layer norm (used in the paper)."""

    def _setup_layernorm(self, model):
        if hasattr(model, "ln_final"):
            norm = model.ln_final
        elif hasattr(model, "model") and hasattr(model.model, "norm"):
            norm = model.model.norm
        elif hasattr(model, "transformer") and hasattr(model.transformer, "ln_f"):
            norm = model.transformer.ln_f
        else:
            logger.warning("PreLNScoreFunction: Could not find ln_final. Assuming Identity.")
            norm = nn.Identity()

        # Deepcopy safely separates this module from the main model
        self.ln_final = copy.deepcopy(norm)
        for param in self.ln_final.parameters():
            param.requires_grad_(False)

    def _apply_layer_norm(self, x: torch.Tensor) -> torch.Tensor:
        return self.ln_final(x)
