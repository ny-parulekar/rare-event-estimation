import math
from abc import ABC, abstractmethod
from typing import Any, Tuple

import torch


class AMLSAdapterMixin:
    """
    Implements Global, Constant-Gain Adaptation with a hard Burn-in stop.
    """

    def adapt_param(self, current_val: float, acc_rate: float, target: float, gamma: float = 0.1):
        """
        Log-Space Multiplicative Rule
        Scale = exp(gamma * (acc - target))
        This is symmetric and handles orders of magnitude change efficiently.
        """
        scaling = math.exp(gamma * (acc_rate - target))
        return current_val * scaling

    def adapt(self, acceptance_rate: Any, step_idx: int, n_burn: int = 50):
        """
        Args:
            acceptance_rate: Tensor or float.
            step_idx: The current step in the level loop (0 to mh_steps).
            n_burn: How many steps to adapt before freezing.
        """
        # Stop adapting after burn-in to restore Detailed Balance
        if step_idx >= n_burn:
            return

        if not hasattr(self, 'h'):
            return
        current_val = self.h
        min_val, max_val = 1e-9, 10.0
        target = getattr(self, 'target_acc', 0.50)

        if isinstance(acceptance_rate, torch.Tensor):
            avg_acc = acceptance_rate.float().mean().item()
        else:
            avg_acc = float(acceptance_rate)

        new_val = self.adapt_param(current_val, avg_acc, target)
        self.h = max(min_val, min(max_val, new_val))


class BaseSampler(ABC):
    """
    Abstract base class for MCMC samplers used inside AMLS.
    """

    @abstractmethod
    def step(self, x: torch.Tensor, L: float, score_fn: Any, prior: Any) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Perform one MCMC step.
        Args:
            x: Current particles [N, D]
            L: Current level threshold
            score_fn: Score function
            prior: Prior distribution (used for gradient computation)
        Returns: (x_prop, log_q_fwd, log_q_rev)
          - x_prop: The proposed state
          - log_q_fwd: log q(x_prop | x)
          - log_q_rev: log q(x | x_prop)
        """
        pass

    @abstractmethod
    def adapt(self, acceptance_rate: Any, step_idx: int, n_burn: int = 50):
        """Update internal parameters based on acceptance rate."""
        pass

    @abstractmethod
    def on_accept(self, accept_mask: torch.Tensor):
        """Hook called when a proposal is accepted (to update caches)."""
        pass

    def on_level_start(self):
        """Hook called at the start of each AMLS level to reset caches."""
        pass

    def get_diagnostics(self) -> dict:
        return {}


class MALASampler(AMLSAdapterMixin, BaseSampler):
    """
    Metropolis-Adjusted Langevin sampler targeting the prior restricted to the
    current AMLS level set {x : score(x) >= L}.
    """

    def __init__(self, h: float, target_acc: float = 0.574):
        """
        Args:
            h: Langevin step size
            target_acc: Target acceptance rate for the burn-in adaptation
        """
        self.h = float(h)
        self.target_acc = target_acc
        self.device = None
        self.grad_cache = None
        self._pending_grad = None

    def _get_grad(self, x: torch.Tensor, prior: Any):
        # Detach x to stop gradient propagation into previous steps
        x_req = x.detach().requires_grad_(True)
        log_p = prior.log_prob(x_req).sum()
        grad = torch.autograd.grad(log_p, x_req)[0]
        return grad.detach()

    def _log_q(self, x_to, x_from, grad_from):
        """
        log q(x' | x) = log N(x' | x + h*grad, 2h * I)
        """
        mean = x_from + self.h * grad_from
        diff = x_to - mean

        norm_sq = torch.sum(diff**2, dim=-1)
        d = x_to.shape[1]

        const = -0.5 * d * math.log(4 * math.pi * self.h)
        return const - (norm_sq / (4.0 * self.h))

    def step(self, x: torch.Tensor, L: float, score_fn: Any, prior: Any):
        if self.device is None:
            self.device = x.device

        # 1. Get Gradient at current x
        if self.grad_cache is None:
            self.grad_cache = self._get_grad(x, prior)

        grad_x = self.grad_cache

        # 2. Propose: x_prop = x + h*grad + sqrt(2h)*z
        z = torch.randn_like(x)
        noise_scale = math.sqrt(2 * self.h)
        x_prop = x + (self.h * grad_x) + (noise_scale * z)

        # 3. Check the hard constraint
        with torch.no_grad():
            s_prop = score_fn(x_prop)

        if s_prop.dim() > 1:
            s_prop = s_prop.view(-1)

        is_valid = (s_prop >= L)

        # 4. Get Gradient at proposed x (only for valid particles)
        grad_prop = torch.zeros_like(x)
        if is_valid.any():
            valid_x = x_prop[is_valid]
            grad_prop[is_valid] = self._get_grad(valid_x, prior)

        # 5. Compute transition probs
        log_q_fwd = self._log_q(x_prop, x, grad_x)
        log_q_rev = self._log_q(x, x_prop, grad_prop)

        # SAFETY MASKING: if a particle is invalid, its reverse transition
        # probability implies starting from an invalid state, which is
        # impossible. Overwrite log_q_rev with -inf to guarantee rejection.
        log_q_rev = torch.where(is_valid, log_q_rev, torch.tensor(-float('inf'), device=x.device))

        # Store pending grad
        self._pending_grad = grad_prop

        return x_prop, log_q_fwd, log_q_rev

    # adapt() is provided by AMLSAdapterMixin

    def on_accept(self, accept_mask):
        if self.grad_cache is not None and self._pending_grad is not None:
            self.grad_cache[accept_mask] = self._pending_grad[accept_mask]

    def on_level_start(self):
        self.grad_cache = None
