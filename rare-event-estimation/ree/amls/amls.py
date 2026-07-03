import logging
import math
import time
from typing import Any, Optional

import torch

from .config import AMLSConfig
from .samplers import BaseSampler


def run_amls(
    sampler: BaseSampler,
    score_fn: Any,  # Callable that takes x and returns scores
    prior: Any,  # e.g. torch.distributions.Distribution
    config: AMLSConfig,
    initial_acts: Optional[torch.Tensor] = None,
    logger: Optional[logging.Logger] = None
) -> dict:
    """
    Executes the Adaptive Multilevel Splitting (AMLS) algorithm.

    Args:
        sampler: An instance of a Sampler (e.g. MALASampler).
                 Owns the 'step' logic and 'adapt' logic.
        score_fn: Calculates the hard score s(x); the rare event is {s(x) >= 0}.
        prior: The proposal distribution (e.g., Student-T).
        config: AMLSConfig containing n_particles, rho, mh_steps, etc.
        initial_acts: Optional activation samples used to initialise particles
                      from the data distribution instead of the prior.

    Returns:
        dict with 'log_probability' (natural log of the estimated probability),
        'levels', 'final_particles', 'stats' and 'successful_particles'.
    """

    # 1. Initialization
    # Sample the initial particles from the loaded activations (when enough
    # are available) so we rely less on the parametric prior at the start.
    device = config.device if isinstance(config.device, torch.device) else torch.device(config.device)

    if initial_acts is not None and len(initial_acts) >= config.n_particles:
        idx = torch.randperm(len(initial_acts), device=initial_acts.device)[:config.n_particles]
        x = initial_acts[idx].to(device, non_blocking=True)
    else:
        x = prior.sample((config.n_particles,)).to(device, non_blocking=True)

    # Track statistics
    lg_p = 0.0
    levels = []
    stats_per_level = []
    L_prev = -math.inf
    successful_particles = None

    # 2. Main Level Loop
    for level_idx in range(config.n_levels):
        t0 = time.time()

        # Get scores (score_fn handles whitening/unwhitening internally)
        with torch.inference_mode():
            s_x = score_fn(x)

        # --- Get Level Threshold (L) with Tie-Breaking (Cerou et al. 2015) ---
        n_keep = int(round(config.rho * config.n_particles))
        if n_keep == 0 and config.n_particles > 0:
            n_keep = 1
        n_kill = config.n_particles - n_keep

        # Impose a random total order among ties:
        # add a tiny, strictly-ordered jitter so argsort becomes a uniform random tie-break
        perm = torch.randperm(config.n_particles, device=s_x.device)
        jitter = (perm.float() / (config.n_particles + 1)) * 1e-12
        s_order = s_x + jitter

        # Sort from worst to best using the randomized total order
        sorted_indices = torch.argsort(s_order, dim=0)  # ascending

        # Define level L from the first kept element
        L_score = s_x[sorted_indices[n_kill]].item()
        L = min(L_score, 0.0)

        # Stagnation check
        if L == L_prev:
            L = 0.0

        levels.append(L)

        if logger:
            logger.info(f"Level {level_idx+1}: L={L:.6f} | keep={n_keep} kill={n_kill}")

        if n_keep == 0:
            return {
                "log_probability": -math.inf,
                "levels": levels,
                "final_particles": x,
                "stats": stats_per_level,
                "successful_particles": successful_particles
            }

        # Update probability estimate
        lg_p += math.log(n_keep) - math.log(config.n_particles)

        if lg_p <= -33.0:
            stats_per_level.append({
                "level": level_idx,
                "L": L,
                "runtime_sec": time.time() - t0
            })
            return {
                "log_probability": -33.0,
                "levels": levels,
                "final_particles": x,
                "stats": stats_per_level,
                "successful_particles": successful_particles
            }

        # If we hit the target (L=0), stop
        if L >= 0.0:
            if logger:
                logger.info("Target level 0.0 reached.")
            success_mask = s_x >= 0
            if success_mask.any():
                successful_particles = x[success_mask].clone()
            stats_per_level.append({
                "level": level_idx,
                "L": L,
                "runtime_sec": time.time() - t0
            })
            break

        # --- C. Selection & Resampling ---
        # Keep the top-ranked indices (exactly n_keep)
        keep_indices = sorted_indices[n_kill:]
        keep_mask = torch.zeros_like(s_x, dtype=torch.bool)
        keep_mask[keep_indices] = True

        x_kept = x[keep_mask]

        # Resample (clone) to fill back to N
        resample_idx = torch.randint(
            low=0, high=len(x_kept), size=(n_kill,), device=device
        )
        x_clones = x_kept[resample_idx]

        x = torch.cat([x_kept, x_clones], dim=0)

        # Reset sampler caches at start of each level so gradients are
        # recomputed for the new particle positions
        sampler.on_level_start()

        # Prior log-prob of the current batch (needed for the MH ratio)
        lp_x = prior.log_prob(x)

        # --- D. Evolution (The MCMC Step) ---
        accept_counts = torch.zeros(config.n_particles, device=device)

        # Burn-in steps during which the sampler adapts its step size
        if config.n_burn is not None:
            n_burn = config.n_burn
        else:
            n_burn = int(config.burn_percentage * config.mh_steps)

        for t in range(config.mh_steps):
            # 1. Propose New Particles
            x_prop, log_q_fwd, log_q_rev = sampler.step(x, L, score_fn, prior)

            # 2. Check Hard Constraint (Validity)
            with torch.inference_mode():
                s_prop = score_fn(x_prop)
            valid_mask = (s_prop >= L)

            # 3. Metropolis-Hastings Acceptance Ratio
            lp_prop = prior.log_prob(x_prop)

            # log alpha = (Prior_new + Trans_rev) - (Prior_old + Trans_fwd)
            log_alpha = (lp_prop + log_q_rev) - (lp_x + log_q_fwd)

            # 4. Accept/Reject
            log_u = torch.log(torch.rand_like(log_alpha))
            accept = (log_u <= log_alpha) & valid_mask

            # 5. Update State
            if accept.any():
                x[accept] = x_prop[accept]
                lp_x = torch.where(accept, lp_prop, lp_x)
                sampler.on_accept(accept)

            accept_counts += accept.float()

            # 6. Adapt (the sampler internally checks t < n_burn)
            sampler.adapt(accept.float(), step_idx=t, n_burn=n_burn)

        # --- E. Stats ---
        acc_ratio = accept_counts / float(config.mh_steps)
        avg_acc_rate = acc_ratio.mean().item()
        median_acc_rate = acc_ratio.median().item()
        std_acc = acc_ratio.std(unbiased=True).item() if config.n_particles > 1 else 0.0
        sem_acc = (std_acc / math.sqrt(max(config.n_particles, 1)))

        stats_per_level.append({
            "level": level_idx,
            "L": L,
            "acc_rate_mean": avg_acc_rate,
            "acc_rate_median": median_acc_rate,
            "acc_rate_sem": sem_acc,
            "runtime_sec": time.time() - t0,
            "sampler_stats": sampler.get_diagnostics()
        })

        L_prev = L

    return {
        "log_probability": lg_p,
        "levels": levels,
        "final_particles": x,
        "stats": stats_per_level,
        "successful_particles": successful_particles
    }
