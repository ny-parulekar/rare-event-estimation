"""
Shifted Power Bregman (SPB) loss used for evaluation in the paper.

The per-example loss for ground truth q and estimate p is

    w(q) * B_eps(q | p),   B_eps(q | p) = integral_q^p (t - q) (t + eps)^(-alpha) dt,

with weight w(q) = q^(alpha - 2 - gamma). The shift eps > 0 makes the loss
finite at p = 0 and is chosen via Corollary 1 so that both the scale-distortion
and asymmetry guarantees hold over the benchmarked range
q in [1e-9, 1e-5], relative error factor m in [1.01, 10].
"""

from __future__ import annotations

import numpy as np

# Range of ground-truth probabilities and relative error factors that the
# Corollary-1 epsilon is calibrated for (matching the paper's benchmark).
Q_MIN = 1e-9
Q_MAX = 1e-5
M_MIN = 1.01
M_MAX = 10.0
ETA = 0.05
ALPHA_CALIB = 2.5

# Named (alpha, gamma) scenario combinations reported in the paper.
SCENARIOS = [
    ("Unbiased method benchmarking", 1.5, 0.0),
    ("Product monitoring", 2.0, 0.0),
    ("Deep-tail stress testing", 2.0, 1.0),
    ("Catastrophic-risk audit", 4.0, 2.0),
]


def shifted_power_bregman_div(q, p, alpha: float, eps: float):
    """
    Per-example shifted power Bregman divergence:
        B_eps(q | p) = integral_q^p (t - q) (t + eps)^(-alpha) dt

    For eps = 0 we require q > 0 and p > 0.
    """
    q = np.asarray(q, dtype=np.float64)
    p = np.asarray(p, dtype=np.float64)

    if eps < 0:
        raise ValueError("eps must be >= 0")
    if np.any(q < 0) or np.any(p < 0):
        raise ValueError("q and p must be >= 0")
    if eps == 0.0 and (np.any(q <= 0) or np.any(p <= 0)):
        raise ValueError("eps = 0 requires q > 0 and p > 0.")

    uq = q + eps  # always > 0
    # log((p+eps)/(q+eps)) computed stably using log1p since (p+eps) = (q+eps) + (p-q)
    log_ratio = np.log1p((p - q) / uq)

    if np.isclose(alpha, 1.0):
        div = (p - q) - uq * log_ratio
        return np.maximum(div, 0.0)

    if np.isclose(alpha, 2.0):
        div = log_ratio + np.expm1(-log_ratio)
        return np.maximum(div, 0.0)

    k2 = 2.0 - alpha
    k1 = 1.0 - alpha
    uq_k2 = np.exp(k2 * np.log(uq))
    uq_k1 = np.exp(k1 * np.log(uq))
    delta2 = uq_k2 * np.expm1(k2 * log_ratio)
    delta1 = uq_k1 * np.expm1(k1 * log_ratio)
    div = (delta2 / k2) - uq * (delta1 / k1)
    return np.maximum(div, 0.0)


def weighted_pointwise_loss(q, p, alpha: float, eps: float, gamma: float = 0.0):
    """SPB loss with the appendix weight q^(alpha - 2 - gamma)."""
    q = np.asarray(q, dtype=np.float64)
    if np.any(q <= 0):
        raise ValueError("q must be strictly positive for the weight q^(alpha-2-gamma).")
    weight = np.power(q, alpha - 2.0 - gamma)
    return weight * shifted_power_bregman_div(q=q, p=p, alpha=alpha, eps=eps)


def raw_pointwise_loss(q, p, alpha: float, eps: float):
    return shifted_power_bregman_div(q=q, p=p, alpha=alpha, eps=eps)


# ---------------------------------------------------------------------------
# Corollary-1 epsilon
# ---------------------------------------------------------------------------

def I_fp(alpha: float, m):
    m = np.asarray(m, dtype=np.float64)
    if np.any(m <= 1.0):
        raise ValueError("FP baseline expects m > 1.")
    if np.isclose(alpha, 1.0):
        return m - 1.0 - np.log(m)
    if np.isclose(alpha, 2.0):
        return np.log(m) + (1.0 / m) - 1.0
    return ((np.power(m, 2.0 - alpha) - 1.0) / (2.0 - alpha)) - (
        (np.power(m, 1.0 - alpha) - 1.0) / (1.0 - alpha)
    )


def I_fn(alpha: float, m):
    m = np.asarray(m, dtype=np.float64)
    if np.any(m <= 1.0):
        raise ValueError("FN baseline expects m > 1.")
    if np.isclose(alpha, 1.0):
        return (1.0 / m) - 1.0 + np.log(m)
    if np.isclose(alpha, 2.0):
        return m - 1.0 - np.log(m)
    return ((np.power(m, alpha - 2.0) - 1.0) / (2.0 - alpha)) - (
        (np.power(m, alpha - 1.0) - 1.0) / (1.0 - alpha)
    )


def eps_scale_upper_bound(q_min: float, m_max: float, alpha: float, eta: float) -> float:
    if not 0.0 < eta < 1.0:
        raise ValueError("eta must lie in (0, 1).")
    if q_min <= 0.0 or m_max <= 1.0:
        raise ValueError("Require q_min > 0 and m_max > 1.")
    return (q_min / m_max) * ((1.0 - eta) ** (-1.0 / alpha) - 1.0)


def eps_asym_upper_bound(q_min: float, m_max: float, alpha: float, m_0: float) -> float:
    """
    Upper bound on epsilon for the asymmetric-sensitivity guarantee.
    R_alpha(m) = I_FN(alpha, m) / I_FP(alpha, m) is increasing for alpha > 3/2,
    so over m in [m_0, m_max] the binding constraint is at the lower endpoint m_0.
    """
    if alpha <= 1.5:
        raise ValueError("The asymmetry bound requires alpha > 1.5.")
    if q_min <= 0.0:
        raise ValueError("Require q_min > 0.")
    if not (1.0 < m_0 < m_max):
        raise ValueError("Require 1 < m_0 < m_max.")

    ratio = I_fn(alpha=alpha, m=m_0) / I_fp(alpha=alpha, m=m_0)
    return (q_min / m_max) * (np.power(ratio, 1.0 / alpha) - 1.0)


def corollary_1_upper_bound(q_min: float, m_max: float, alpha: float, eta: float, m_0: float) -> float:
    """
    Simultaneous finite-error guarantee: epsilon must satisfy both the
    scale-distortion bound and the asymmetry bound.
    """
    return min(
        eps_scale_upper_bound(q_min=q_min, m_max=m_max, alpha=alpha, eta=eta),
        eps_asym_upper_bound(q_min=q_min, m_max=m_max, alpha=alpha, m_0=m_0),
    )


def strictify_upper_bound(value: float) -> float:
    # The theorem uses a strict inequality, so step to the nearest smaller float.
    return np.nextafter(float(value), 0.0)


# The corrected Corollary-1 epsilon used throughout the paper's evaluation.
EPS_CORR = strictify_upper_bound(
    corollary_1_upper_bound(q_min=Q_MIN, m_max=M_MAX, alpha=ALPHA_CALIB, eta=ETA, m_0=M_MIN)
)


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------

def aggregate_spb_loss(qs, ps, alpha: float, gamma: float = 0.0, eps: float = EPS_CORR) -> float:
    """
    Average SPB loss of estimates `ps` against ground truths `qs` for a given
    (alpha, gamma) scenario. When alpha - 2 - gamma == 0 the weight is 1 and
    the raw divergence is used.
    """
    qs = np.asarray(qs, dtype=np.float64)
    ps = np.asarray(ps, dtype=np.float64)

    valid = np.isfinite(qs) & np.isfinite(ps)
    qs = qs[valid]
    ps = ps[valid]
    if qs.size == 0:
        return float("nan")

    if np.isclose(alpha - 2.0 - gamma, 0.0):
        losses = raw_pointwise_loss(q=qs, p=ps, alpha=alpha, eps=eps)
    else:
        losses = weighted_pointwise_loss(q=qs, p=ps, alpha=alpha, eps=eps, gamma=gamma)

    metric = np.mean(losses)
    return float(metric) if np.isfinite(metric) else float("nan")
