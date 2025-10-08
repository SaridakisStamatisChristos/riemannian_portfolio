
from __future__ import annotations
import numpy as np
from .bands import kl_divergence

_EPS = 1e-12

def project_to_simplex(w: np.ndarray) -> np.ndarray:
    v = np.asarray(w, dtype=float)
    if np.all(v >= 0) and abs(v.sum() - 1.0) <= 1e-10:
        return v
    u = np.sort(v)[::-1]
    cssv = np.cumsum(u)
    rho = np.nonzero(u * np.arange(1, len(u) + 1) > (cssv - 1))[0][-1]
    theta = (cssv[rho] - 1) / (rho + 1)
    w_proj = np.maximum(v - theta, 0.0)
    s = w_proj.sum()
    if s <= 0:
        w_proj = np.ones_like(v) / v.size
    else:
        w_proj /= s
    return w_proj

def natural_mirror_step(
    w: np.ndarray,
    grad: np.ndarray,
    inv_precond: np.ndarray | float,
    eta: float,
) -> np.ndarray:
    w = np.asarray(w)
    grad = np.asarray(grad)

    if np.isscalar(inv_precond):
        step = eta * inv_precond * grad
    else:
        step = eta * inv_precond * grad

    step = np.asarray(step, dtype=float)
    step -= np.max(step)

    z = w * np.exp(step)
    z_sum = z.sum()

    if not np.isfinite(z_sum) or z_sum <= _EPS:
        step = np.clip(step, -700.0, 700.0)
        z = w * np.exp(step)
        z_sum = z.sum()

    if not np.isfinite(z_sum) or z_sum <= _EPS:
        z = w + 1e-4 * grad

    z = np.clip(z, _EPS, None)
    z /= z.sum()
    return z


def _eg_step(w: np.ndarray, step: np.ndarray) -> np.ndarray:
    """Helper: exponentiated-gradient step on the simplex."""
    step = np.asarray(step, dtype=float)
    step -= np.max(step)
    z = w * np.exp(step)
    z = np.clip(z, _EPS, None)
    z /= z.sum()
    return z


def natural_mirror_step_trust(
    w: np.ndarray,
    grad: np.ndarray,
    inv_precond: np.ndarray | float,
    eta: float,
    kl_step: float = 2e-4,
) -> np.ndarray:
    """Natural-gradient EG step with KL trust region and Fisher normalization."""
    w = np.asarray(w)
    grad = np.asarray(grad)

    if np.isscalar(inv_precond):
        invp = float(inv_precond)
    else:
        invp = np.asarray(inv_precond, dtype=float)
        med = float(np.median(invp)) if invp.size else 1.0
        if med <= 0:
            med = 1.0
        invp = invp / med

    step = eta * (invp * grad if not np.isscalar(invp) else invp * grad)
    step = np.asarray(step, dtype=float)

    z = _eg_step(w, step)
    d = kl_divergence(w, z)
    if d <= kl_step:
        return z

    lo, hi = 0.0, 1.0
    for _ in range(30):
        mid = 0.5 * (lo + hi)
        z = _eg_step(w, step * mid)
        d = kl_divergence(w, z)
        if d > kl_step:
            hi = mid
        else:
            lo = mid
    return _eg_step(w, step * lo)
