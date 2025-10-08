
from __future__ import annotations
import numpy as np

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
