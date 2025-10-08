
from __future__ import annotations
import numpy as np

_EPS = 1e-12

def grad_log_utility(mu: np.ndarray, w: np.ndarray) -> np.ndarray:
    mu = np.asarray(mu, dtype=float)
    w = np.asarray(w, dtype=float)
    denom = float(w @ mu)
    scale = max(abs(denom), _EPS)
    denom_safe = np.copysign(scale, denom if denom != 0 else 1.0)
    return mu / denom_safe
