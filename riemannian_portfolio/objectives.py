
from __future__ import annotations
import numpy as np

_EPS = 1e-12

def grad_log_utility(mu: np.ndarray, w: np.ndarray) -> np.ndarray:
    mu = np.asarray(mu, dtype=float)
    w = np.asarray(w, dtype=float)
    denom = max(float(w @ mu), 1e-6)
    return mu / denom
