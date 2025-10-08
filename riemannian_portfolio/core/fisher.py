
from __future__ import annotations
import numpy as np
from .distributions import NormalPortfolioModel

_EPS = 1e-12

def empirical_fisher_diag(
    w: np.ndarray,
    samples: np.ndarray,
    mu: np.ndarray,
    Sigma: np.ndarray,
    clip: float | None = 10.0,
) -> np.ndarray:
    w = np.asarray(w)
    mu = np.asarray(mu)
    Sigma = np.asarray(Sigma)

    if samples.ndim == 2:
        r_p = samples @ w
    else:
        r_p = np.asarray(samples)

    model = NormalPortfolioModel(mu, Sigma)

    acc = np.zeros_like(w)
    for x in np.atleast_1d(r_p):
        g = model.grad_w_logp(float(x), w)
        if clip is not None:
            ng = np.linalg.norm(g) + _EPS
            if ng > clip:
                g = g * (clip / ng)
        acc += g * g

    diagF = acc / max(len(r_p), 1)
    return np.maximum(diagF, 1e-8)
