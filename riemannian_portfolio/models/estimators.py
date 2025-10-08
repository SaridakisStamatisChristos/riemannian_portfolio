
from __future__ import annotations
import numpy as np

_EPS = 1e-12

class EWMA:
    def __init__(self, alpha_mu: float = 0.05, alpha_cov: float = 0.05):
        self.alpha_mu = alpha_mu
        self.alpha_cov = alpha_cov
        self.mu = None
        self.S = None

    def update(self, x: np.ndarray):
        x = np.asarray(x, dtype=float)
        if self.mu is None:
            self.mu = x.copy()
            self.S = np.outer(x, x)
            return self.mu, self.cov()
        a1, a2 = self.alpha_mu, self.alpha_cov
        self.mu = (1 - a1) * self.mu + a1 * x
        self.S = (1 - a2) * self.S + a2 * np.outer(x, x)
        return self.mu, self.cov()

    def cov(self) -> np.ndarray:
        C = self.S - np.outer(self.mu, self.mu)
        d = np.trace(C) / (C.shape[0] + _EPS)
        lam = 0.01
        return (1 - lam) * C + lam * d * np.eye(C.shape[0])
