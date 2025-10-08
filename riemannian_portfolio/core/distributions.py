
from __future__ import annotations
import numpy as np

_EPS = 1e-12

class NormalPortfolioModel:
    def __init__(self, mu: np.ndarray, Sigma: np.ndarray):
        self.mu = np.asarray(mu)
        self.Sigma = np.asarray(Sigma)
        assert self.mu.ndim == 1
        assert self.Sigma.shape == (self.mu.size, self.mu.size)

    def ms(self, w: np.ndarray) -> tuple[float, float]:
        w = np.asarray(w)
        m = float(w @ self.mu)
        s2 = float(w @ self.Sigma @ w)
        s = float(np.sqrt(max(s2, _EPS)))
        return m, s

    def score_ms(self, x: float, m: float, s: float) -> tuple[float, float]:
        dm = (x - m) / (s * s + _EPS)
        ds = -(1.0 / (s + _EPS)) + ((x - m) ** 2) / ((s ** 3) + _EPS)
        return dm, ds

    def grad_w_logp(self, x: float, w: np.ndarray) -> np.ndarray:
        m, s = self.ms(w)
        dm, ds = self.score_ms(x, m, s)
        return dm * self.mu + ds * (self.Sigma @ w) / (s + _EPS)
