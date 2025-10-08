
from __future__ import annotations
import numpy as np

_EPS = 1e-12

def kl_divergence(p: np.ndarray, q: np.ndarray) -> float:
    p = np.clip(np.asarray(p, dtype=float), _EPS, None)
    q = np.clip(np.asarray(q, dtype=float), _EPS, None)
    p /= p.sum(); q /= q.sum()
    return float(np.sum(p * (np.log(p) - np.log(q))))

def no_trade_policy(
    w_prev: np.ndarray,
    w_prop: np.ndarray,
    delta_in: float,
    delta_out: float,
) -> np.ndarray:
    w_prev = np.asarray(w_prev, dtype=float)
    w_prop = np.asarray(w_prop, dtype=float)

    def mix_in_log_coordinates(p, q, t):
        lp = np.log(np.clip(p, _EPS, None))
        lq = np.log(np.clip(q, _EPS, None))
        z = (1 - t) * lp + t * lq
        z = np.exp(z)
        z /= z.sum()
        return z

    d = kl_divergence(w_prev, w_prop)
    if d <= delta_in:
        return w_prev.copy()
    if d >= delta_out:
        lo, hi = 0.0, 1.0
        target = delta_out
        for _ in range(30):
            mid = 0.5 * (lo + hi)
            wm = mix_in_log_coordinates(w_prev, w_prop, mid)
            dm = kl_divergence(w_prev, wm)
            if dm < target:
                lo = mid
            else:
                hi = mid
        return mix_in_log_coordinates(w_prev, w_prop, lo)
    t = (d - delta_in) / max(delta_out - delta_in, 1e-6)
    t = np.clip(t, 0.0, 1.0)
    return mix_in_log_coordinates(w_prev, w_prop, t)
