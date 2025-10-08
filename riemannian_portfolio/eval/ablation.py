
from __future__ import annotations
import argparse
import numpy as np
import pandas as pd

from .backtest import synthetic_prices
from ..core.optim_ng_eg import natural_mirror_step
from ..core.fisher import empirical_fisher_diag
from ..core.bands import no_trade_policy
from ..models.estimators import EWMA
from ..objectives import grad_log_utility

_EPS = 1e-12

def run_strategy(kind: str, rets: np.ndarray, eta: float, delta_in: float, delta_out: float, window: int, cost_bps: float):
    n = rets.shape[1]
    est = EWMA(alpha_mu=0.05, alpha_cov=0.05)
    w = np.ones(n) / n
    wealth = 1.0
    wealth_series = []
    turnovers = []
    roll = []

    for t in range(rets.shape[0]):
        x = rets[t]
        mu, Sigma = est.update(x)

        if kind == "bh":
            w_new = w  # fixed initial uniform; no trades
        else:
            g = grad_log_utility(mu, w)

            if kind == "eg":
                invF = 1.0  # no Fisher preconditioning
            elif kind == "ng":
                roll.append(x)
                if len(roll) > window:
                    roll.pop(0)
                samples = np.vstack(roll)
                diagF = empirical_fisher_diag(w, samples, mu, Sigma)
                invF = 1.0 / np.maximum(diagF, 1e-8)
            else:
                raise ValueError("unknown kind")

            w_prop = natural_mirror_step(w, g, invF, eta)
            w_new = no_trade_policy(w, w_prop, delta_in, delta_out)

        r_p = float(w @ x)
        wealth *= (1.0 + r_p)
        # proportional cost paid on trade size (half L1 norm)
        turnover = 0.5 * float(np.abs(w_new - w).sum())
        cost = (cost_bps / 10000.0) * turnover
        wealth *= (1.0 - cost)

        turnovers.append(turnover)
        wealth_series.append(wealth)
        w = w_new

    return pd.DataFrame({"wealth": wealth_series, "turnover": turnovers})

def summarize(df: pd.DataFrame):
    ret = df["wealth"].to_numpy()
    # simple daily returns from wealth
    rw = np.diff(np.r_[1.0, ret])
    mu = float(np.mean(rw))
    sd = float(np.std(rw) + 1e-12)
    sharpe = mu / sd if sd > 0 else 0.0
    return {
        "final_wealth": float(ret[-1]),
        "mean_turnover": float(df["turnover"].mean()),
        "net_sharpe": sharpe,
    }

def run_ablation(T=800, n=10, seeds=(7,11,13), eta=0.5, delta_in=1e-4, delta_out=5e-4, window=64, cost_bps=5.0):
    rows = []
    for seed in seeds:
        prices = synthetic_prices(T=T, n=n, seed=seed)
        rets = prices.pct_change().dropna().to_numpy()

        for kind in ["bh", "eg", "ng"]:
            res = run_strategy(kind, rets, eta, delta_in, delta_out, window, cost_bps)
            s = summarize(res)
            s.update({"seed": seed, "kind": kind})
            rows.append(s)
    return pd.DataFrame(rows)

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--T", type=int, default=800)
    p.add_argument("--n", type=int, default=10)
    p.add_argument("--seeds", type=int, nargs="+", default=[7,11,13,17,19])
    p.add_argument("--eta", type=float, default=0.5)
    p.add_argument("--delta-in", type=float, default=1e-4, dest="delta_in")
    p.add_argument("--delta-out", type=float, default=5e-4, dest="delta_out")
    p.add_argument("--window", type=int, default=64)
    p.add_argument("--cost-bps", type=float, default=5.0)
    args = p.parse_args()

    df = run_ablation(T=args.T, n=args.n, seeds=tuple(args.seeds), eta=args.eta,
                      delta_in=args.delta_in, delta_out=args.delta_out, window=args.window, cost_bps=args.cost_bps)
    # Pretty print grouped medians (seed-robust view)
    summary = df.groupby("kind").median(numeric_only=True).reset_index()[["kind","final_wealth","mean_turnover","net_sharpe"]]
    print(summary.to_string(index=False))

if __name__ == "__main__":
    main()
