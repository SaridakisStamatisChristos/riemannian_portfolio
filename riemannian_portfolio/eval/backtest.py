
from __future__ import annotations
import argparse
import importlib
import numpy as np
import pandas as pd

from ..core.optim_ng_eg import natural_mirror_step_trust
from ..core.fisher import empirical_fisher_diag
from ..core.bands import no_trade_policy
from ..models.estimators import EWMA
from ..objectives import grad_log_utility

def synthetic_prices(T=1000, n=8, seed=7):
    rng = np.random.default_rng(seed)
    A = rng.normal(size=(n, n))
    Sigma = A @ A.T
    Sigma /= np.linalg.norm(Sigma, ord=2)
    mu = rng.uniform(0.0002, 0.0008, size=n)
    X = np.zeros((T, n))
    for t in range(T):
        X[t] = rng.multivariate_normal(mu, Sigma * 1e-3)
    P = 100 * np.exp(np.cumsum(X, axis=0))
    return pd.DataFrame(P, columns=[f"A{i}" for i in range(n)])

def run_backtest(seed=7, T=800, n=10, eta=0.5, delta_in=1e-4, delta_out=5e-4, window=64, kl_step=2e-4):
    prices = synthetic_prices(T=T, n=n, seed=seed)
    rets = prices.pct_change().dropna().to_numpy()

    est = EWMA(alpha_mu=0.05, alpha_cov=0.05)
    w = np.ones(n) / n

    wealth = 1.0
    turnovers = []
    wealth_series = []

    roll = []

    for t in range(rets.shape[0]):
        x = rets[t]
        mu, Sigma = est.update(x)

        g = grad_log_utility(mu, w)

        roll.append(x)
        if len(roll) > window:
            roll.pop(0)
        samples = np.vstack(roll)
        diagF = empirical_fisher_diag(w, samples, mu, Sigma)
        invF = 1.0 / np.maximum(diagF, 1e-8)

        w_prop = natural_mirror_step_trust(w, g, invF, eta, kl_step=kl_step)
        w_new = no_trade_policy(w, w_prop, delta_in, delta_out)

        r_p = float(w @ x)
        wealth *= (1.0 + r_p)
        wealth_series.append(wealth)
        turnovers.append(0.5 * np.abs(w_new - w).sum())
        w = w_new

    df = pd.DataFrame({"wealth": wealth_series, "turnover": turnovers})
    return df

def _maybe_plot(res) -> bool:
    try:
        plt = importlib.import_module("matplotlib.pyplot")
    except ModuleNotFoundError:
        print("[info] matplotlib not installed; skipping plot. Install with `pip install -e .[plot]`.")
        return False
    fig, ax = plt.subplots()
    ax.plot(res["wealth"].values)
    ax.set_title("Wealth trajectory")
    ax.set_xlabel("Time"); ax.set_ylabel("Wealth")
    plt.show()
    return True

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--T", type=int, default=800)
    p.add_argument("--n", type=int, default=10)
    p.add_argument("--eta", type=float, default=0.5)
    p.add_argument("--delta-in", type=float, default=1e-4, dest="delta_in")
    p.add_argument("--delta-out", type=float, default=5e-4, dest="delta_out")
    p.add_argument("--window", type=int, default=64)
    p.add_argument("--kl-step", type=float, default=2e-4)
    p.add_argument("--plot", action="store_true")
    args = p.parse_args()

    res = run_backtest(seed=args.seed, T=args.T, n=args.n, eta=args.eta,
                       delta_in=args.delta_in, delta_out=args.delta_out, window=args.window, kl_step=args.kl_step)
    print(res.describe().T)
    if args.plot:
        _maybe_plot(res)

if __name__ == "__main__":
    main()
