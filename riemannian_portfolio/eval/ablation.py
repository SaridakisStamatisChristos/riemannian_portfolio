from __future__ import annotations
import argparse
import numpy as np
import pandas as pd

from .backtest import synthetic_prices
from ..core.optim_ng_eg import natural_mirror_step_trust
from ..core.fisher import empirical_fisher_diag
from ..core.bands import no_trade_policy
from ..models.estimators import EWMA
from ..objectives import grad_log_utility


def run_strategy(
    kind: str,
    rets: np.ndarray,
    eta: float,
    delta_in: float,
    delta_out: float,
    window: int,
    cost_bps: float,
    kl_step: float,
):
    if kl_step <= delta_in:
        raise ValueError(
            "The KL trust-region step must be larger than the inner no-trade band. "
            "Otherwise every proposal stays inside the band and no trades occur. "
            "Increase `--kl-step` or decrease `--delta-in`."
        )
    n = rets.shape[1]
    est = EWMA(alpha_mu=0.05, alpha_cov=0.05)
    w = np.ones(n) / n
    wealth = 1.0
    wealth_series = []
    turnovers = []
    roll: list[np.ndarray] = []

    for t in range(rets.shape[0]):
        x = rets[t]
        mu, Sigma = est.update(x)

        if kind == "bh":
            w_new = w
        else:
            g = grad_log_utility(mu, w)
            if kind == "eg":
                invF: float | np.ndarray = 1.0
            elif kind == "ng":
                roll.append(x)
                if len(roll) > window:
                    roll.pop(0)
                samples = np.vstack(roll)
                diagF = empirical_fisher_diag(w, samples, mu, Sigma)
                invF = 1.0 / np.maximum(diagF, 1e-8)
            else:
                raise ValueError(f"unknown kind: {kind}")
            w_prop = natural_mirror_step_trust(w, g, invF, eta, kl_step=kl_step)
            w_new = no_trade_policy(w, w_prop, delta_in, delta_out)

        r_p = float(w @ x)
        wealth *= 1.0 + r_p
        turnover = 0.5 * float(np.abs(w_new - w).sum())
        cost = (cost_bps / 10000.0) * turnover
        wealth *= 1.0 - cost

        turnovers.append(turnover)
        wealth_series.append(wealth)
        w = w_new

    return pd.DataFrame({"wealth": wealth_series, "turnover": turnovers})


def summarize(df: pd.DataFrame):
    ret = df["wealth"].to_numpy()
    rw = np.diff(np.r_[1.0, ret])
    mu = float(np.mean(rw))
    sd = float(np.std(rw) + 1e-12)
    sharpe = mu / sd if sd > 0 else 0.0
    return {
        "final_wealth": float(ret[-1]),
        "mean_turnover": float(df["turnover"].mean()),
        "net_sharpe": sharpe,
    }


def run_ablation(
    T: int = 800,
    n: int = 10,
    seeds: tuple[int, ...] = (7, 11, 13),
    eta_eg: float = 0.4,
    eta_ng: float = 0.4,
    delta_in: float = 1e-4,
    delta_out: float = 5e-4,
    window: int = 64,
    cost_bps: float = 10.0,
    kl_step: float = 2e-4,
    eta: float | None = None,
):
    if eta is not None:
        eta_eg = eta_ng = eta
    rows = []
    for seed in seeds:
        prices = synthetic_prices(T=T, n=n, seed=seed)
        rets = prices.pct_change().dropna().to_numpy()
        for kind, eta in [("bh", 0.0), ("eg", eta_eg), ("ng", eta_ng)]:
            res = run_strategy(
                kind,
                rets,
                eta,
                delta_in,
                delta_out,
                window,
                cost_bps,
                kl_step,
            )
            summary = summarize(res)
            summary.update({"seed": seed, "kind": kind})
            rows.append(summary)
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--T", type=int, default=800)
    parser.add_argument("--n", type=int, default=10)
    parser.add_argument("--seeds", type=int, nargs="+", default=[7, 11, 13, 17, 19])
    parser.add_argument("--eta-eg", type=float, default=0.4, dest="eta_eg")
    parser.add_argument("--eta-ng", type=float, default=0.4, dest="eta_ng")
    parser.add_argument("--delta-in", type=float, default=1e-4, dest="delta_in")
    parser.add_argument("--delta-out", type=float, default=5e-4, dest="delta_out")
    parser.add_argument("--window", type=int, default=64)
    parser.add_argument("--cost-bps", type=float, default=10.0)
    parser.add_argument("--kl-step", type=float, default=2e-4)
    args = parser.parse_args()

    df = run_ablation(
        T=args.T,
        n=args.n,
        seeds=tuple(args.seeds),
        eta_eg=args.eta_eg,
        eta_ng=args.eta_ng,
        delta_in=args.delta_in,
        delta_out=args.delta_out,
        window=args.window,
        cost_bps=args.cost_bps,
        kl_step=args.kl_step,
    )
    summary = (
        df.groupby("kind").median(numeric_only=True).reset_index()[
            ["kind", "final_wealth", "mean_turnover", "net_sharpe"]
        ]
    )
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
