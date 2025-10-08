
import numpy as np
import pandas as pd

from riemannian_portfolio.eval.ablation import run_ablation

def test_ablation_returns_dataframe():
    df = run_ablation(T=100, n=4, seeds=(1,2), eta=0.2, delta_in=1e-4, delta_out=5e-4, window=16, cost_bps=5.0)
    assert isinstance(df, pd.DataFrame)
    assert set(["seed","kind","final_wealth","mean_turnover","net_sharpe"]).issubset(df.columns)
    assert df["kind"].isin(["bh","eg","ng"]).all()


def test_ablation_handles_multiple_random_seeds():
    rng = np.random.default_rng(1234)
    seeds = tuple(int(s) for s in rng.choice(10_000, size=5, replace=False))
    df = run_ablation(
        T=60,
        n=3,
        seeds=seeds,
        eta=0.2,
        delta_in=5e-5,
        delta_out=2.5e-4,
        window=8,
        cost_bps=2.5,
    )

    assert isinstance(df, pd.DataFrame)
    assert set(df["seed"]) == set(seeds)
    counts = df["seed"].value_counts()
    # Each strategy kind should appear exactly once per seed
    assert (counts == 3).all()
    assert df.groupby("seed")["kind"].nunique().eq(3).all()

    summary_columns = ["seed", "kind", "final_wealth", "mean_turnover", "net_sharpe"]
    df_summary = df.sort_values(["seed", "kind"])[summary_columns]
    print("\nAblation results by seed and strategy:\n", df_summary.to_string(index=False))
