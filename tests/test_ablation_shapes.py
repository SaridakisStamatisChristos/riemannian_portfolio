
from riemannian_portfolio.eval.ablation import run_ablation
import pandas as pd

def test_ablation_returns_dataframe():
    df = run_ablation(T=100, n=4, seeds=(1,2), eta=0.2, delta_in=1e-4, delta_out=5e-4, window=16, cost_bps=5.0)
    assert isinstance(df, pd.DataFrame)
    assert set(["seed","kind","final_wealth","mean_turnover","net_sharpe"]).issubset(df.columns)
    assert df["kind"].isin(["bh","eg","ng"]).all()
