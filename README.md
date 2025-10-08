
# Riemannian Portfolio Flow (v0.2.1)

Natural-gradient mirror descent on the portfolio simplex with an Empirical Fisher preconditioner (Normal model pullback) and transaction-cost-aware execution via no-trade bands. Includes robust μ/Σ estimators, a minimal backtester, and a **baseline ablation harness**.

## What's new in v0.2.1
- **KL trust-region** update: constrain step size via `KL(w_prev || w_step) ≤ --kl-step` for stability and fair comparisons.
- **Fisher normalization**: per-step median normalization of the inverse Fisher to avoid scale pathologies.
- **Expanded ablation harness**: compare **NG-EG**, **EG (no Fisher)**, and **Buy-and-Hold** with **net-of-costs** metrics.
- **CLI knobs**: `--eta-eg/--eta-ng`, `--kl-step`, `--cost-bps`, alongside existing controls.

## Quick start
```bash
pytest -q
python -m riemannian_portfolio.eval.backtest --seed 7 --T 800 --n 10 --eta 0.5 --delta-in 1e-4 --delta-out 5e-4 --kl-step 2e-4
python -m riemannian_portfolio.eval.ablation --T 800 --n 10 --seeds 7 11 13 17 19 --cost-bps 10 --kl-step 2e-4 --eta-eg 0.4 --eta-ng 0.4
```

## Optional plotting
```bash
pip install -e .[plot]
python -m riemannian_portfolio.eval.backtest --seed 7 --plot
```
