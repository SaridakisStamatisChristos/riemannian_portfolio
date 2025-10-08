
# Riemannian Portfolio Flow (v0.2.0)

Natural‑gradient mirror descent on the portfolio simplex with an Empirical Fisher preconditioner (Normal model pullback) and transaction‑cost‑aware execution via no‑trade bands. Includes robust μ/Σ estimators, a minimal backtester, and a **baseline ablation harness**.

## What's new in v0.2.0
- **Ablation harness** comparing **NG‑EG**, **EG (no Fisher)**, and **Buy‑and‑Hold** across seeds with **net‑of‑costs**.
- **CLI knobs:** `--eta`, `--delta-in`, `--delta-out`, `--T`, `--n`, `--window`, `--cost-bps`, `--seeds`.
- Plotting remains optional/lazy.

## Quick start
```bash
pytest -q
python -m riemannian_portfolio.eval.backtest --seed 7 --T 800 --n 10 --eta 0.5 --delta-in 1e-4 --delta-out 5e-4
python -m riemannian_portfolio.eval.ablation --T 800 --n 10 --seeds 7 11 13 17 --cost-bps 5
```

## Optional plotting
```bash
pip install -e .[plot]
python -m riemannian_portfolio.eval.backtest --seed 7 --plot
```
