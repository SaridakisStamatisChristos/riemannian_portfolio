from __future__ import annotations

import numpy as np
import pytest

from riemannian_portfolio.eval.ablation import run_strategy


def test_run_strategy_raises_when_kl_step_inside_band():
    rets = np.random.default_rng(0).normal(size=(5, 3)) * 1e-3
    with pytest.raises(ValueError):
        run_strategy(
            kind="eg",
            rets=rets,
            eta=0.1,
            delta_in=2e-4,
            delta_out=8e-4,
            window=4,
            cost_bps=10.0,
            kl_step=1e-4,
        )
