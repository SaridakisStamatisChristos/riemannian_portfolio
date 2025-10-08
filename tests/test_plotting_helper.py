
import importlib
import numpy as np
from riemannian_portfolio.eval.backtest import _maybe_plot

def test_maybe_plot_skips_without_matplotlib(monkeypatch):
    def fake_import(name):
        if name == "matplotlib.pyplot":
            raise ModuleNotFoundError
        return importlib.import_module(name)
    monkeypatch.setattr(importlib, "import_module", fake_import)
    res = _maybe_plot({'wealth': np.array([1.0, 1.1, 1.2])})
    assert res is False
