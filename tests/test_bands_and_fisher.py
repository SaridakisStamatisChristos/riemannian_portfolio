
import numpy as np
from riemannian_portfolio.core.bands import kl_divergence, no_trade_policy
from riemannian_portfolio.core.fisher import empirical_fisher_diag

def test_kl_nonnegative_and_zero_on_self():
    p = np.array([0.2, 0.3, 0.5])
    q = np.array([0.2, 0.3, 0.5])
    d = kl_divergence(p, q)
    assert d >= 0
    assert abs(d - 0.0) < 1e-12

def test_no_trade_policy_holds_within_band():
    w_prev = np.array([0.4, 0.3, 0.3])
    w_prop = np.array([0.41, 0.29, 0.30])
    w_new = no_trade_policy(w_prev, w_prop, delta_in=1e-3, delta_out=5e-3)
    assert np.allclose(w_new, w_prev)

def test_no_trade_policy_snaps_near_boundary():
    w_prev = np.array([0.4, 0.3, 0.3])
    w_prop = np.array([0.1, 0.8, 0.1])
    delta_in, delta_out = 1e-4, 2e-3
    w_new = no_trade_policy(w_prev, w_prop, delta_in, delta_out)
    d = kl_divergence(w_prev, w_new)
    assert d <= delta_out * 1.05

def test_empirical_fisher_diag_positive():
    rng = np.random.default_rng(1)
    n = 5
    w = np.ones(n) / n
    X = rng.normal(size=(32, n)) * 0.01
    mu = X.mean(0)
    Sigma = np.cov(X.T) + 1e-3 * np.eye(n)
    diagF = empirical_fisher_diag(w, X, mu, Sigma)
    assert diagF.shape == (n,)
    assert np.all(diagF > 0)
