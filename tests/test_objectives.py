import numpy as np
from riemannian_portfolio.objectives import grad_log_utility


def test_grad_log_utility_matches_formula():
    mu = np.array([0.02, 0.01, -0.03])
    w = np.array([0.4, 0.3, 0.3])
    expected = mu / (w @ mu)
    got = grad_log_utility(mu, w)
    assert np.allclose(got, expected)


def test_grad_log_utility_handles_negative_denominator():
    mu = np.array([-0.04, -0.01, -0.02])
    w = np.array([0.2, 0.5, 0.3])
    denom = w @ mu
    assert denom < 0
    grad = grad_log_utility(mu, w)
    expected = mu / denom
    assert np.allclose(grad, expected)
