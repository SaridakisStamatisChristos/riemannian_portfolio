import numpy as np

from riemannian_portfolio.core.optim_ng_eg import natural_mirror_step_trust
from riemannian_portfolio.core.bands import kl_divergence


def test_trust_region_limits_kl():
    w = np.array([0.3, 0.4, 0.3])
    grad = np.array([0.5, -0.2, -0.1])
    invF = np.array([2.0, 1.0, 0.5])
    z = natural_mirror_step_trust(w, grad, invF, eta=1.0, kl_step=1e-4)
    assert kl_divergence(w, z) <= 1.0001e-4
    assert np.isclose(z.sum(), 1.0)
    assert np.all(z >= 0)
