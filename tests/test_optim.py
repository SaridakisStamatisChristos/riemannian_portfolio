
import numpy as np
from riemannian_portfolio.core.optim_ng_eg import project_to_simplex, natural_mirror_step

def test_project_to_simplex_basic():
    w = np.array([0.2, 0.5, 0.7])
    w = project_to_simplex(w)
    assert np.all(w >= 0)
    assert abs(w.sum() - 1.0) < 1e-9

def test_natural_mirror_step_shapes():
    w = np.array([0.25, 0.25, 0.5])
    g = np.array([0.1, -0.2, 0.05])
    invF = np.array([10.0, 5.0, 2.0])
    z = natural_mirror_step(w, g, invF, eta=0.1)
    assert z.shape == w.shape
    assert np.all(z >= 0)
    assert abs(z.sum() - 1.0) < 1e-9

def test_project_to_simplex_edgecases():
    w = np.array([-1.0, 0.0, 2.0])
    w = project_to_simplex(w)
    assert np.all(w >= 0)
    assert abs(w.sum() - 1.0) < 1e-9
