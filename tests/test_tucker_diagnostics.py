"""Check blocked diagnostics against independent polynomial cubature."""
import numpy as np
from scipy.special import roots_legendre

from ftde.tucker_diagnostics import fourth_power_integral_4d, product_measure_integral
from ftde.tucker_operations import init_tucker_constant


def test_blocked_fourth_and_product_measure():
    model = init_tucker_constant([[-1., 2.]]*4, rank=2, modes=4, perturbation=.4)
    t, w = roots_legendre(8)
    x, w = .5+1.5*t, 1.5*w
    indices = np.indices((8,)*4).reshape(4, -1).T
    values = np.asarray(model(x[indices]))
    expected = np.sum(np.prod(w[indices], axis=1)*values**4)
    np.testing.assert_allclose(fourth_power_integral_4d(model, block_size=11), expected, rtol=1e-12)
    nodes = np.broadcast_to(x, (3, 4, 8))
    weights = np.broadcast_to(w, nodes.shape).copy()
    weights[1] *= 2
    expected = np.sum(np.prod(w[indices], axis=1)*values**2)
    np.testing.assert_allclose(product_measure_integral(model, nodes, weights, batch_size=2),
                               expected*np.array([1, 16, 1]), rtol=1e-12)


def test_native_pair_grid():
    from ftde.marginals import squared_pair_coefficients, squared_pair_grid
    from ftde.tucker_operations import tucker_to_tt
    model = init_tucker_constant([[-1., 2.]]*4, rank=3, modes=5, perturbation=.4)
    tt = tucker_to_tt(model)
    x = np.linspace(-1.2, 2.2, 13)
    for i, j in [(0, 1), (0, 3), (2, 3)]:
        np.testing.assert_allclose(squared_pair_grid(model, i, j, x, x), squared_pair_grid(tt, i, j, x, x), atol=1e-12)
        np.testing.assert_allclose(squared_pair_coefficients(model, i, j)[0], squared_pair_coefficients(tt, i, j)[0], atol=1e-12)


def test_importance_qmc_against_exact_integral():
    from ftde.tucker_density import squared_integral_tucker
    from ftde.tucker_diagnostics import density_squared_integral_qmc
    model = init_tucker_constant([[-1., 1.]]*4, rank=2, modes=3, perturbation=.2)
    observations = np.random.default_rng(24).uniform(-1, 1, (1000, 4))
    record = density_squared_integral_qmc(model, observations, power=12, replicates=4, centers_count=32, bandwidth=.5)
    expected = fourth_power_integral_4d(model)/float(squared_integral_tucker(model))**2
    assert abs(record['value']-expected) < 6*record['standard_error']
    assert record['samples'] == 16384
