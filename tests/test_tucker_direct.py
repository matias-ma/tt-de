"""Independent quadrature and TT equivalence for native Tucker fitting."""
from dataclasses import replace
from itertools import product

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from scipy.special import roots_legendre
from test_tt_convolution import correlated_polynomial_kernel

from ftde import (
    TuckerConvolution,
    TuckerTrainingProblem,
    init_tucker_constant,
    log_density_noisy_tucker,
    squared_integral_tucker,
    squared_marginal_tucker,
)
from ftde.direct_training import DirectTrainingProblem
from ftde.tt_convolution import TTConvolution
from ftde.tucker_density import rebuild_tucker, tucker_params
from ftde.tucker_operations import (
    expand_tucker_rank,
    load_tucker,
    save_tucker,
    tucker_to_tt,
)


def model():
    return init_tucker_constant([[-1., 1.]]*3, rank=2, modes=5, perturbation=.3)


def test_rank_expansion_preserves_function_and_adds_trainable_directions():
    g = model()
    expanded = expand_tucker_rank(g, (3, 4, 5))
    x = jnp.asarray(np.random.default_rng(18).uniform(-1, 1, (13, 3)))
    np.testing.assert_allclose(expanded(x), g(x), atol=1e-13)
    np.testing.assert_allclose(squared_integral_tucker(expanded), squared_integral_tucker(g), atol=1e-13)
    from ftde.cheb_gram import chebyshev_gram_matrix
    for factor in expanded.factors:
        c = np.asarray(factor.core.coeffs[0])
        gram = np.asarray(chebyshev_gram_matrix(c.shape[-1], factor.domain))
        np.testing.assert_allclose(c@gram@c.T, np.eye(len(c)), atol=1e-13)
    problem = TuckerTrainingProblem(expanded)
    gradient = jax.grad(problem.loss)(problem.params, problem.features(x))[0]
    assert np.linalg.norm(np.asarray(gradient)[2:]) > 1e-6
    with pytest.raises(ValueError, match='Expanded ranks'):
        expand_tucker_rank(g, 1)
    with pytest.raises(ValueError, match='max_core_entries'):
        expand_tucker_rank(g, 5, max_core_entries=10)


def test_parallel_chebyshev_basis_and_endpoint_gradients():
    from ftde.tucker_convolution import _chebyshev_basis
    t = np.array([-1., -.99999, -.7, 0., .6, .99999, 1.])
    actual = jax.jit(lambda x: _chebyshev_basis(x, 513))(jnp.asarray(t))
    np.testing.assert_allclose(actual, np.polynomial.chebyshev.chebvander(t, 512), atol=2e-11)
    c = np.random.default_rng(2).normal(size=513)/(1+np.arange(513)**2)
    gradient = jax.jit(jax.grad(lambda x: jnp.sum(_chebyshev_basis(x, 513)@c)))(jnp.asarray(t))
    expected = np.polynomial.chebyshev.chebval(t, np.polynomial.chebyshev.chebder(c))
    np.testing.assert_allclose(gradient, expected, rtol=1e-10, atol=1e-10)


@pytest.mark.parametrize('dimension', [4, 5])
def test_multiple_pair_contractions(dimension):
    from ftde.simple_initialization import init_tt_constant
    g = init_tucker_constant([[-1., 1.]]*dimension, rank=2, modes=4, perturbation=.2)
    kernel = init_tt_constant([[-2., 2.]]*dimension, rank=3, modes=5, perturbation=.001)
    pair = TuckerConvolution.from_tucker(g, kernel).prepare()
    sequential = replace(pair, pair_axes=False)
    y = jnp.asarray(np.random.default_rng(3).uniform(-.5, .5, (3, dimension)))
    params = tucker_params(g)
    fp = lambda p: pair.log_density(rebuild_tucker(g, p), y).sum()
    fs = lambda p: sequential.log_density(rebuild_tucker(g, p), y).sum()
    a, ga = jax.jit(jax.value_and_grad(fp))(params)
    b, gb = jax.jit(jax.value_and_grad(fs))(params)
    np.testing.assert_allclose(a, b, atol=1e-12)
    for x, z in zip(jax.tree.leaves(ga), jax.tree.leaves(gb), strict=True):
        np.testing.assert_allclose(x, z, atol=1e-12)


@pytest.mark.parametrize('dimension', [4, 6])
def test_mixture_pair_blocks_accumulate_values_and_gradients(dimension):
    from chebttge.chebtt import ChebTT
    from chebttge.cores import ChebyshevCoeffCore
    ranks, offsets = [1, 2, 3], [0, 1, 3]
    cores = []
    for k in range(dimension):
        left = 1 if k == 0 else (6 if k % 2 else 3)
        right = 1 if k == dimension-1 else (3 if k % 2 else 6)
        c = np.zeros((left, right, 3))
        for component in range(3):
            if k % 2 == 0:
                weight = 1/(3*4**dimension) if k == 0 else 1.
                row = 0 if k == 0 else component
                c[row, offsets[component], 0] = weight
                for extra in range(1, ranks[component]):
                    c[row, offsets[component]+extra, extra] = weight*.03*(component+1)/extra
            else:
                column = 0 if k == dimension-1 else component
                c[offsets[component], column, 0] = 1
                for extra in range(1, ranks[component]):
                    c[offsets[component]+extra, column, extra] = 1
        cores.append(ChebyshevCoeffCore(jnp.asarray(c), (-2., 2.)))
    kernel = ChebTT(cores=tuple(cores), domain=jnp.asarray([[-2., 2.]]*dimension))
    g = init_tucker_constant([[-1., 1.]]*dimension, rank=2, modes=4, perturbation=.4)
    grouped = TuckerConvolution.from_tucker(g, kernel).prepare()
    assert all(len(grouped.pair_groups[k]) == 3 for k in range(0, dimension-1, 2))
    dense = replace(grouped, pair_groups=None)
    sequential = replace(grouped, pair_axes=False)
    y = jnp.asarray(np.random.default_rng(82).uniform(-.5, .5, (3, dimension)))
    params = tucker_params(g)
    values, gradients = [], []
    for op in [grouped, dense, sequential]:
        function = lambda p, op=op: op.log_density(rebuild_tucker(g, p), y).sum()
        value, gradient = jax.jit(jax.value_and_grad(function))(params)
        values.append(value)
        gradients.append(jax.tree.leaves(gradient))
    np.testing.assert_allclose(values, values[0], atol=1e-12)
    for reference, *others in zip(*gradients, strict=True):
        for actual in others:
            np.testing.assert_allclose(actual, reference, atol=1e-12)


def test_conversion_normalization_and_energies(tmp_path):
    g = model()
    tt = tucker_to_tt(g)
    x = jnp.asarray(np.random.default_rng(1).uniform(-1, 1, (13, 3)))
    np.testing.assert_allclose(g(x), tt(x), atol=1e-13)
    np.testing.assert_allclose(squared_integral_tucker(g), 1., atol=1e-13)
    a, b = TuckerTrainingProblem(g, marginal_strength=.2), DirectTrainingProblem(tt, marginal_strength=.2)
    np.testing.assert_allclose(a.energies(a.params), b.energies(b.params), atol=1e-13)
    np.testing.assert_allclose(a.loss(a.params, a.features(x)), b.loss(b.params, b.features(x)), atol=1e-13)
    save_tucker(tmp_path/'model.npz', g)
    np.testing.assert_array_equal(load_tucker(tmp_path/'model.npz').core, g.core)


def test_convolution_quadrature_preparation_and_gradients():
    g, u = model(), correlated_polynomial_kernel()
    op = TuckerConvolution.from_tucker(g, u)
    prepared = op.prepare()
    y = jnp.array([[.1, .2, -.3], [.8, -.7, .4]])
    x, w = roots_legendre(8)
    idx = np.array(list(product(range(8), repeat=3)))
    grid = x[idx]
    weights = w[idx].prod(axis=1)
    kernel = np.asarray(u((y[:, None, :]-grid[None]).reshape(-1, 3))).reshape(len(y), -1)
    expected = np.log(np.sum(kernel*(weights*np.asarray(g(grid))**2), axis=1))
    np.testing.assert_allclose(op.log_density(g, y), expected, atol=1e-12)
    np.testing.assert_allclose(prepared.log_density(g, y), expected, atol=1e-12)
    tt = tucker_to_tt(g)
    np.testing.assert_allclose(TTConvolution.from_tt(tt, u).log_density(tt, y), expected, atol=1e-12)
    params = tucker_params(g)
    f = lambda p: op.log_density(rebuild_tucker(g, p), y).sum()
    fp = lambda p: prepared.log_density(rebuild_tucker(g, p), y).sum()
    ga, gb = jax.jit(jax.grad(f))(params), jax.jit(jax.grad(fp))(params)
    for a, b in zip(jax.tree.leaves(ga), jax.tree.leaves(gb), strict=True):
        np.testing.assert_allclose(a, b, atol=1e-12)
    sequential = replace(prepared, pair_axes=False)
    fs = lambda p: sequential.log_density(rebuild_tucker(g, p), y).sum()
    np.testing.assert_allclose(fp(params), fs(params), atol=1e-12)
    for a, b in zip(jax.tree.leaves(gb), jax.tree.leaves(jax.jit(jax.grad(fs))(params)), strict=True):
        np.testing.assert_allclose(a, b, atol=1e-12)
    delta = jax.tree.map(lambda a: jnp.ones_like(a)*.1, params)
    numerical = (f(jax.tree.map(lambda a, d: a+1e-5*d, params, delta))-f(jax.tree.map(lambda a, d: a-1e-5*d, params, delta)))/2e-5
    analytic = sum(jnp.sum(a*d) for a, d in zip(jax.tree.leaves(ga), jax.tree.leaves(delta), strict=True))
    np.testing.assert_allclose(analytic, numerical, atol=1e-9)
    np.testing.assert_allclose(log_density_noisy_tucker(g, None, y, forward_operator=prepared), expected, atol=1e-12)
    assert np.isnan(op.log_density(g, jnp.array([4., 0., 0.])))
    with pytest.raises(ValueError, match='quadrature_order'):
        TuckerConvolution.from_tucker(g, u, quadrature_order=1)


def test_marginal_independent_quadrature():
    g = model()
    x, w = roots_legendre(8)
    points = np.array([[.2, .3], [-.4, .1]])
    grid = np.array([[a, t, b] for a, b in points for t in x])
    expected = (np.asarray(g(grid)).reshape(2, -1)**2)@w
    np.testing.assert_allclose(squared_marginal_tucker(g, points, (0, 2)), expected, atol=1e-13)


@pytest.mark.parametrize('noisy', [False, True])
def test_training_and_mode_schedule(tmp_path, noisy):
    g = model()
    op = TuckerConvolution.from_tucker(g, correlated_polynomial_kernel()).prepare() if noisy else None
    problem = TuckerTrainingProblem(g, forward=op)
    x = np.random.default_rng(2).uniform(-.5, .5, (32, 3))
    fitted, record = problem.fit_adam(x, x, tmp_path, steps=4, batch_size=8, checkpoint_every=2,
                                      mode_schedule=[(0, 3), (2, 5)], microbatches=2)
    assert np.isfinite(record['best_nll'])
    assert record['history'][0]['inactive_coefficient_max'] == 0
    assert fitted.core.shape == g.core.shape


def test_initializer_capacity_guard():
    with pytest.raises(ValueError, match='max_core_entries'):
        init_tucker_constant([[-1., 1.]]*20, rank=5)


def test_native_tucker_kernel_and_legacy_convolution_gradients():
    from chebtuckerge.chebtucker import ChebTucker
    from chebtuckerge.factors import ChebyshevCoeffFactor

    g = model()
    core = jnp.zeros((2, 2, 2)).at[0, 0, 0].set(1.).at[1, 1, 0].set(.3)
    u = ChebTucker(core, [ChebyshevCoeffFactor(jnp.eye(2)/6, (-3., 3.)) for _ in range(3)], jnp.array([[-3., 3.]]*3))
    op = TuckerConvolution.from_tucker(g, u).prepare()
    y = jnp.array([[.1, -.2, .3], [.4, .5, -.3]])
    params = tucker_params(g)
    reference = lambda p: log_density_noisy_tucker(rebuild_tucker(g, p), u, y).sum()
    direct = lambda p: op.log_density(rebuild_tucker(g, p), y).sum()
    a, b = jax.value_and_grad(reference)(params), jax.jit(jax.value_and_grad(direct))(params)
    for x, y in zip(jax.tree.leaves(a), jax.tree.leaves(b), strict=True):
        np.testing.assert_allclose(x, y, atol=2e-12)
    from ftde import fit_noisy_tucker
    _, history = fit_noisy_tucker(g, None, jnp.zeros((4, 3)), n_steps=3, log_every=0, forward_operator=op)
    assert np.isfinite(history).all()
