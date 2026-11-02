"""General-kernel operator validation, independent of density fitting truth."""

from itertools import product

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from chebttge import convolve_tt
from chebttge.chebtt import ChebTT
from chebttge.cores import ChebyshevCoeffCore
from scipy.special import roots_legendre
from test_gaussian_convolution import random_tt

from ftde.noisy_tt_density import fit_noisy_tt, log_density_noisy_tt, make_noisy_train_step
from ftde.cheb_product import square_tt
import optax
from ftde.tt_convolution import TTConvolution
from ftde.tt_density import rebuild_tt, squared_integral_tt, tt_params


def correlated_polynomial_kernel():
    # U(z) = (1 + .3*(z0/3)*(z1/3))/6**3 on [-3,3]**3.
    a = np.zeros((1, 2, 2)); a[0, 0, 0] = 1; a[0, 1, 1] = 1
    b = np.zeros((2, 2, 2)); b[0, 0, 0] = 1; b[1, 1, 1] = 1
    c = np.zeros((2, 1, 2)); c[0, 0, 0] = 1; c[1, 0, 0] = .3
    return ChebTT(cores=tuple(ChebyshevCoeffCore(jnp.asarray(v/6), (-3., 3.))
                             for v in (a, b, c)), domain=jnp.array([[-3., 3.]]*3))


def test_correlated_non_gaussian_against_independent_cubature():
    g = random_tt(n=7)
    u = correlated_polynomial_kernel()
    op = TTConvolution.from_tt(g, u)
    y = jnp.array([[.1, .2, -.3], [.8, -.7, .4]])
    nodes, weights = roots_legendre(9)
    indices = np.array(list(product(range(9), repeat=3)))
    grid = nodes[indices]
    w = weights[indices].prod(axis=1)
    signal = np.asarray(g(jnp.asarray(grid)))**2
    z = np.asarray(y)[:, None, :] - grid[None, :, :]
    kernel = (1 + .3*z[..., 0]*z[..., 1]/9)/6**3
    expected = np.log((kernel*(w*signal)[None, :]).sum(axis=1)/float(squared_integral_tt(g)))
    np.testing.assert_allclose(op.log_density(g, y), expected, atol=1e-12)
    np.testing.assert_allclose(op.log_density(g, y), log_density_noisy_tt(g, u, y), atol=1e-12)


def test_general_kernel_gradients_and_training():
    g, u = random_tt(n=5), correlated_polynomial_kernel()
    op = TTConvolution.from_tt(g, u)
    y = jnp.array([[.2, -.1, .3], [-.2, .4, .3]])
    params = tt_params(g)
    old = jax.grad(lambda p: log_density_noisy_tt(rebuild_tt(g, p), u, y).mean())(params)
    new = jax.jit(jax.grad(lambda p: op.log_density(rebuild_tt(g, p), y).mean()))(params)
    for a, b in zip(old, new, strict=True):
        np.testing.assert_allclose(a, b, atol=2e-12)
    _, history = fit_noisy_tt(g, None, y, n_steps=5, log_every=0, forward_operator=op)
    assert np.isfinite(history).all()
    assert history[-1] < history[0]


def test_rsi_convolution_training_has_finite_gradients():
    g, u = random_tt(rank=2, n=3), correlated_polynomial_kernel()
    y = jnp.array([[.2, -.1, .3], [-.2, .4, .3]])
    params = tt_params(g)
    selected = convolve_tt(square_tt(g), u, max_rank=8, sketch_size=5, validation_samples=0)
    rows = tuple(stage["pivot_rows"] for stage in selected.diagnostics["stages"])
    def rsi_loss(p):
        return -jnp.mean(log_density_noisy_tt(
            rebuild_tt(g, p), u, y, convolution_method="rsi",
            rsi_max_rank=8, rsi_sketch_size=5, rsi_pivot_rows=rows))
    def naive_loss(p):
        return -jnp.mean(log_density_noisy_tt(rebuild_tt(g, p), u, y))
    value, gradient = jax.jit(jax.value_and_grad(rsi_loss))(params)
    exact_value, exact_gradient = jax.jit(jax.value_and_grad(naive_loss))(params)
    np.testing.assert_allclose(value, exact_value, atol=1e-10)
    for actual, expected in zip(gradient, exact_gradient, strict=True):
        np.testing.assert_allclose(actual, expected, atol=1e-9)
    optimizer = optax.adam(1e-4)
    step = make_noisy_train_step(g, u, optimizer, rsi_max_rank=8,
                                 rsi_sketch_size=5, rsi_pivot_rows=rows)
    updated, _, loss = step(params, optimizer.init(params), y)
    assert np.isfinite(loss)
    assert all(np.isfinite(np.asarray(p)).all() for p in updated)
    _, _, next_loss = step(updated, optimizer.init(updated), y)
    assert np.isfinite(next_loss)


def test_general_operator_domain_and_order_checks():
    g, u = random_tt(n=5), correlated_polynomial_kernel()
    op = TTConvolution.from_tt(g, u)
    assert np.isnan(op.log_density(g, jnp.array([4., 0., 0.])))
    with pytest.raises(ValueError, match="quadrature_order"):
        TTConvolution.from_tt(g, u, quadrature_order=3)
    with pytest.raises(ValueError, match="dimensions"):
        TTConvolution.from_tt(random_tt(d=2), u)


def test_zero_extension_intersection_and_gradient():
    """Small correlated kernel: independent dense integration on each overlap."""
    g = random_tt(d=3, n=5)
    old = correlated_polynomial_kernel()
    # Rescale a normalized polynomial kernel from [-3,3] to [-.4,.4].
    u = ChebTT(cores=tuple(ChebyshevCoeffCore(c.coeffs*7.5, (-.4,.4))
                          for c in old.cores), domain=jnp.array([[-.4,.4]]*3))
    op = TTConvolution.from_tt(g, u, zero_extension=True)
    y = np.array([[.2,-.1,.3], [1.1,-.9,.4], [1.5,0.,0.]])
    t, w = roots_legendre(8)
    indices = np.array(list(product(range(8), repeat=3)))
    expected = []
    for point in y[:2]:
        lo, hi = np.maximum(-1,point-.4), np.minimum(1,point+.4)
        x = lo+(hi-lo)*(t[indices]+1)/2
        weights = np.prod(w[indices]*(hi-lo)/2, axis=1)
        expected.append(np.log(np.sum(weights*np.asarray(g(x))**2*np.asarray(u(point-x)))
                               /float(squared_integral_tt(g))))
    actual = np.asarray(jax.jit(op.log_density)(g, y))
    np.testing.assert_allclose(actual[:2], expected, atol=2e-12)
    assert np.isneginf(actual[2])
    p = tt_params(g)
    loss = lambda p: op.log_density(rebuild_tt(g,p),y[:2]).sum()
    direction = [jnp.ones_like(c)/c.size for c in p]
    gradient = jax.jit(jax.grad(loss))(p)
    step = 1e-5
    finite = (loss([a+step*b for a,b in zip(p,direction)])-
              loss([a-step*b for a,b in zip(p,direction)]))/(2*step)
    np.testing.assert_allclose(sum(jnp.sum(a*b) for a,b in zip(gradient,direction)), finite, atol=1e-7)
    with pytest.raises(ValueError, match='piecewise'):
        op.prepare()


def test_prepared_moments_values_gradients_and_window():
    g, u = random_tt(n=9), correlated_polynomial_kernel()
    direct = TTConvolution.from_tt(g, u)
    prepared = direct.prepare(jnp.array([[-1.5, 1.5]]*3))
    y = jnp.array([[.23, -.71, .49], [1.2, -.8, .31]])
    np.testing.assert_allclose(prepared.log_density(g, y), direct.log_density(g, y), atol=2e-12)
    p = tt_params(g)
    a = jax.grad(lambda p: direct.log_density(rebuild_tt(g, p), y).sum())(p)
    b = jax.grad(lambda p: prepared.log_density(rebuild_tt(g, p), y).sum())(p)
    for aa, bb in zip(a, b, strict=True):
        np.testing.assert_allclose(aa, bb, atol=2e-12)
    assert np.isnan(prepared.log_density(g, jnp.array([1.6, 0., 0.])))
    with pytest.raises(ValueError, match='safe window'):
        direct.prepare(jnp.array([[-5., 5.]]*3))


def test_prepared_high_degree_non_gaussian_kernel():
    g = random_tt(d=5, rank=5, n=33)
    coeff = np.zeros((1, 1, 65)); coeff[0, 0, 0] = 1; coeff[0, 0, 64] = .4
    u = ChebTT(cores=tuple(ChebyshevCoeffCore(jnp.asarray(coeff/6), (-3., 3.)) for _ in range(5)),
               domain=jnp.array([[-3., 3.]]*5))
    direct = TTConvolution.from_tt(g, u)
    prepared = direct.prepare(jnp.array([[-1.5, 1.5]]*5), tolerance=1e-14)
    y = jnp.asarray(np.random.default_rng(8).uniform(-1.4, 1.4, (12, 5)))
    np.testing.assert_allclose(prepared.log_density(g, y), direct.log_density(g, y), atol=3e-11)
