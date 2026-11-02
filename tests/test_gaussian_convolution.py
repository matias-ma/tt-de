"""Operator validation only: known functions never enter estimator training."""

from itertools import pairwise

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from chebttge.chebtt import ChebTT
from chebttge.cores import ChebyshevCoeffCore
from scipy.integrate import quad
from scipy.special import ndtr

from ftde.gaussian_convolution import GaussianConvolution
from ftde.noisy_tt_density import fit_noisy_tt
from ftde.tt_density import rebuild_tt, tt_params


def random_tt(d=3, rank=3, n=7, seed=0):
    rng = np.random.default_rng(seed)
    ranks = (1,) + (rank,) * (d - 1) + (1,)
    cores = tuple(ChebyshevCoeffCore(
        jnp.asarray(rng.normal(size=(a, b, n)) / np.sqrt(a * n)), (-1., 1.))
        for a, b in pairwise(ranks))
    return ChebTT(cores=cores, domain=jnp.array([[-1., 1.]] * d), ranks=ranks[1:-1])


def test_uniform_box_and_outside_observations():
    g = random_tt(d=10, rank=1, n=1)
    std = np.linspace(.05, .4, 10)
    y = np.random.default_rng(1).uniform(-1.4, 1.4, (12, 10))
    op = GaussianConvolution.from_tt(g, std)
    expected = np.log((ndtr((1 - y) / std) - ndtr((-1 - y) / std)) / 2).sum(axis=1)
    np.testing.assert_allclose(op.log_density(g, y), expected, atol=2e-9, rtol=1e-10)


def test_high_degree_against_adaptive_integration():
    g = random_tt(d=1, rank=1, n=65)
    c = np.asarray(g.cores[0].coeffs).ravel()
    norm = quad(lambda x: np.polynomial.chebyshev.chebval(x, c)**2,
                -1, 1, epsabs=1e-11, limit=500)[0]
    std = .04
    y = np.array([[-.99], [-.3], [.2], [1.02]])
    op = GaussianConvolution.from_tt(g, [std])
    expected = [np.log(quad(
        lambda x, v=v: np.polynomial.chebyshev.chebval(x, c)**2
        * np.exp(-.5*((x-v)/std)**2)/(std*np.sqrt(2*np.pi)),
        -1, 1, epsabs=1e-11, limit=500)[0] / norm) for v in y[:, 0]]
    np.testing.assert_allclose(op.log_density(g, y), expected, atol=2e-10)


def test_quadrature_convergence_and_positive_contraction():
    g = random_tt(d=10, rank=8, n=17)
    y = jnp.asarray(np.random.default_rng(3).uniform(-1, 1, (8, 10)))
    ops = [GaussianConvolution.from_tt(g, [.15]*10, quadrature_order=q, positive=p)
           for q, p in [(128, False), (256, False), (128, True)]]
    values = [jax.jit(op.log_density)(g, y) for op in ops]
    np.testing.assert_allclose(values[0], values[1], atol=1e-10)
    np.testing.assert_allclose(values[0], values[2], atol=1e-10)


def test_gradient_and_gauge_invariance():
    g = random_tt()
    op = GaussianConvolution.from_tt(g, [.1, .2, .3])
    y = jnp.asarray([[.2, -.1, .3], [.7, -.2, -.4]])
    params = tt_params(g)
    loss = lambda p: -op.log_density(rebuild_tt(g, p), y).mean()
    grad = jax.jit(jax.grad(loss))(params)
    direction = [jnp.ones_like(c) / c.size for c in params]
    h = 1e-5
    fd = (loss([c+h*v for c, v in zip(params, direction, strict=True)])
          - loss([c-h*v for c, v in zip(params, direction, strict=True)])) / (2*h)
    analytic = sum(jnp.sum(a*b) for a, b in zip(grad, direction, strict=True))
    np.testing.assert_allclose(analytic, fd, atol=1e-7, rtol=1e-6)
    scaled = rebuild_tt(g, [params[0]*1e100, params[1]*1e-100, params[2]])
    np.testing.assert_allclose(op.log_density(g, y), op.log_density(scaled, y), atol=1e-11)


def test_training_uses_only_observations():
    g = random_tt(n=3, rank=2)
    op = GaussianConvolution.from_tt(g, [.2]*3)
    y = jnp.asarray(np.random.default_rng(4).normal(0, .3, (64, 3)))
    _, history = fit_noisy_tt(g, None, y, n_steps=20, lr=.01, log_every=0,
                             gaussian_convolution=op)
    assert np.all(np.isfinite(history))
    assert history[-1] < history[0]


def test_invalid_noise():
    with pytest.raises(ValueError, match="positive"):
        GaussianConvolution.from_tt(random_tt(), [0, .1, .1])


def test_rejects_underresolved_normalization_and_mismatched_domains():
    g = random_tt(n=7)
    with pytest.raises(ValueError, match="number of coefficients"):
        GaussianConvolution.from_tt(g, [.2]*3, quadrature_order=4)
    op = GaussianConvolution.from_tt(g, [.2]*3)
    other = ChebTT(cores=tuple(ChebyshevCoeffCore(c.coeffs, (-2., 2.)) for c in g.cores),
                   domain=jnp.array([[-2., 2.]]*3))
    with pytest.raises(ValueError, match="same domains"):
        op.log_density(other, jnp.zeros(3))
    assert jnp.isnan(jax.jit(op.log_density)(other, jnp.zeros(3)))
