import jax
import jax.numpy as jnp
import numpy as np
from scipy.special import roots_legendre
from test_gaussian_convolution import random_tt

from ftde.reference_score import ReferenceScorePenalty
from ftde.sobolev import AmplitudeSobolevPenalty
from ftde.tt_density import rebuild_tt, tt_params


def test_reference_score_against_independent_cubature_and_gauge():
    g = random_tt(d=2, rank=3, n=6)
    log_coeffs = np.array([[3., .2, -.5, .1], [-2., -.1, -.8, -.2]])
    scales = np.array([.3, .8])
    penalty = ReferenceScorePenalty.from_tt(g, log_coeffs, length_scales=scales)
    nodes, w = roots_legendre(20)
    xx, yy = np.meshgrid(nodes, nodes, indexing='ij')
    grid = jnp.asarray(np.stack([xx.ravel(), yy.ravel()], axis=1))
    weights = (w[:, None]*w[None, :]).ravel()
    values = np.asarray(g(grid))
    derivatives = np.asarray(jax.vmap(jax.grad(lambda x: g(x[None])[0]))(grid))
    score = np.stack([np.polynomial.chebyshev.chebval(np.asarray(grid)[:, k],
                     np.polynomial.chebyshev.chebder(c)) for k, c in enumerate(log_coeffs)], axis=1)
    expected = np.sum(weights*np.sum(scales**2*(derivatives-.5*values[:, None]*score)**2, axis=1))/np.sum(weights*values**2)
    np.testing.assert_allclose(jax.jit(penalty)(g), expected, rtol=1e-12)
    params = tt_params(g)
    matrix = jnp.array([[2., .4, .1], [.1, .7, .2], [.2, .1, 1.3]])
    changed = [jnp.einsum('abn,bc->acn', params[0], matrix)*1e40,
               jnp.einsum('ab,bcn->acn', jnp.linalg.inv(matrix), params[1])*1e-60]
    np.testing.assert_allclose(penalty(rebuild_tt(g, changed)), expected, rtol=1e-12)
    other = log_coeffs.copy()
    other[:, 0] += 100.
    np.testing.assert_allclose(ReferenceScorePenalty.from_tt(g, other, length_scales=scales)(g), expected, rtol=1e-12)


def test_constant_reference_and_coefficient_gradient():
    g = random_tt(d=2, rank=2, n=5)
    penalty = ReferenceScorePenalty.from_tt(g, [np.zeros(1)]*2)
    np.testing.assert_allclose(penalty(g), AmplitudeSobolevPenalty.from_tt(g, order=1)(g), rtol=1e-12)
    params = tt_params(g)
    loss = lambda p: penalty(rebuild_tt(g, p))
    gradient = jax.jit(jax.grad(loss))(params)
    direction = [jnp.asarray(np.random.default_rng(k).normal(size=c.shape)) for k, c in enumerate(params)]
    h = 1e-5
    finite_difference = (loss([c+h*v for c, v in zip(params, direction, strict=True)])
                         - loss([c-h*v for c, v in zip(params, direction, strict=True)]))/(2*h)
    np.testing.assert_allclose(sum(jnp.sum(a*b) for a, b in zip(gradient, direction, strict=True)), finite_difference, rtol=1e-7)
