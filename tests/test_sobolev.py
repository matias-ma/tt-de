import jax
import jax.numpy as jnp
import numpy as np
from scipy.special import roots_legendre
from test_gaussian_convolution import random_tt

from ftde.cheb_gram import chebyshev_gram_matrix
from ftde.sobolev import AmplitudeSobolevPenalty, amplitude_h1, derivative_grams
from ftde.tt_density import rebuild_tt, tt_params


def test_amplitude_h1_against_independent_derivatives_and_cubature():
    g = random_tt(d=2, rank=2, n=5)
    nodes, w = roots_legendre(12)
    x, y = np.meshgrid(nodes, nodes, indexing='ij')
    grid = jnp.asarray(np.stack([x.ravel(), y.ravel()], axis=1))
    weights = (w[:, None]*w[None, :]).ravel()
    derivatives = np.asarray(jax.vmap(jax.grad(lambda x: g(x[None])[0]))(grid))
    signal = np.asarray(g(grid))
    axis_weights = jnp.array([.2, .6])
    expected = np.sum(weights*np.sum(derivatives**2*np.asarray(axis_weights), axis=1))/np.sum(weights*signal**2)
    grams = tuple(chebyshev_gram_matrix(5, c.domain) for c in g.cores)
    actual = amplitude_h1(g, grams, derivative_grams(g), axis_weights)
    np.testing.assert_allclose(actual, expected, rtol=1e-12)


def test_curvature_against_hessian_cubature_on_physical_box():
    from chebttge.chebtt import ChebTT
    from chebttge.cores import ChebyshevCoeffCore

    random = random_tt(d=2, rank=2, n=5)
    boxes = ((-2., 3.), (-.4, .8))
    g = ChebTT(cores=tuple(ChebyshevCoeffCore(c.coeffs, box)
                          for c, box in zip(random.cores, boxes, strict=True)), domain=jnp.array(boxes))
    nodes, w = roots_legendre(12)
    axes = [a+(b-a)*(nodes+1)/2 for a, b in boxes]
    xx, yy = np.meshgrid(*axes, indexing='ij')
    grid = jnp.asarray(np.stack([xx.ravel(), yy.ravel()], axis=1))
    weights = (w[:, None]*w[None, :]).ravel()
    diagonal = np.asarray(jax.vmap(jax.hessian(lambda x: g(x[None])[0]))(grid)).diagonal(axis1=1, axis2=2)
    scales = np.array([.3, .7])
    expected = np.sum(weights*np.sum(diagonal**2*scales**4, axis=1))/np.sum(weights*np.asarray(g(grid))**2)
    penalty = AmplitudeSobolevPenalty.from_tt(g, order=2, length_scales=scales)
    np.testing.assert_allclose(jax.jit(penalty)(g), expected, rtol=2e-12)


def test_sobolev_gauge_scale_invariance_and_gradient():
    g = random_tt(d=3, rank=3, n=5)
    penalty = AmplitudeSobolevPenalty.from_tt(g)
    params = tt_params(g)
    # A nonorthogonal bond transformation and its inverse leave g unchanged.
    matrix = jnp.array([[2., .4, .1], [.1, .7, .2], [.2, .1, 1.3]])
    changed = [jnp.einsum('abn,bc->acn', params[0], matrix),
               jnp.einsum('ab,bcn->acn', jnp.linalg.inv(matrix), params[1]), params[2]]
    np.testing.assert_allclose(penalty(rebuild_tt(g, changed)), penalty(g), rtol=1e-12)
    for scales in ((.01, .01, .01), (1e100, 1e-100, -3.)):
        np.testing.assert_allclose(penalty(rebuild_tt(g, [c*s for c, s in zip(params, scales, strict=True)])),
                                   penalty(g), rtol=1e-12)
    loss = lambda p: penalty(rebuild_tt(g, p))
    grad = jax.jit(jax.grad(loss))(params)
    direction = [jnp.asarray(np.random.default_rng(k).normal(size=c.shape)) for k, c in enumerate(params)]
    analytic = sum(jnp.sum(a*b) for a, b in zip(grad, direction, strict=True))
    h = 1e-5
    finite_difference = (loss([c+h*v for c, v in zip(params, direction, strict=True)])
                         - loss([c-h*v for c, v in zip(params, direction, strict=True)]))/(2*h)
    np.testing.assert_allclose(analytic, finite_difference, rtol=1e-7)
    # Euler derivative in a pure scale direction must vanish.
    np.testing.assert_allclose(jnp.sum(grad[0]*params[0]), 0., atol=1e-9)


def test_custom_regularizer_training_and_loss():
    from ftde.gaussian_convolution import GaussianConvolution
    from ftde.noisy_tt_density import fit_noisy_tt, nll_noisy_tt

    g = random_tt(d=2, rank=2, n=4)
    op = GaussianConvolution.from_tt(g, [.2, .3])
    penalty = AmplitudeSobolevPenalty.from_tt(g)
    y = jnp.asarray(np.random.default_rng(4).normal(0., .3, (64, 2)))
    params = tt_params(g)
    base = nll_noisy_tt(params, g, None, y, forward_operator=op)
    actual = nll_noisy_tt(params, g, None, y, forward_operator=op,
                          reg_strength=.01, regularizer=penalty)
    np.testing.assert_allclose(actual-base, .01*penalty(g), rtol=1e-12)
    _, history = fit_noisy_tt(g, None, y, forward_operator=op, reg_strength=.01,
                              regularizer=penalty, n_steps=20, lr=.01, log_every=0)
    assert np.isfinite(history).all()
    assert history[-1] < history[0]
