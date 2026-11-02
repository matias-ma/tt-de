import jax
import jax.numpy as jnp
import numpy as np
from test_gaussian_convolution import random_tt

from ftde.cheb_gram import chebyshev_gram_matrix
from ftde.fisher_visibility import FisherVisibilityPenalty, marginal_fisher_filter
from ftde.tt_density import rebuild_tt, tt_params


def test_gaussian_information_spectrum():
    # For X~N(0,1), Y=X+N(0,sigma²), Hermite scores have information
    # eigenvalues (1+sigma²)^(-m). The finite box omits < 2e-15 mass.
    for sigma in (.5, 1.):
        _, spectrum, diagnostic = marginal_fisher_filter(
            41, (-8., 8.), [0., 0., -16.], sigma, order=320)
        np.testing.assert_allclose(spectrum[-6:][::-1],
                                   (1+sigma**2)**(-np.arange(1., 7.)), rtol=2e-7)
        assert abs(diagnostic['reference_projection_squared_error']) < 1e-10


def test_filter_quadrature_nullspace_and_bounds():
    matrix, spectrum, _ = marginal_fisher_filter(9, (-1., 1.), [0.], .7, order=160)
    refined, refined_spectrum, _ = marginal_fisher_filter(9, (-1., 1.), [0.], .7, order=320)
    np.testing.assert_allclose(matrix, refined, atol=5e-12)
    np.testing.assert_allclose(spectrum, refined_spectrum, atol=5e-12)
    np.testing.assert_allclose(matrix[:, 0], 0., atol=5e-12)
    gram = np.asarray(chebyshev_gram_matrix(9, (-1., 1.)))
    assert np.linalg.eigvalsh(matrix).min() > -1e-12
    assert np.linalg.eigvalsh(gram-matrix).min() > -1e-12


def test_fisher_penalty_gauge_and_gradient():
    g = random_tt(d=2, rank=2, n=7)
    penalty = FisherVisibilityPenalty.from_tt(g, [[0., .1, -.4], [0., -.2, -.6]], [.5, .8])
    params = tt_params(g)
    transform = jnp.array([[2., .4], [.1, .7]])
    changed = [jnp.einsum('abn,bc->acn', params[0], transform)*1e40,
               jnp.einsum('ab,bcn->acn', jnp.linalg.inv(transform), params[1])*1e-60]
    np.testing.assert_allclose(penalty(rebuild_tt(g, changed)), jax.jit(penalty)(g), rtol=1e-12)
    loss = lambda p: penalty(rebuild_tt(g, p))
    gradient = jax.jit(jax.grad(loss))(params)
    direction = [jnp.asarray(np.random.default_rng(k).normal(size=c.shape)) for k, c in enumerate(params)]
    h = 1e-5
    difference = (loss([c+h*v for c, v in zip(params, direction, strict=True)])
                  - loss([c-h*v for c, v in zip(params, direction, strict=True)]))/(2*h)
    np.testing.assert_allclose(sum(jnp.sum(a*b) for a, b in zip(gradient, direction, strict=True)),
                               difference, rtol=1e-7)
