"""Independent mass, convolution, and gradient checks for joint clipping."""
import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from chebttge.chebtt import ChebTT
from chebttge.cores import ChebyshevCoeffCore
from scipy.special import ndtr

from ftde.clipped_tt_density import ClippedTTQuadrature4D
from ftde.tt_density import rebuild_tt


def model():
    domain = jnp.tile(jnp.array([-1., 1.]), (4, 1))
    cores = [ChebyshevCoeffCore(jnp.array([[[1., 2.]]])/16, (-1., 1.))]
    cores.extend(ChebyshevCoeffCore(jnp.ones((1, 1, 1)), (-1., 1.)) for _ in range(3))
    return ChebTT(cores=tuple(cores), domain=domain)


def test_clipped_mass_marginals_and_support():
    f = model()
    op = ClippedTTQuadrature4D.from_tt(f, np.eye(4)[None]*.09, [1.], order=33)
    weighted, mass, negative = op.weighted_grid(f)
    np.testing.assert_allclose(mass, 1.125, atol=.001)
    np.testing.assert_allclose(negative, .125, atol=.001)
    np.testing.assert_allclose(weighted.sum(), 1., atol=1e-13)
    for i, marginal in enumerate(op.pair_marginals(f)):
        np.testing.assert_allclose(op.weights[2*i+1]@marginal@op.weights[2*i], 1., atol=1e-13)
    np.testing.assert_allclose(op.pair_marginals(f)[1], .25, atol=1e-13)
    p = op.density(f, jnp.array([[-.75, 0, 0, 0], [2., 0, 0, 0], [0., 0, 0, 0]]))
    np.testing.assert_allclose(p[:2], 0.)
    assert p[2] > 0


def test_convolution_matches_truncated_normal_integrals_and_gradient():
    f = model()
    sigmas = np.array([[.3, .4, .5, .6], [.6, .3, .4, .5]])
    mix = np.array([.7, .3])
    op = ClippedTTQuadrature4D.from_tt(f, np.array([np.diag(s*s) for s in sigmas]), mix, order=49)
    y = np.array([[.2, -.1, .3, -.2], [-.7, .3, 0., .1]])
    expected = np.zeros(2)
    for s, weight in zip(sigmas, mix, strict=True):
        a, b = (-.5-y[:, 0])/s[0], (1-y[:, 0])/s[0]
        phi = lambda z: np.exp(-z*z/2)/np.sqrt(2*np.pi)
        first = (1+2*y[:, 0])*(ndtr(b)-ndtr(a))+2*s[0]*(phi(a)-phi(b))
        others = np.prod(ndtr((1-y[:, 1:])/s[1:])-ndtr((-1-y[:, 1:])/s[1:]), axis=1)
        expected += weight*first*others/18
    np.testing.assert_allclose(op.observed_density(f, jnp.asarray(y)), expected, rtol=.002)
    def score(t):
        params = [f.cores[0].coeffs.at[0, 0, 1].set(t), *[c.coeffs for c in f.cores[1:]]]
        return jnp.log(op.observed_density(rebuild_tt(f, params), jnp.asarray(y))).sum()
    analytic = jax.grad(score)(.125)
    h = 1e-6
    numeric = (score(.125+h)-score(.125-h))/(2*h)
    np.testing.assert_allclose(analytic, numeric, rtol=1e-7, atol=1e-8)


def test_rejects_unhandled_cross_pair_noise():
    cov = np.eye(4)
    cov[0, 2] = cov[2, 0] = .1
    with pytest.raises(ValueError, match='block diagonal'):
        ClippedTTQuadrature4D.from_tt(model(), cov[None], [1.])


def test_correlated_noise_mixture_matches_independent_joint_sum():
    f = model()
    covariance = np.array([[[.2, .08, 0, 0], [.08, .3, 0, 0],
                            [0, 0, .4, -.1], [0, 0, -.1, .2]], np.eye(4)*.16])
    weights = np.array([.6, .4])
    op = ClippedTTQuadrature4D.from_tt(f, covariance, weights, order=9)
    weighted, _, _ = op.weighted_grid(f)
    points = np.stack(np.meshgrid(*map(np.asarray, op.nodes), indexing='ij'), axis=-1).reshape(-1, 4)
    y = np.array([[.3, -.2, .1, .4], [-.4, .2, 0., -.1]])
    reference = np.zeros(2)
    for cov, weight in zip(covariance, weights, strict=True):
        delta = y[:, None]-points[None]
        kernel = np.exp(-.5*np.einsum('bni,ij,bnj->bn', delta, np.linalg.inv(cov), delta))
        kernel /= (2*np.pi)**2*np.sqrt(np.linalg.det(cov))
        reference += weight*(kernel@np.asarray(weighted).ravel())
    np.testing.assert_allclose(op.observed_density(f, jnp.asarray(y)), reference, atol=1e-13)


def test_direct_conversion_preserves_a_known_squared_tt():
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
    from initialize_direct_tt_grid import convert

    from ftde.tt_density import density_tt
    first = jnp.array([[[1., 0.], [0., .3]]])
    middle = jnp.zeros((2, 2, 2)).at[0, 0, 0].set(1.).at[1, 1, 1].set(1.)
    last = jnp.array([[[1., 0.]], [[0., 1.]]])
    domain = jnp.tile(jnp.array([-1., 1.]), (4, 1))
    amplitude = ChebTT(cores=tuple(ChebyshevCoeffCore(c, (-1., 1.)) for c in (first, middle, middle, last)), domain=domain)
    direct, report = convert(amplitude, rank=4, modes=5)
    x = jnp.asarray(np.random.default_rng(7).uniform(-1, 1, (100, 4)))
    np.testing.assert_allclose(direct(x), density_tt(amplitude, x), atol=1e-12)
    assert report['total_relative_l2'] < 1e-7
