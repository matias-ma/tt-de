"""Independent identities needed by the noisy-estimator diagnostics."""
import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from scipy.special import erf, roots_legendre

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
from check_noisy_stability import frequency_geometry
from noise_convolution_audit import analytic_pair_reference
from noise_diagnosis import GaussianMomentProblem

from ftde.gaussian_convolution import GaussianConvolution
from ftde.simple_initialization import init_tt_constant
from ftde.tt_density import log_density_tt


def test_cached_moments_and_gradients_match_direct_quadrature():
    model = init_tt_constant(np.tile([-2., 2.], (4, 1)), rank=3, modes=7, perturbation=.5)
    points = np.random.default_rng(3).normal(size=(9, 4))
    std = np.array([.4, .5, .6, .7])
    problem = GaussianMomentProblem(model, std, quadrature_order=64)
    features = problem.features(points)
    direct = GaussianConvolution.from_tt(model, std, quadrature_order=64, positive=True)
    cached = lambda p: jnp.mean(problem.logs(p, features))
    reference = lambda p: jnp.mean(direct.log_density(problem.model(p), points))
    np.testing.assert_allclose(cached(problem.params), reference(problem.params), atol=1e-12)
    for a, b in zip(jax.grad(cached)(problem.params), jax.grad(reference)(problem.params), strict=True):
        np.testing.assert_allclose(a, b, atol=2e-11, rtol=2e-10)


def test_pair_reference_matches_diagonal_gaussian():
    model = init_tt_constant(np.tile([-2., 2.], (4, 1)), rank=3, modes=7, perturbation=.5)
    points = np.random.default_rng(7).normal(size=(4, 4))
    std = np.array([.4, .5, .6, .7])
    reference = analytic_pair_reference(model, points, np.ones(1), np.diag(std**2)[None], 64)
    direct = GaussianConvolution.from_tt(model, std, quadrature_order=64, positive=True)
    np.testing.assert_allclose(reference(tuple(c.coeffs for c in model.cores)), direct.log_density(model, points), atol=1e-11)


def test_correlated_reference_matches_dense_four_dimensional_cubature():
    model = init_tt_constant(np.tile([-2., 2.], (4, 1)), rank=2, modes=5, perturbation=.8)
    points = np.array([[.3, -.4, .2, .8], [-.6, .2, .9, -.1]])
    weights = np.array([.3, .7])
    covariance = np.tile(np.eye(4)[None], (2, 1, 1))
    covariance[0, 0, 1] = covariance[0, 1, 0] = .55
    covariance[0, 2, 3] = covariance[0, 3, 2] = -.4
    covariance[1] *= .6
    covariance[1, 0, 1] = covariance[1, 1, 0] = -.2
    covariance[1, 2, 3] = covariance[1, 3, 2] = .25
    order = 16
    t, w = roots_legendre(order)
    x = np.stack(np.meshgrid(*[2*t]*4, indexing='ij'), axis=-1).reshape(-1, 4)
    volume = np.prod(np.stack(np.meshgrid(*[2*w]*4, indexing='ij'), axis=-1), axis=-1).ravel()
    density = np.exp(np.asarray(log_density_tt(model, jnp.asarray(x))))
    answer = np.zeros(len(points))
    for weight, cov in zip(weights, covariance, strict=True):
        delta = points[:, None]-x[None]
        kernel = np.exp(-.5*np.einsum('bni,ij,bnj->bn', delta, np.linalg.inv(cov), delta))
        answer += weight*(kernel@(density*volume))/((2*np.pi)**2*np.sqrt(np.linalg.det(cov)))
    reference = analytic_pair_reference(model, points, weights, covariance, order)
    np.testing.assert_allclose(np.exp(reference(tuple(c.coeffs for c in model.cores))), answer, rtol=2e-12)


def test_parseval_geometry_against_analytic_blurred_constant():
    box = np.array([[-1., 1.], [-1.5, 1.5]])
    std = np.array([.22, .5])
    transform, measure = frequency_geometry(box, (1, 1), np.ones(1), np.diag(std**2)[None], (0, 1))
    widths = np.ptp(box, axis=1)
    characteristic = transform[1]@np.array([[1/np.prod(widths)]])@transform[0].T
    numerical = np.sum(np.abs(characteristic)**2*measure)
    norms = (widths*erf(widths/(2*std))-2*std/np.sqrt(np.pi)*(1-np.exp(-widths**2/(4*std**2))))/widths**2
    np.testing.assert_allclose(numerical, np.prod(norms), rtol=2e-12)
