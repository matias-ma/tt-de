"""Numerical representation/noise checks; no clean signal enters selection."""

import sys
from pathlib import Path

import numpy as np
from test_gaussian_convolution import random_tt

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
from cryoem_cache import marginal_noise_covariances
from refine_cryoem_gallery import diagonal_mixture_kernel, expand_tt
from refresh_density_fits import fit_candidate

from ftde.noise_marginals import noise_marginals_from_tt
from ftde.sobolev import AmplitudeSobolevPenalty
from ftde.tt_convolution import TTConvolution
from ftde.tt_density import squared_integral_tt


def test_project_noise_after_inverting_precision():
    covariance = np.array([[2., .3, .8], [.3, 1., .4], [.8, .4, 1.5]])
    precision = np.linalg.inv(covariance)[None]
    actual = marginal_noise_covariances(precision, 2)
    np.testing.assert_allclose(actual[0], covariance[:2, :2], atol=1e-14)
    conditional = np.linalg.inv(precision[0, :2, :2])
    assert np.all(np.diag(actual[0]) > np.diag(conditional))


def test_rank_and_basis_expansion_preserves_density_and_normalizes_gauge():
    model = random_tt(d=4, rank=3, n=7)
    expanded = expand_tt(model, 13, 5, jitter=0)
    points = np.random.default_rng(12).uniform(-1, 1, (40, 4))
    expected = np.asarray(model(points))**2/float(squared_integral_tt(model))
    np.testing.assert_allclose(np.asarray(expanded(points))**2, expected, atol=1e-12)
    np.testing.assert_allclose(squared_integral_tt(expanded), 1., atol=1e-12)
    perturbed = expand_tt(model, 13, 5)
    assert np.isfinite(np.asarray(perturbed(points))).all()
    np.testing.assert_allclose(squared_integral_tt(perturbed), 1., atol=1e-5)


def test_reexpand_rank_larger_than_number_of_basis_modes():
    model = random_tt(d=4, rank=3, n=5)
    wide = expand_tt(model, 5, 8, jitter=0.)
    repeated = expand_tt(wide, 5, 8, jitter=0.)
    points = np.random.default_rng(11).uniform(-1, 1, (30, 4))
    np.testing.assert_allclose(np.asarray(repeated(points))**2, np.asarray(wide(points))**2, atol=1e-12)
    np.testing.assert_allclose(squared_integral_tt(repeated), 1., atol=1e-12)


def test_mixture_uses_common_domain_and_matches_prescribed_noise():
    variance = np.array([[.1, .2, .3], [.4, .2, .05]])
    weights = np.array([.3, .7])
    kernel = diagonal_mixture_kernel(variance, weights, np.array([[-2., 1.], [-1., 2.], [-2., 2.]]))
    points = np.random.default_rng(31).normal(size=(50, 3))
    analytic = sum(w*np.prod(np.exp(-points**2/(2*v))/np.sqrt(2*np.pi*v), axis=1)
                   for v, w in zip(variance, weights, strict=True))
    np.testing.assert_allclose(kernel(points), analytic, atol=1e-12)
    marginals = noise_marginals_from_tt(kernel)
    for k, marginal in enumerate(marginals):
        mean, std = marginal.moments()
        np.testing.assert_allclose([mean, std**2], [0., weights@variance[:, k]], atol=1e-10)


def test_zero_step_rescore_uses_new_kernel_without_changing_density(tmp_path):
    model = random_tt(d=2, rank=2, n=5)
    kernel = diagonal_mixture_kernel(np.array([[.1, .2]]), np.ones(1), model.domain, modes=129)
    observations = np.random.default_rng(2).uniform(-.5, .5, (20, 2))
    fitted = fit_candidate(model, observations[:12], observations[12:], tmp_path,
                           kernel=kernel, steps=0)
    expected = np.asarray(model(observations))**2/float(squared_integral_tt(model))
    actual = np.asarray(fitted(observations))**2/float(squared_integral_tt(fitted))
    np.testing.assert_allclose(actual, expected, atol=1e-12)
    logs = TTConvolution.from_tt(model, kernel).log_density(model, observations[12:])
    np.testing.assert_allclose(np.load(tmp_path/'validation.npy'), logs, atol=1e-11)


def test_sobolev_coordinates_preserve_density_and_convolved_scores(tmp_path):
    model = random_tt(d=2, rank=2, n=7)
    kernel = diagonal_mixture_kernel(np.array([[.1, .2]]), np.ones(1), model.domain, modes=129)
    points = np.random.default_rng(5).uniform(-.8, .8, (16, 2))
    penalty = AmplitudeSobolevPenalty.from_tt(model, order=2)
    fitted = fit_candidate(model, points[:8], points[8:], tmp_path, kernel=kernel,
        steps=0, penalty=penalty, strength=.01, preconditioning='sobolev')
    expected = np.asarray(model(points))**2/float(squared_integral_tt(model))
    actual = np.asarray(fitted(points))**2/float(squared_integral_tt(fitted))
    np.testing.assert_allclose(actual, expected, atol=1e-12)
    logs = TTConvolution.from_tt(model, kernel).log_density(model, points[8:])
    np.testing.assert_allclose(np.load(tmp_path/'validation.npy'), logs, atol=1e-11)
