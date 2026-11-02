import sys
from pathlib import Path

import jax.numpy as jnp
import numpy as np
import pytest
from scipy.stats import multivariate_normal

from ftde.noise_kernel import build_paired_gaussian_mixture_noise_kernel_tt
from ftde.swirl import (
    sample_swirl,
    swirl_log_density,
    swirl_pair_marginal,
    swirl_squared_norm,
)

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))


def test_swirl_marginal_is_normalized_and_matches_samples():
    xs = np.linspace(-3, 3, 241)
    z = swirl_pair_marginal(6, 2, 3, xs, xs)
    assert abs(z.sum()*(xs[1]-xs[0])**2-1) < 1e-4
    x = sample_swirl(0, 200_000, 6)
    h, _, _ = np.histogram2d(x[:, 2], x[:, 3], bins=24, range=[[-3, 3], [-3, 3]], density=True)
    centers = -3+(np.arange(240)+.5)*.025  # 10x10 sub-cells per histogram bin
    coarse = swirl_pair_marginal(6, 2, 3, centers, centers).reshape(24, 10, 24, 10).mean((1, 3)).T
    assert np.linalg.norm(h-coarse)/np.linalg.norm(coarse) < .05


def test_swirl_density_consistent_with_squared_norm():
    x = sample_swirl(1, 100_000, 4)
    mc = np.exp(swirl_log_density(x))
    assert abs(mc.mean()/swirl_squared_norm(4)-1) < 5*mc.std()/np.sqrt(len(mc))/mc.mean()+1e-3


def test_paired_mixture_kernel_matches_analytic_density():
    covs = np.zeros((2, 4, 4))
    covs[0] = np.eye(4)*.09
    covs[1][:2, :2] = .25*np.array([[1, .6], [.6, 1]])
    covs[1][2:, 2:] = .16*np.array([[1, -.6], [-.6, 1]])
    weights = np.array([.7, .3])
    kernel = build_paired_gaussian_mixture_noise_kernel_tt(covs, weights, np.tile([-2., 2.], (4, 1)))
    assert kernel.ranks[1] == 2
    z = np.random.default_rng(0).normal(size=(300, 4))*.4
    truth = sum(w*multivariate_normal(np.zeros(4), c).pdf(z) for w, c in zip(weights, covs, strict=True))
    assert np.max(np.abs(np.asarray(kernel(jnp.asarray(z)))-truth)) < 1e-9*truth.max()


def test_paired_mixture_kernel_rejects_cross_pair_covariance():
    cov = np.eye(4)[None]*.1
    cov[0, 1, 2] = cov[0, 2, 1] = .01
    with pytest.raises(ValueError, match='adjacent coordinate pairs'):
        build_paired_gaussian_mixture_noise_kernel_tt(cov, np.ones(1), np.tile([-2., 2.], (4, 1)))


def test_microbatched_gradient_matches_full_batch(tmp_path):
    from refresh_density_fits import fit_candidate

    from ftde.init import init_tt_from_kde
    x = sample_swirl(3, 3000, 4)
    domain = np.stack([x.min(0)-.5, x.max(0)+.5], axis=1)
    init = init_tt_from_kde(jnp.asarray(x), max_rank=3, domain=jnp.asarray(domain), n_coeffs=(9,),
                            n_starts=8, gd_steps=0)
    results = []
    for m in (1, 4):
        fit = fit_candidate(init, x[:2000], x[2000:], tmp_path/f'm{m}', steps=3, lr=1e-2,
                            batch_size=256, microbatches=m)
        results.append(np.concatenate([np.ravel(c.coeffs) for c in fit.cores]))
    np.testing.assert_allclose(results[0], results[1], rtol=1e-9, atol=1e-12)
