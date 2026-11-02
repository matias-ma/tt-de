"""Density averaging preserves probability and commutes with convolution."""

import jax.numpy as jnp
import numpy as np
from scipy.special import roots_legendre
from test_gaussian_convolution import random_tt

from ftde.density_ensemble import SquaredTTMixture
from ftde.gaussian_convolution import GaussianConvolution
from ftde.tt_density import rebuild_tt, tt_params


def test_normalization_gauge_invariance_and_convolution_linearity():
    first, second = random_tt(d=2, rank=2, n=4), random_tt(d=2, rank=2, n=4, seed=11)
    mix = SquaredTTMixture((first, second), (.3, .7))
    t, w = roots_legendre(12)
    xx, yy = np.meshgrid(t, t, indexing='xy')
    points = jnp.asarray(np.column_stack([xx.ravel(), yy.ravel()]))
    density = np.exp(np.asarray(mix.log_density(points))).reshape(xx.shape)
    np.testing.assert_allclose(np.sum(density*w[:, None]*w[None, :]), 1., atol=2e-12)
    np.testing.assert_allclose(mix.pair_grid(0, 1, t, t), density, atol=2e-12)
    flipped = rebuild_tt(first, tuple(-c if k == 0 else c for k, c in enumerate(tt_params(first))))
    np.testing.assert_allclose(SquaredTTMixture((flipped, second), (.3, .7)).log_density(points),
                               mix.log_density(points), atol=2e-12)
    assert np.isneginf(np.asarray(mix.log_density(jnp.array([[2., 0.]])))).all()
    ops = tuple(GaussianConvolution.from_tt(g, [.2, .3]) for g in mix.models)
    y = jnp.array([[.1, .2], [-1.1, .3]])
    # Independent tensor quadrature of the mixture and known test kernel.
    t, w = roots_legendre(60)
    xx, yy = np.meshgrid(t, t, indexing='xy')
    points = jnp.asarray(np.column_stack([xx.ravel(), yy.ravel()]))
    density = np.exp(np.asarray(mix.log_density(points)))
    kernel = np.exp(-.5*np.sum(((np.asarray(y)[:, None]-np.asarray(points))/[.2, .3])**2, axis=-1))/(2*np.pi*.2*.3)
    expected = np.sum(kernel*density*(w[:, None]*w[None, :]).ravel(), axis=1)
    np.testing.assert_allclose(np.exp(np.asarray(mix.convolved_log_density(ops, y))), expected, atol=2e-12)
