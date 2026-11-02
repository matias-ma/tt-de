"""Synthetic sampling targets for testing the noise-free density estimator."""

from __future__ import annotations

import jax
import jax.numpy as jnp


def sample_correlated_gaussian(key, n: int, d: int = 2, rho: float = 0.85):
    """A correlated Gaussian: exactly rank 1 only after diagonalizing the
    covariance, so in raw coordinates it is only approximately low TT rank.
    """
    cov = jnp.full((d, d), rho) + (1.0 - rho) * jnp.eye(d)
    l = jnp.linalg.cholesky(cov)
    z = jax.random.normal(key, (n, d), dtype=jnp.float64)
    return z @ l.T


def sample_gaussian_mixture(key, n: int, d: int = 2, n_components: int = 3, spread: float = 3.0):
    """A mixture of isotropic Gaussians with random offsets -- multi-modal,
    approximately but not exactly low rank.
    """
    k1, k2, k3, k4 = jax.random.split(key, 4)
    means = jax.random.uniform(k1, (n_components, d), minval=-spread, maxval=spread, dtype=jnp.float64)
    weights = jax.random.dirichlet(k2, jnp.ones(n_components), dtype=jnp.float64)
    comps = jax.random.categorical(k3, jnp.log(weights), shape=(n,))
    noise = jax.random.normal(k4, (n, d), dtype=jnp.float64)
    return means[comps] + noise


def sample_banana(key, n: int, curvature: float = 0.2):
    """Classic 2D 'banana' distribution: strongly non-separable, smooth."""
    k1, k2 = jax.random.split(key)
    x1 = jax.random.normal(k1, (n,), dtype=jnp.float64)
    x2 = jax.random.normal(k2, (n,), dtype=jnp.float64) + curvature * (x1**2 - 1.0)
    return jnp.stack([x1, x2], axis=1)


def add_gaussian_noise(key, clean_samples: jnp.ndarray, cov: jnp.ndarray) -> jnp.ndarray:
    """Add N(0, cov) measurement noise to clean samples, shape (n, d)."""
    n, d = clean_samples.shape
    chol = jnp.linalg.cholesky(cov)
    noise = jax.random.normal(key, (n, d), dtype=jnp.float64) @ chol.T
    return clean_samples + noise
