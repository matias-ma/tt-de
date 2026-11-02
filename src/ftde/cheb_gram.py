"""Closed-form Chebyshev inner-product (Gram) matrices.

These give the exact L2 inner product between Chebyshev basis functions
without any quadrature or resampling, so squared-norm computations built on
top of them stay differentiable and exact with respect to the raw
coefficient arrays stored in ``chebttge``/``chebtuckerge`` cores.
"""

from __future__ import annotations

import jax.numpy as jnp


def _canonical_moments(max_degree: int) -> jnp.ndarray:
    """``moments[k] = integral_{-1}^{1} T_k(t) dt`` for ``k = 0..max_degree``."""
    k = jnp.arange(max_degree + 1, dtype=jnp.float64)
    safe_denom = jnp.where(k == 1.0, 1.0, 1.0 - k**2)
    even = jnp.where(k == 0.0, 2.0, 2.0 / safe_denom)
    return jnp.where((k % 2.0) == 0.0, even, 0.0)


def chebyshev_moment_vector(n_coeffs: int, domain: tuple[float, float]) -> jnp.ndarray:
    """Exact ``moments[i] = integral_a^b T_i(t(x)) dx`` (the affine-mapped
    physical-domain moments), same closed form used by chebttge's own
    ``integrate()``/``_integral_matrix``.
    """
    a, b = domain
    return 0.5 * (b - a) * _canonical_moments(n_coeffs - 1)


def chebyshev_gram_matrix(n_coeffs: int, domain: tuple[float, float]) -> jnp.ndarray:
    """Exact Gram matrix ``G[i, j] = integral_a^b T_i(t(x)) T_j(t(x)) dx``.

    Uses ``T_i T_j = (T_{i+j} + T_{|i-j|}) / 2`` and the closed-form moments of
    individual Chebyshev polynomials on the canonical interval, then applies
    the affine-map Jacobian ``(b - a) / 2`` for the physical domain.
    """
    a, b = domain
    n = jnp.arange(n_coeffs)
    moments = _canonical_moments(2 * (n_coeffs - 1))
    plus = moments[n[:, None] + n[None, :]]
    minus = moments[jnp.abs(n[:, None] - n[None, :])]
    return 0.25 * (b - a) * (plus + minus)
