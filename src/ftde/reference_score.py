"""Relative Fisher regularization around a data-derived product reference.

The reference is specified by univariate polynomial log densities. Only its
score is needed, so additive log normalizers cancel. The trained density
remains an ordinary squared ChebTT on the original physical coordinates.
"""

from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np
from scipy.special import roots_legendre

from .cheb_gram import chebyshev_gram_matrix
from .sobolev import amplitude_h1


@dataclass(frozen=True, eq=False)
class ReferenceScorePenalty:
    """One quarter of weighted interior relative Fisher divergence.

    R(g) = sum_k s_k² integral (partial_k g - g*partial_k log rho/2)² / Z.
    Here rho is the product reference, not a prescribed signal truth. This
    penalizes dependence and marginal deviations softly; the full noisy
    likelihood still uses the prescribed correlated noise model.
    """

    grams: tuple
    derivatives: tuple
    weights: tuple
    domains: tuple
    sizes: tuple

    @classmethod
    def from_tt(cls, template, log_coeffs, *, length_scales=None):
        domains = tuple(c.domain for c in template.cores)
        sizes = tuple(c.coeffs.shape[-1] for c in template.cores)
        coefficients = tuple(np.asarray(c, dtype=float) for c in log_coeffs)
        if len(coefficients) != len(sizes) or any(c.ndim != 1 or not len(c) or not np.isfinite(c).all() for c in coefficients):
            raise ValueError('one finite log-density coefficient vector per axis is required')
        scales = np.ones(len(sizes)) if length_scales is None else np.asarray(length_scales, dtype=float)
        if scales.shape != (len(sizes),) or not np.all(np.isfinite(scales) & (scales > 0)):
            raise ValueError('length_scales must be finite and positive')
        grams, penalties = [], []
        for n, box, coeff in zip(sizes, domains, coefficients, strict=True):
            a, b = box
            # Integrand is a polynomial: resolve (T_i' - score*T_i/2)^2.
            t, w = roots_legendre(n+len(coeff))
            w = w*(b-a)/2
            basis = np.polynomial.chebyshev.chebvander(t, n-1)
            derivative = np.stack([np.polynomial.chebyshev.chebval(
                t, np.polynomial.chebyshev.chebder(np.eye(n)[k]))*2/(b-a) for k in range(n)], axis=1)
            score = np.polynomial.chebyshev.chebval(t, np.polynomial.chebyshev.chebder(coeff))*2/(b-a)
            residual = derivative-.5*score[:, None]*basis
            penalties.append(jnp.asarray(residual.T@(w[:, None]*residual)))
            grams.append(chebyshev_gram_matrix(n, box))
        return cls(tuple(grams), tuple(penalties), tuple(scales**2), domains, sizes)

    def __call__(self, g):
        if tuple(c.coeffs.shape[-1] for c in g.cores) != self.sizes:
            raise ValueError('reference penalty requires the same polynomial sizes')
        matches = jnp.all(jnp.asarray([c.domain for c in g.cores]) == jnp.asarray(self.domains))
        if not isinstance(matches, jax.core.Tracer) and not bool(matches):
            raise ValueError('reference penalty requires the same domains')
        return jnp.where(matches, amplitude_h1(g, self.grams, self.derivatives, self.weights), jnp.nan)
