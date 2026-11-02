"""Gauge- and scale-invariant smoothness of a normalized squared-TT density.

These penalties integrate derivatives of the represented amplitude, including
all other cores. Entrywise core penalties do not have this invariance.
"""

from dataclasses import dataclass

import jax.numpy as jnp
import numpy as np

from .cheb_gram import chebyshev_gram_matrix
from .gaussian_convolution import _transfer


def derivative_grams(template, order=1):
    """Matrices integrating products of physical derivatives of given order."""
    if not isinstance(order, int) or order < 1:
        raise ValueError('derivative order must be a positive integer')
    matrices = []
    for core in template.cores:
        n = core.coeffs.shape[-1]
        a, b = core.domain
        derivative = np.zeros((n, n))
        for k in range(order, n):
            values = np.polynomial.chebyshev.chebder(np.eye(n)[k], m=order)*(2/(b-a))**order
            derivative[k, :len(values)] = values
        gram = np.asarray(chebyshev_gram_matrix(n, (a, b)))
        matrices.append(jnp.asarray(derivative@gram@derivative.T))
    return tuple(matrices)


def amplitude_h1(g, grams, derivatives, weights):
    """sum_k weights[k] integral |partial_k g|^2 / integral |g|^2."""
    norm = jnp.ones((1, 1, 1))
    energy = jnp.zeros_like(norm)
    for c, gram, derivative, w in zip(g.cores, grams, derivatives, weights, strict=True):
        coeffs = c.coeffs/jnp.maximum(jnp.max(jnp.abs(c.coeffs)), 1e-300)
        energy = _transfer(energy, coeffs, gram[None]) + w*_transfer(norm, coeffs, derivative[None])
        norm = _transfer(norm, coeffs, gram[None])
        scale = jnp.maximum(jnp.max(jnp.abs(norm)), 1e-300)
        norm, energy = norm/scale, energy/scale
    return energy[0, 0, 0]/norm[0, 0, 0]


@dataclass(frozen=True)
class AmplitudeSobolevPenalty:
    """Sum of integrated squared amplitude derivatives divided by its norm.

    Construct outside JIT and pass the instance as ``regularizer=`` to
    ``fit_noisy_tt``. Order 1 is one quarter of the interior Fisher information
    when weights are one. Order 2 penalizes curvature more strongly. This
    axiswise seminorm omits mixed derivatives and box-boundary jumps.
    Optional length scales give weights ``scale[k]**(2*order)``; their choice
    and the outer regularization strength are statistical tuning parameters,
    not an exact inverse of a correlated convolution operator.
    """

    grams: tuple
    derivatives: tuple
    weights: tuple
    domains: tuple
    sizes: tuple

    @classmethod
    def from_tt(cls, template, *, order=2, length_scales=None):
        d = len(template.cores)
        scales = np.ones(d) if length_scales is None else np.asarray(length_scales, dtype=float)
        if scales.shape != (d,) or not np.all(np.isfinite(scales) & (scales > 0)):
            raise ValueError(f'length_scales must contain {d} finite positive values')
        derivatives = derivative_grams(template, order)
        domains = tuple(c.domain for c in template.cores)
        sizes = tuple(c.coeffs.shape[-1] for c in template.cores)
        grams = tuple(chebyshev_gram_matrix(n, box) for n, box in zip(sizes, domains, strict=True))
        return cls(grams, derivatives, tuple(scales**(2*order)), domains, sizes)

    def __call__(self, g):
        if (tuple(c.domain for c in g.cores) != self.domains
                or tuple(c.coeffs.shape[-1] for c in g.cores) != self.sizes):
            raise ValueError('regularizer requires the same domains and coefficient counts')
        return amplitude_h1(g, self.grams, self.derivatives, self.weights)
