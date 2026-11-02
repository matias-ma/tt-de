"""Exact curvature of the normalized one-coordinate density marginals.

This penalizes density (not amplitude) oscillations without constructing the
rank-squared density TT. It cannot control interactions invisible in 1D.
"""
import jax.numpy as jnp
import numpy as np
from scipy.special import roots_legendre

from .gaussian_convolution import _transfer


def marginal_curvature_geometry(sizes, domain, inverse, length_scales):
    """Polynomial quadrature exact for the squared second density derivative."""
    geometry = []
    for n, box, inv, scale in zip(sizes, domain, inverse, length_scales, strict=True):
        x, w = roots_legendre(max(1, 2*n-1))
        width = np.ptp(box)
        matrices = []
        for order in range(3):
            columns = [np.polynomial.chebyshev.chebval(
                x, np.polynomial.chebyshev.chebder(np.eye(n)[k], m=order))
                for k in range(n)]
            matrices.append(jnp.asarray(np.stack(columns, axis=1) @ inv.T * (2/width)**order))
        geometry.append((*matrices, jnp.asarray(w*width/2 * width*scale**4)))
    return tuple(geometry)


def marginal_curvature(cores, grams, geometry):
    """Sum_k width_k scale_k^4 integral (p_k''(x))^2 dx.

    The width factor makes the penalty dimensionless and coordinate-scale
    invariant when length scales follow the box. TT gauges and overall
    amplitude scaling cancel through exact normalization.
    """
    left = [jnp.ones((1, 1))]
    for c, g in zip(cores, grams, strict=True):
        left.append(_transfer(left[-1][None], c, g[None])[0])
    norm = left[-1][0, 0]
    right = [None]*(len(cores)+1)
    right[-1] = jnp.ones((1, 1))
    for k in range(len(cores)-1, -1, -1):
        right[k] = _transfer(right[k+1][None], cores[k].transpose(1, 0, 2), grams[k][None])[0]
    result = jnp.asarray(0.)
    for k, (c, (b, db, ddb, w)) in enumerate(zip(cores, geometry, strict=True)):
        h = jnp.einsum('ab,ain,ij,bjm->nm', left[k], c, right[k+1], c, optimize='optimal')
        second = 2*(jnp.sum((ddb@h)*b, axis=1)+jnp.sum((db@h)*db, axis=1))/norm
        result += jnp.sum(w*second**2)
    return result
