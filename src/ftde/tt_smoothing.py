"""Mass-preserving spatial diffusion of a Chebyshev functional TT.

This is a linear Galerkin heat filter, with natural no-flux box boundaries.
It changes only the existing TT coefficients, preserving modes and ranks.
It is not a positivity certificate for a signed polynomial.
"""

import jax.numpy as jnp
import numpy as np
from scipy.linalg import eigh

from .cheb_gram import chebyshev_gram_matrix
from .sobolev import derivative_grams
from .tt_density import rebuild_tt


def heat_smooth_tt(model, standard_deviations):
    """Apply exp(sum_k sigma_k²/2 * Laplacian_k) in the same Chebyshev space.

    Natural Neumann boundaries retain mass inside the existing box. This
    differs from free-space Gaussian convolution near its boundaries.
    Negative input values can survive diffusion; assess them separately.
    The operator acts on the represented function itself. Smoothing an
    amplitude and then squaring it is different from smoothing its density.
    """
    sigma = np.asarray(standard_deviations, dtype=float)
    if sigma.shape != (len(model.cores),) or np.any(~np.isfinite(sigma)) or np.any(sigma < 0):
        raise ValueError('One finite nonnegative smoothing scale per coordinate is required')
    stiffness = derivative_grams(model, order=1)
    output = []
    for core, derivative, scale in zip(model.cores, stiffness, sigma, strict=True):
        n = core.coeffs.shape[-1]
        if scale == 0:
            output.append(core.coeffs)
            continue
        gram = np.asarray(chebyshev_gram_matrix(n, core.domain))
        eigenvalues, vectors = eigh(np.asarray(derivative), gram)
        # Constants span the known nullspace; roundoff must not damp them.
        eigenvalues[0] = 0.
        if np.any(eigenvalues[1:] <= 0):
            raise FloatingPointError('Invalid diffusion eigensystem')
        factors = np.exp(-.5*scale**2*eigenvalues)
        transform = (gram@vectors*factors)@vectors.T
        moments = gram[:, 0]
        # Remove eigensolver roundoff in the exact conservation identities.
        transform += np.outer(moments-transform@moments, moments)/(moments@moments)
        transform[0] = np.eye(n)[0]
        output.append(core.coeffs@jnp.asarray(transform))
    return rebuild_tt(model, tuple(output))
