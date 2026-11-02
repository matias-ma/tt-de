"""Elementary TT starts: a constant with a small, smooth random perturbation.

No observations, density pilot, KDE, cross approximation, or optimization
enters this initializer. Its only geometric input is the supplied box.
"""

import jax.numpy as jnp
import numpy as np
from chebttge.chebtt import ChebTT
from chebttge.cores import ChebyshevCoeffCore

from .cheb_gram import chebyshev_gram_matrix


def init_tt_constant(domain, *, rank=16, modes=25, seed=0, perturbation=.01,
                     spectral_decay=1., spectral_scale=None):
    """Near-constant amplitude with active TT bonds, normalized in physical L2.

    The constant path is core entry (0,0,0). Random entries elsewhere avoid
    the identically zero gradients of exactly zero-padded higher ranks.
    Random coefficient amplitudes decay as (1 + degree**2)**(-spectral_decay).
    An optional spectral_scale additionally damps high degrees by a Gaussian
    low-pass envelope, floored at 1e-6 to retain nonzero random bond paths.
    Neither option restricts the coefficient basis available during fitting.
    QR only changes the TT gauge; it does not fit any function or data.
    """
    domain = np.asarray(domain, dtype=float)
    if domain.ndim != 2 or domain.shape[1] != 2 or not np.isfinite(domain).all() or np.any(domain[:, 1] <= domain[:, 0]):
        raise ValueError('domain must contain increasing intervals')
    if rank < 1 or modes < 2 or perturbation < 0:
        raise ValueError('invalid rank, modes, or perturbation')
    if not np.isfinite(spectral_decay) or spectral_decay < 0:
        raise ValueError('spectral_decay must be finite and nonnegative')
    if spectral_scale is not None and (not np.isfinite(spectral_scale) or spectral_scale <= 0):
        raise ValueError('spectral_scale must be finite and positive')
    envelope = (1+np.arange(modes, dtype=float)**2)**(-spectral_decay)
    if spectral_scale is not None:
        envelope *= np.maximum(np.exp(-.5*(np.arange(modes)/spectral_scale)**2), 1e-6)
    rng = np.random.default_rng(seed)
    arrays, inverses = [], []
    for k, box in enumerate(domain):
        shape = (1 if k == 0 else rank, 1 if k == len(domain)-1 else rank, modes)
        rng_values = rng.normal(size=shape)
        c = perturbation*rng_values/np.sqrt(rank)/(1+np.arange(modes)**2)
        if spectral_decay != 1. or spectral_scale is not None:
            c = perturbation*rng_values/np.sqrt(rank)*envelope
        c[0, 0, 0] += 1.
        chol = np.linalg.cholesky(np.asarray(chebyshev_gram_matrix(modes, box)))
        arrays.append(c@chol)
        inverses.append(np.linalg.inv(chol))
    for k in range(len(arrays)-1):
        a, b, n = arrays[k].shape
        q, r = np.linalg.qr(arrays[k].transpose(0, 2, 1).reshape(a*n, b))
        arrays[k] = q.reshape(a, n, -1).transpose(0, 2, 1)
        arrays[k+1] = np.einsum('ab,bcn->acn', r, arrays[k+1])
    arrays[-1] /= np.linalg.norm(arrays[-1])
    cores = tuple(ChebyshevCoeffCore(jnp.asarray(c@inv), tuple(box))
                  for c, inv, box in zip(arrays, inverses, domain, strict=True))
    return ChebTT(cores=cores, domain=jnp.asarray(domain))
