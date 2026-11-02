"""Marginal Fisher-information filtering for a squared Chebyshev TT.

The forward model remains the full correlated convolution. This regularizer
uses marginal noise information geometry, centered at log-density references
fitted from noisy samples. Marginals can come directly from a prescribed TT
kernel; a Gaussian standard-deviation shortcut is also available.
"""

from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np
from scipy.linalg import null_space
from scipy.special import roots_legendre

from .cheb_gram import chebyshev_gram_matrix
from .noise_marginals import noise_marginals_from_tt
from .sobolev import amplitude_h1


def marginal_fisher_filter(n, domain, log_coeffs, noise_std=None, *, noise_marginal=None,
                          visibility_floor=1e-3, order=256):
    """Return a coefficient penalty and retained-information eigenvalues.

    For density rho and L2-unit amplitude perturbation h orthogonal to sqrt(rho),
    complete-data information / 4 is 1; observed information / 4 is
    integral [U*(sqrt(rho)*h)]² / (U*rho). Conditional expectation is a
    contraction, so its eigenvalues lie in [0,1]. Regularization weights
    alpha/(alpha+eigenvalue) target weakly observed directions and annihilate
    the projected reference amplitude. There is no exponentially increasing
    Chebyshev-degree weight or inverse singular value in the objective.
    """
    if n < 2 or order < max(2*n, 32):
        raise ValueError('at least two coefficients and adequate quadrature are required')
    if noise_marginal is None:
        if noise_std is None or not np.isfinite(noise_std) or noise_std <= 0:
            raise ValueError('noise_std must be finite and positive')
    elif noise_std is not None:
        raise ValueError('provide either a noise marginal or a Gaussian scale, not both')
    if not np.isfinite(visibility_floor) or visibility_floor <= 0:
        raise ValueError('visibility_floor must be finite and positive')
    a, b = domain
    if not np.isfinite([a, b]).all() or a >= b:
        raise ValueError('domain must be a finite nonempty interval')
    c = np.asarray(log_coeffs, dtype=float)
    if c.ndim != 1 or not len(c) or not np.isfinite(c).all():
        raise ValueError('log_coeffs must be a finite vector')
    t, w = roots_legendre(order)
    x, wx = a+(b-a)*(t+1)/2, w*(b-a)/2
    gram = np.asarray(chebyshev_gram_matrix(n, (a, b)))
    chol = np.linalg.cholesky(gram)
    basis = np.polynomial.chebyshev.chebvander(t, n-1)
    phi = np.linalg.solve(chol, basis.T).T
    log_rho = np.polynomial.chebyshev.chebval(t, c)
    log_shift = log_rho.max()
    rho = np.exp(log_rho-log_shift)
    normalization = wx@rho
    rho /= normalization
    amplitude = np.sqrt(rho)
    reference = phi.T@(wx*amplitude)
    reference_norm = np.linalg.norm(reference)
    if reference_norm < .99:
        raise ValueError('reference amplitude needs more Chebyshev coefficients')
    complement = null_space((reference/reference_norm)[None, :])
    if noise_marginal is None:
        # Include Gaussian tails of the entire bounded signal support.
        left, right = a-9*noise_std, b+9*noise_std
        y, wy = left+(right-left)*(t+1)/2, w*(right-left)/2
        kernel = np.exp(-.5*((y[:, None]-x)/noise_std)**2)/(noise_std*np.sqrt(2*np.pi))
        q = kernel@(wx*rho)
        response = kernel@((wx*amplitude)[:, None]*phi)
    else:
        # Compact noise may have jumps at its support boundary. Split the
        # observation integral where the overlap geometry changes, and clip
        # each inner signal integral to that overlap instead of masking a
        # fixed quadrature grid across a moving discontinuity.
        u, v = noise_marginal.domain
        breaks = np.unique([a+u, a+v, b+u, b+v])
        widths = np.diff(breaks)
        y = (breaks[:-1, None]+widths[:, None]*(t+1)/2).ravel()
        wy = (widths[:, None]*w/2).ravel()
        xx, ww = noise_marginal.signal_quadrature(y, domain, order)
        tt = 2*(xx-a)/(b-a)-1
        rr = np.exp(np.polynomial.chebyshev.chebval(tt, c)-log_shift)/normalization
        kernel_weights = noise_marginal.pdf(y[:, None]-xx)*ww
        q = np.sum(kernel_weights*rr, axis=1)
        local_basis = np.polynomial.chebyshev.chebvander(tt, n-1)
        # Contract in Chebyshev coordinates before Gram whitening.
        response = np.einsum('yx,yxi->yi', kernel_weights*np.sqrt(rr), local_basis)
        response = np.linalg.solve(chol, response.T).T
    if (not np.all(np.isfinite(q) & (q >= 0))
            or np.any((q == 0) & np.any(response != 0, axis=1))):
        raise FloatingPointError('observed reference underflow; adjust the domain or quadrature')
    observed_mass = float(wy@q)
    if abs(observed_mass-1.) > 1e-6:
        raise FloatingPointError('observed reference normalization failed; refine quadrature')
    response = (response-q[:, None]*reference)@complement
    # A compact kernel can have genuinely zero-density observation regions.
    # They contribute zero information when the response is also zero.
    information_weights = np.divide(wy, q, out=np.zeros_like(wy), where=q > 0)
    fisher = response.T@(information_weights[:, None]*response)
    fisher = .5*(fisher+fisher.T)
    eigenvalues, vectors = np.linalg.eigh(fisher)
    if eigenvalues.min() < -1e-9 or eigenvalues.max() > 1+1e-7:
        raise FloatingPointError('Fisher contraction failed; refine quadrature')
    eigenvalues = np.clip(eigenvalues, 0., 1.)
    weights = visibility_floor/(visibility_floor+eigenvalues)
    factor = chol@complement@vectors*np.sqrt(weights)[None, :]
    penalty = factor@factor.T
    return penalty, eigenvalues, {'reference_projection_squared_error': float(1-reference_norm**2),
                                 'visibility_floor': visibility_floor, 'quadrature_order': order,
                                 'observed_reference_mass': observed_mass,
                                 'zero_observation_nodes': int(np.sum(q == 0)),
                                 'noise_source': 'TT marginal' if noise_marginal is not None else 'Gaussian scale'}


@dataclass(frozen=True, eq=False)
class FisherVisibilityPenalty:
    """Gauge-invariant sum of marginal information filters on the TT amplitude.

    Supply ``noise_tt`` for arbitrary prescribed TT noise, or ``noise_std``
    for Gaussian marginal noise. No signal covariance is needed. Correlations
    still enter the full likelihood; this is not its full multidimensional
    Fisher Hessian. The reference and visibility floor are prior choices to
    select/check using noisy data.
    """

    grams: tuple
    derivatives: tuple
    weights: tuple
    domains: tuple
    sizes: tuple
    eigenvalues: tuple
    diagnostics: tuple

    @classmethod
    def from_tt(cls, template, log_coeffs, noise_std=None, *, noise_tt=None, visibility_floor=1e-3, order=256):
        domains = tuple(c.domain for c in template.cores)
        sizes = tuple(c.coeffs.shape[-1] for c in template.cores)
        if len(log_coeffs) != len(sizes):
            raise ValueError('one reference per axis is required')
        if noise_tt is None:
            if np.shape(noise_std) != (len(sizes),):
                raise ValueError('one Gaussian noise standard deviation per axis is required')
            marginals = [None]*len(sizes)
        else:
            if noise_std is not None or len(noise_tt.cores) != len(sizes):
                raise ValueError('provide a dimension-matched noise TT without Gaussian scales')
            marginals = noise_marginals_from_tt(noise_tt)
            noise_std = [None]*len(sizes)
        matrices, eigenvalues, diagnostics = [], [], []
        for n, box, c, std, noise in zip(sizes, domains, log_coeffs, noise_std, marginals, strict=True):
            matrix, spectrum, diagnostic = marginal_fisher_filter(n, box, c, std, noise_marginal=noise,
                visibility_floor=visibility_floor, order=order)
            matrices.append(jnp.asarray(matrix))
            eigenvalues.append(spectrum)
            diagnostics.append(diagnostic)
        return cls(tuple(chebyshev_gram_matrix(n, box) for n, box in zip(sizes, domains, strict=True)),
                   tuple(matrices), (1.,)*len(sizes), domains, sizes, tuple(eigenvalues), tuple(diagnostics))

    def __call__(self, g):
        if tuple(c.coeffs.shape[-1] for c in g.cores) != self.sizes:
            raise ValueError('Fisher penalty requires the same polynomial sizes')
        matches = jnp.all(jnp.asarray([c.domain for c in g.cores]) == jnp.asarray(self.domains))
        if not isinstance(matches, jax.core.Tracer) and not bool(matches):
            raise ValueError('Fisher penalty requires the same domains')
        return jnp.where(matches, amplitude_h1(g, self.grams, self.derivatives, self.weights), jnp.nan)
