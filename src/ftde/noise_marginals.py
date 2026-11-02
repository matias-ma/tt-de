"""Normalized univariate noise marginals extracted from a prescribed ChebTT.

No Gaussian parameters or covariance matrices are needed. Construction and
quadrature preparation happen outside JIT; the resulting fixed arrays can
be used by differentiable fitting objectives.
"""

from dataclasses import dataclass

import numpy as np
from scipy.special import roots_legendre

from .cheb_gram import chebyshev_moment_vector


@dataclass(frozen=True, eq=False)
class PolynomialNoiseMarginal:
    coefficients: np.ndarray
    domain: tuple

    def __post_init__(self):
        c = np.asarray(self.coefficients, dtype=float)
        a, b = self.domain
        if c.ndim != 1 or not len(c) or not np.isfinite(c).all():
            raise ValueError('noise coefficients must be a finite nonempty vector')
        if not np.isfinite([a, b]).all() or a >= b:
            raise ValueError('noise domain must be finite and nonempty')
        norm = c@np.asarray(chebyshev_moment_vector(len(c), (a, b)))
        if not np.isfinite(norm) or norm <= 0:
            raise ValueError('noise marginal must have positive finite mass')
        c = c/norm
        object.__setattr__(self, 'coefficients', c)
        object.__setattr__(self, 'domain', (float(a), float(b)))
        # Reject a signed approximation, allowing tiny spectral roundoff.
        t = np.cos(np.linspace(0., np.pi, max(65, 4*len(c))))
        values = np.polynomial.chebyshev.chebval(t, c)
        tolerance = 1e-10*max(float(values.max()), 1/(b-a))
        if values.min() < -tolerance:
            raise ValueError('noise marginal is materially negative; improve its TT approximation')
        object.__setattr__(self, '_negative_tolerance', tolerance)

    def pdf(self, x):
        a, b = self.domain
        x = np.asarray(x)
        t = np.clip(2*(x-a)/(b-a)-1, -1., 1.)
        values = np.polynomial.chebyshev.chebval(t, self.coefficients)
        if np.any(values < -self._negative_tolerance):
            raise ValueError('noise marginal is materially negative at quadrature nodes')
        return np.where((x >= a) & (x <= b), np.maximum(values, 0.), 0.)

    def cdf(self, x):
        a, b = self.domain
        t = np.clip(2*(np.asarray(x)-a)/(b-a)-1, -1., 1.)
        primitive = np.polynomial.chebyshev.chebint(self.coefficients)*(b-a)/2
        values = np.polynomial.chebyshev.chebval(t, primitive)-np.polynomial.chebyshev.chebval(-1., primitive)
        return np.clip(values, 0., 1.)

    def moments(self):
        t, w = roots_legendre(max(32, (len(self.coefficients)+3)//2))
        a, b = self.domain
        x = a+(t+1)*(b-a)/2
        mass = w*(b-a)/2*self.pdf(x)
        mean = mass@x
        variance = mass@((x-mean)**2)
        return float(mean), float(np.sqrt(max(0., variance)))

    def signal_quadrature(self, observations, signal_domain, order):
        """Per-observation signal nodes, respecting both finite supports."""
        a, b = signal_domain
        u, v = self.domain
        y = np.asarray(observations)
        left, right = np.maximum(a, y-v), np.minimum(b, y-u)
        if np.any(right <= left):
            raise ValueError('observations lie outside signal support plus noise support')
        t, w = roots_legendre(order)
        x = left[:, None]+(right-left)[:, None]*(t+1)/2
        return x, (right-left)[:, None]*w/2

    def bin_quadrature(self, lower, upper, signal_domain, order):
        """Split at every bin/noise-support kink before signal integration."""
        a, b = signal_domain
        u, v = self.domain
        lo, hi = np.asarray(lower), np.asarray(upper)
        breaks = np.sort(np.clip(np.stack([np.full_like(lo, a), np.full_like(lo, b),
                                          lo-u, lo-v, hi-u, hi-v], axis=1), a, b), axis=1)
        t, w = roots_legendre(order)
        widths = np.diff(breaks, axis=1)
        # Avoid allocating four redundant zero-width panels when the kernel
        # box already covers every signal/observation difference.
        indices = np.argsort(widths <= 0, axis=1, kind='stable')[:, :int(np.max(np.sum(widths > 0, axis=1)))]
        starts = np.take_along_axis(breaks[:, :-1], indices, axis=1)
        widths = np.take_along_axis(widths, indices, axis=1)
        x = starts[:, :, None]+widths[:, :, None]*(t+1)/2
        weights = widths[:, :, None]*w/2
        return x.reshape(len(lo), -1), weights.reshape(len(lo), -1)


def noise_marginals_from_tt(noise_tt):
    """Integrate all other modes of the known noise TT by coefficient contraction.

    Correlations are allowed and are retained in the caller's joint
    convolution likelihood. Only this prior-building step uses marginals.
    """
    cores, matrices = [], []
    for core in noise_tt.cores:
        c = np.asarray(core.coeffs, dtype=float)
        scale = np.max(np.abs(c))
        if not np.isfinite(scale) or scale == 0:
            raise ValueError('noise TT must have finite nonzero cores')
        c = c/scale
        cores.append(c)
        moments = np.asarray(chebyshev_moment_vector(c.shape[-1], core.domain))
        matrices.append(np.einsum('abn,n->ab', c, moments))

    def normalize(state):
        scale = np.max(np.abs(state))
        if not np.isfinite(scale) or scale == 0:
            raise ValueError('noise TT has a zero or nonfinite marginal environment')
        return state/scale

    left, right = [np.ones(1)], [None]*(len(cores)+1)
    right[-1] = np.ones(1)
    for matrix in matrices:
        left.append(normalize(left[-1]@matrix))
    for k in range(len(cores)-1, -1, -1):
        right[k] = normalize(matrices[k]@right[k+1])
    return tuple(PolynomialNoiseMarginal(np.einsum('a,abn,b->n', left[k], c, right[k+1]), core.domain)
                 for k, (c, core) in enumerate(zip(cores, noise_tt.cores, strict=True)))
