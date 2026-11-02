"""Multiscale Fourier kernel scores for a squared Chebyshev functional TT.

Only the TT coefficients are fitted. Empirical Fourier means are observation
statistics, not a density pilot. Known independent additive noise multiplies
the model characteristic function; it is never divided out of the data.
Finite frequency sets approximate a characteristic Gaussian-mixture kernel.
"""

import jax
import jax.numpy as jnp
import numpy as np
from scipy.special import roots_legendre

from .gaussian_convolution import _transfer


def gaussian_mixture_characteristic(frequencies, weights, covariances):
    frequencies, weights, covariances = map(np.asarray, (frequencies, weights, covariances))
    if covariances.shape != (len(weights), frequencies.shape[1], frequencies.shape[1]):
        raise ValueError('Noise covariance dimensions do not match frequencies')
    if np.any(weights < 0) or not np.isclose(weights.sum(), 1):
        raise ValueError('Noise weights must be probabilities')
    if not np.allclose(covariances, covariances.swapaxes(-1, -2)) or np.min(np.linalg.eigvalsh(covariances)) < -1e-12:
        raise ValueError('Noise covariances must be symmetric positive semidefinite')
    return np.exp(-.5*np.einsum('fi,cij,fj->fc', frequencies, covariances, frequencies))@weights


def draw_frequencies(observations, count, seed, bandwidths=(.2, .5, 1.)):
    """Fixed multiscale Gaussian kernel in training-observation standard units."""
    observations = np.asarray(observations)
    scales = np.maximum(observations.std(axis=0), 1e-6)
    rng = np.random.default_rng(seed)
    widths = np.resize(np.asarray(bandwidths), count)
    if np.any(widths <= 0):
        raise ValueError('Kernel bandwidths must be positive')
    frequencies = rng.normal(size=(count, observations.shape[1]))
    return frequencies/(widths[:, None]*scales*np.sqrt(observations.shape[1]))


def empirical_characteristic(observations, frequencies, batch_size=2048):
    total = np.zeros(len(frequencies), dtype=complex)
    for start in range(0, len(observations), batch_size):
        total += np.exp(1j*np.asarray(observations[start:start+batch_size])@frequencies.T).sum(axis=0)
    return total/len(observations)


class FourierKernelScore:
    """Normalized TT Fourier integrals via convergent physical-box quadrature.

    ``problem`` supplies its coefficient coordinates and exact norm. The
    weighted Gram integrals resolve the polynomial and oscillatory factors;
    quadrature convergence is checked separately, not assumed exact.
    """

    def __init__(self, problem, frequencies, *, noise_characteristic=None, order_multiplier=1):
        self.problem = problem
        self.frequencies = np.asarray(frequencies, dtype=float)
        if self.frequencies.ndim != 2 or self.frequencies.shape[1] != len(problem.domain) or not np.isfinite(self.frequencies).all():
            raise ValueError('Invalid Fourier frequencies')
        self.noise = jnp.asarray(np.ones(len(frequencies)) if noise_characteristic is None else noise_characteristic)
        if self.noise.shape != (len(frequencies),):
            raise ValueError('Invalid noise characteristic function')
        grams = []
        self.orders = []
        for k, (box, n, inv) in enumerate(zip(problem.domain, problem.sizes, problem.inverse, strict=True)):
            order = int(order_multiplier*max(2*n, n+32+int(np.ceil(np.max(np.abs(self.frequencies[:, k]))*np.ptp(box)/2))))
            t, w = roots_legendre(order)
            x = box.mean()+np.ptp(box)*t/2
            basis = np.polynomial.chebyshev.chebvander(t, n-1)@inv.T
            phase = np.exp(1j*self.frequencies[:, k, None]*x[None])*w[None]*np.ptp(box)/2
            gram = np.einsum('fq,qi,qj->fij', phase, basis, basis, optimize=True)
            grams.append(jnp.asarray(gram))
            self.orders.append(order)
        self.grams = tuple(grams)
        self.characteristic = jax.jit(self._characteristic)

    def _characteristic(self, params, indices):
        state = jnp.ones((len(indices), 1, 1), dtype=jnp.complex128)
        for core, gram in zip(params, self.grams, strict=True):
            state = _transfer(state, core, gram[indices])
        return state[:, 0, 0]*self.noise[indices]/self.problem.energies(params)[0]

    def discrepancy(self, params, target, indices):
        difference = self._characteristic(params, indices)-target[indices]
        return jnp.mean(jnp.real(difference*jnp.conj(difference)))

    def predict(self, params, batch_size=64):
        return np.concatenate([np.asarray(self.characteristic(params, jnp.arange(i, min(i+batch_size, len(self.frequencies)))))
                               for i in range(0, len(self.frequencies), batch_size)])

    def observation_scores(self, prediction, observations, batch_size=1024):
        """Per-observation proper kernel scores, omitting the data-only term."""
        constant = np.mean(np.abs(prediction)**2)
        result = []
        for start in range(0, len(observations), batch_size):
            phase = np.exp(1j*np.asarray(observations[start:start+batch_size])@self.frequencies.T)
            result.append(constant-2*np.real(phase@np.conj(prediction))/len(prediction))
        return np.concatenate(result)
