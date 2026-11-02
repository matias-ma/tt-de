"""Finite exponential-family marginal pilots fitted through prescribed noise.

The degree and parameter sharing are supplied by the caller; select them
using held-out noisy likelihood. No signal density values are required.
"""

import jax
import jax.numpy as jnp
import numpy as np
from scipy.optimize import minimize
from scipy.special import logsumexp, ndtr, roots_legendre

from .noise_marginals import noise_marginals_from_tt


def fit_noisy_marginal_reference(train, valid, domain, std=None, degree=None, shared=False,
                               *, noise_tt=None, order=160, bins=1600):
    """Fit log-Chebyshev marginals by binned known-noise likelihood.

    Returns log coefficients (including a zero constant), mean marginal
    validation log density per observation, and optimizer diagnostics. All
    supplied observations must be noisy. ``shared`` ties coefficients in
    each axis's standardized coordinate, not in physical coordinates.
    Supply ``noise_tt`` for arbitrary known TT noise, or ``std`` for Gaussian
    marginal noise. Noise moments used for initialization are derived from
    the supplied kernel; no signal covariance or density is assumed known.
    """
    train, valid = np.asarray(train, dtype=float), np.asarray(valid, dtype=float)
    domain = np.asarray(domain, dtype=float)
    if train.ndim != 2 or valid.ndim != 2 or min(len(train), len(valid)) < 2 or train.shape[1] != valid.shape[1]:
        raise ValueError('training and validation require at least two observations and matching axes')
    d = train.shape[1]
    if noise_tt is not None:
        if std is not None or len(noise_tt.cores) != d:
            raise ValueError('provide a dimension-matched noise TT without Gaussian scales')
        marginals = noise_marginals_from_tt(noise_tt)
        noise_mean, std = np.asarray([m.moments() for m in marginals]).T
    else:
        marginals = None
        noise_mean = np.zeros(d)
    std = np.asarray(std, dtype=float)
    if domain.shape != (d, 2) or std.shape != (d,):
        raise ValueError('one domain and noise standard deviation per axis are required')
    if not all(np.isfinite(x).all() for x in (train, valid, domain, std)):
        raise ValueError('observations, domains, and noise scales must be finite')
    if np.any(domain[:, 0] >= domain[:, 1]) or np.any(std <= 0):
        raise ValueError('domains must be nonempty and noise scales positive')
    if degree is None or degree < 1 or bins < 2 or order < max(32, 2*degree):
        raise ValueError('positive degree, at least two bins, and adequate quadrature are required')
    t, w = roots_legendre(order)
    basis = np.polynomial.chebyshev.chebvander(t, degree)[:, 1:]
    nodes = domain[:, :1]+np.diff(domain, axis=1)*(t[None, :]+1)/2
    counts, matrices, bin_bases, bin_weights = [], [], [], []
    for k in range(d):
        edges = np.linspace(train[:, k].min()-.1, train[:, k].max()+.1, bins+1)
        count = np.histogram(train[:, k], edges)[0]
        keep = count > 0
        counts.append(jnp.asarray(count[keep]/len(train)))
        # The common wide-kernel case has no support-boundary crossings;
        # retain the inexpensive matrix-times-quadrature-mass likelihood.
        full_overlap = (marginals is not None and edges[0]-domain[k, 1] >= marginals[k].domain[0]
                        and edges[-1]-domain[k, 0] <= marginals[k].domain[1])
        if marginals is None or full_overlap:
            if marginals is None:
                matrix = ndtr((edges[1:, None]-nodes[k])/std[k])-ndtr((edges[:-1, None]-nodes[k])/std[k])
            else:
                matrix = marginals[k].cdf(edges[1:, None]-nodes[k])-marginals[k].cdf(edges[:-1, None]-nodes[k])
                matrix = np.maximum(matrix, 0.)
            matrices.append(jnp.asarray(matrix[keep]))
            bin_bases.append(None)
            bin_weights.append(None)
        else:
            noise = marginals[k]
            lower, upper = edges[:-1][keep], edges[1:][keep]
            bx, bw = noise.bin_quadrature(lower, upper, domain[k], order)
            probability = noise.cdf(upper[:, None]-bx)-noise.cdf(lower[:, None]-bx)
            weights = bw*np.maximum(probability, 0.)
            if np.any(weights.sum(axis=1) == 0):
                raise ValueError('observed bins lie outside signal support plus noise support')
            bt = 2*(bx-domain[k, 0])/(domain[k, 1]-domain[k, 0])-1
            bin_bases.append(jnp.asarray(np.polynomial.chebyshev.chebvander(bt, degree)[..., 1:]))
            bin_weights.append(jnp.asarray(np.log(weights, where=weights > 0,
                                                  out=np.full_like(weights, -np.inf))))
            matrices.append(None)
    jb, lw = jnp.asarray(basis), jnp.log(jnp.asarray(w))

    def loss(theta):
        coefficient = jnp.broadcast_to(theta, (d, degree)) if shared else theta.reshape(d, degree)
        mass = jax.nn.softmax(coefficient@jb.T+lw, axis=1)
        if marginals is not None:
            log_norm = jax.scipy.special.logsumexp(coefficient@jb.T+lw, axis=1)+jnp.log(jnp.asarray(np.diff(domain).ravel()/2))
            return -sum(jnp.sum(count*(jnp.log(matrix@prob+1e-300) if matrix is not None else
                jax.scipy.special.logsumexp(jnp.einsum('bnj,j->bn', bb, c)+bw, axis=1)-z))
                for count, matrix, prob, bb, bw, c, z in zip(counts, matrices, mass, bin_bases,
                                                           bin_weights, coefficient, log_norm, strict=True))/d
        return -sum(jnp.sum(count*jnp.log(matrix@prob+1e-300))
                    for count, matrix, prob in zip(counts, matrices, mass, strict=True))/d

    evaluate = jax.jit(jax.value_and_grad(loss))
    # Moment-corrected broad Gaussian initialization. No true signal function.
    variance = np.maximum(np.var(train, axis=0)-std**2, .1)
    center = train.mean(axis=0)-noise_mean
    log_values = -.5*(nodes-center[:, None])**2/variance[:, None]
    initial = np.stack([np.polynomial.chebyshev.chebfit(t, row, degree)[1:] for row in log_values])
    if shared:
        initial = initial.mean(axis=0)

    def objective(theta):
        value, gradient = evaluate(jnp.asarray(theta))
        return float(value), np.asarray(gradient)

    result = minimize(objective, initial.ravel(), jac=True, method='L-BFGS-B',
                      options={'maxiter': 1200, 'ftol': 1e-13, 'gtol': 1e-8, 'maxls': 40})
    coefficient = np.broadcast_to(result.x, (d, degree)).copy() if shared else result.x.reshape(d, degree)
    logs = []
    for k in range(d):
        lm = coefficient[k]@basis.T+np.log(w)
        if marginals is None:
            lm -= logsumexp(lm)
            kernel = -.5*((valid[:, k, None]-nodes[k])/std[k])**2-np.log(std[k]*np.sqrt(2*np.pi))
            logs.append(logsumexp(kernel+lm, axis=1))
        else:
            vx, vw = marginals[k].signal_quadrature(valid[:, k], domain[k], order)
            vt = 2*(vx-domain[k, 0])/(domain[k, 1]-domain[k, 0])-1
            log_signal = np.polynomial.chebyshev.chebval(vt, np.r_[0., coefficient[k]])
            weights = vw*marginals[k].pdf(valid[:, k, None]-vx)
            log_weights = np.log(weights, where=weights > 0, out=np.full_like(weights, -np.inf))
            log_norm = logsumexp(lm)+np.log((domain[k, 1]-domain[k, 0])/2)
            logs.append(logsumexp(log_signal+log_weights, axis=1)-log_norm)
    return np.pad(coefficient, ((0, 0), (1, 0))), np.mean(logs, axis=0), {
        'degree': degree, 'shared': shared, 'success': bool(result.success),
        'message': str(result.message), 'iterations': int(result.nit),
        'binned_train_nll': float(result.fun), 'validation_marginal_nll': float(-np.mean(logs)),
        'gradient_norm': float(np.linalg.norm(result.jac)), 'quadrature_order': order, 'bins': bins,
        'noise_source': 'TT marginals' if marginals is not None else 'Gaussian scales'}
