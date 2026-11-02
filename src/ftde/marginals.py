"""Squared-TT pair marginals without materializing rank-four core products."""

import numpy as np
from scipy.fft import dctn

from .cheb_gram import chebyshev_gram_matrix


def squared_pair_grid(model, i, j, xs, ys):
    """Exact integrated marginal at a grid; rows are y and columns are x.

    Only two TT bonds are retained. This avoids building the rank-squared
    density TT, whose individual cores cost O(r**4 * degree) storage.
    """
    if hasattr(model, 'factors'):
        from .tucker_diagnostics import squared_pair_grid_tucker
        return squared_pair_grid_tucker(model, i, j, xs, ys)
    if not 0 <= i < j < len(model.cores):
        raise ValueError('require 0 <= i < j < dimension')
    cores = [np.asarray(c.coeffs)/np.max(np.abs(c.coeffs)) for c in model.cores]
    grams = [np.asarray(chebyshev_gram_matrix(c.shape[-1], model.cores[k].domain))
             for k, c in enumerate(cores)]

    def advance(state, k):
        c = cores[k]
        a, b, n = c.shape
        if state.ndim > 2 and a*b <= 4096:
            # Reuse one transfer matrix for all x-grid points. Explicit GEMMs
            # avoid a slow batched einsum on nonadjacent high-rank marginals.
            # The cap keeps each transfer array at most 128 MiB (float64).
            transfer = ((c@grams[k]).reshape(a*b, n)@c.reshape(a*b, n).T)
            transfer = transfer.reshape(a, b, a, b).transpose(0, 2, 1, 3).reshape(a*a, b*b)
            return (state.reshape(-1, a*a)@transfer).reshape(*state.shape[:-2], b, b)
        return np.einsum('...ab,ain,bjn->...ij', state, c@grams[k], c, optimize=True)

    def values(k, nodes):
        a, b = model.cores[k].domain
        t = 2*(np.asarray(nodes)-a)/(b-a)-1
        basis = np.polynomial.chebyshev.chebvander(np.clip(t, -1, 1), cores[k].shape[-1]-1)
        result = np.einsum('xn,abn->xab', basis, cores[k])
        result[np.abs(t) > 1] = 0
        return result

    right = np.ones((1, 1))
    for k in range(len(cores)-1, j, -1):
        right = np.einsum('ij,ain,bjn->ab', right, cores[k]@grams[k], cores[k], optimize=True)
        right /= np.max(np.abs(right))
    left = np.ones((1, 1))
    for k in range(i):
        left = advance(left, k)
        left /= np.max(np.abs(left))
    norm = advance(left, i)
    v = values(i, xs)
    state = np.einsum('ab,xai,xbj->xij', left, v, v, optimize=True)
    for k in range(i+1, j):
        state, norm = advance(state, k), advance(norm, k)
        scale = np.max(np.abs(norm))
        state, norm = state/scale, norm/scale
    norm = np.einsum('ij,ij', advance(norm, j), right)
    v = values(j, ys)
    local = np.einsum('yai,ij,ybj->yab', v, right, v, optimize=True)
    grid = np.einsum('xab,yab->yx', state, local, optimize=True)/norm
    if not np.isfinite(grid).all() or norm <= 0 or grid.min() < -1e-10*max(1., grid.max()):
        raise FloatingPointError('Invalid squared-TT marginal')
    return np.maximum(grid, 0.)


def squared_pair_coefficients(model, i, j):
    """Exact pair-density Chebyshev coefficients, indexed (y mode, x mode)."""
    cores = [f.core for f in model.factors] if hasattr(model, 'factors') else model.cores
    nx, ny = (2*cores[k].coeffs.shape[-1]-1 for k in (i, j))
    box = np.asarray(model.domain)[[i, j]]
    xs, ys = [a+(np.cos(np.pi*(np.arange(n)+.5)/n)+1)*(b-a)/2
              for (a, b), n in zip(box, (nx, ny), strict=True)]
    coefficients = dctn(squared_pair_grid(model, i, j, xs, ys), type=2)/(nx*ny)
    coefficients[0, :] *= .5
    coefficients[:, 0] *= .5
    return coefficients, box


def relative_pair_hessian_energy(coefficients, box, length_scales):
    """Exact integral of weighted squared density Hessian / integral density².

    Includes the mixed derivative twice. This is a marginal-density diagnostic,
    not the full-dimensional amplitude penalty used in training.
    """
    c = np.asarray(coefficients)
    gx = np.asarray(chebyshev_gram_matrix(c.shape[1], tuple(box[0])))
    gy = np.asarray(chebyshev_gram_matrix(c.shape[0], tuple(box[1])))

    def norm(a):
        return float(np.sum(a*(gy[:a.shape[0], :a.shape[0]]@a@gx[:a.shape[1], :a.shape[1]])))

    dx, dy = 2/np.ptp(box, axis=1)
    xx = np.polynomial.chebyshev.chebder(c, m=2, axis=1)*dx**2
    yy = np.polynomial.chebyshev.chebder(c, m=2, axis=0)*dy**2
    xy = np.polynomial.chebyshev.chebder(np.polynomial.chebyshev.chebder(c, axis=0), axis=1)*dx*dy
    sx, sy = length_scales
    return (sx**4*norm(xx)+2*sx**2*sy**2*norm(xy)+sy**4*norm(yy))/norm(c)


def squared_univariate_grid(model, axis, xs):
    """Exact normalized 1D density marginal, zero outside its coordinate box."""
    if not 0 <= axis < len(model.cores):
        raise ValueError('axis outside model dimensions')
    cores = [np.asarray(c.coeffs)/np.max(np.abs(c.coeffs)) for c in model.cores]
    grams = [np.asarray(chebyshev_gram_matrix(c.shape[-1], model.cores[k].domain))
             for k, c in enumerate(cores)]
    left = np.ones((1, 1))
    for k in range(axis):
        c = cores[k]
        left = np.einsum('ab,ain,bjn->ij', left, c@grams[k], c, optimize=True)
        left /= np.max(np.abs(left))
    right = np.ones((1, 1))
    for k in range(len(cores)-1, axis, -1):
        c = cores[k]
        right = np.einsum('ij,ain,bjn->ab', right, c@grams[k], c, optimize=True)
        right /= np.max(np.abs(right))
    c = cores[axis]
    h = np.einsum('ab,ain,ij,bjm->nm', left, c, right, c, optimize=True)
    norm = np.sum(h*grams[axis])
    a, b = model.cores[axis].domain
    t = (2*np.asarray(xs)-a-b)/(b-a)
    basis = np.polynomial.chebyshev.chebvander(np.clip(t, -1, 1), c.shape[-1]-1)
    values = np.sum((basis@h)*basis, axis=-1)/norm
    values = np.where(np.abs(t) <= 1, values, 0.)
    if not np.isfinite(values).all() or norm <= 0 or values.min() < -1e-10*max(1., values.max()):
        raise FloatingPointError('Invalid squared-TT marginal')
    return np.maximum(values, 0.)
