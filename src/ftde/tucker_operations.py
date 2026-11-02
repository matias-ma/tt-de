"""Core-sized Tucker contractions and lossless interchange for diagnostics."""
import os
import tempfile
from pathlib import Path

import jax.numpy as jnp
import numpy as np
from chebttge.chebtt import ChebTT
from chebttge.cores import ChebyshevCoeffCore
from chebtuckerge.chebtucker import ChebTucker
from chebtuckerge.factors import ChebyshevCoeffFactor

from .cheb_gram import chebyshev_gram_matrix
from .tucker_density import _mode_product, squared_integral_tucker


def init_tucker_constant(domain, *, rank=3, modes=25, seed=0, perturbation=.01,
                         max_core_entries=2**20):
    """Near-uniform, physical-L2-normalized start using only a box and seed.

    Factors are orthonormal in physical L2; a small dense random core makes
    every direction trainable. No data, pilot or true density is consulted.
    """
    domain = np.asarray(domain, dtype=float)
    if domain.ndim != 2 or domain.shape[1] != 2 or not len(domain) or not np.isfinite(domain).all() or np.any(np.diff(domain) <= 0):
        raise ValueError('domain must contain increasing finite intervals')
    ranks = np.broadcast_to(rank, (len(domain),))
    if np.any(ranks < 1) or np.any(ranks != np.floor(ranks)) or np.any(ranks > modes) or modes < 2:
        raise ValueError('ranks must be positive integers no larger than modes')
    ranks = tuple(map(int, ranks))
    if np.prod(ranks, dtype=object) > max_core_entries:
        raise ValueError('dense Tucker core exceeds max_core_entries; reduce mode ranks')
    if not np.isfinite(perturbation) or perturbation < 0:
        raise ValueError('perturbation must be finite and nonnegative')
    rng = np.random.default_rng(seed)
    factors = []
    for box, r in zip(domain, ranks, strict=True):
        c = rng.normal(size=(r, modes))/(1+np.arange(modes)**2)
        c[0] *= perturbation
        c[0, 0] += 1
        chol = np.linalg.cholesky(np.asarray(chebyshev_gram_matrix(modes, box)))
        q, _ = np.linalg.qr((c@chol).T)
        factors.append(ChebyshevCoeffFactor(jnp.asarray(q.T@np.linalg.inv(chol)), tuple(box)))
    core = perturbation*rng.normal(size=ranks)/np.sqrt(np.prod(ranks))
    core[(0,)*len(ranks)] += 1
    core /= np.linalg.norm(core)
    return ChebTucker(core, factors, jnp.asarray(domain))


def integrated_energies_tucker(params, grams, first, second):
    """Exact norm, H1, axis-H2 and full Hessian without a squared dense core."""
    core, *factors = params
    norm = core
    slope = axis = hessian = jnp.zeros_like(core)
    for k, (c, g, d1, d2) in enumerate(zip(factors, grams, first, second, strict=True)):
        m, m1, m2 = (c@a@c.T for a in (g, d1, d2))
        apply = lambda state, matrix, k=k: _mode_product(state, matrix, k)
        hessian = apply(hessian, m)+2*apply(slope, m1)+apply(norm, m2)
        axis = apply(axis, m)+apply(norm, m2)
        slope = apply(slope, m)+apply(norm, m1)
        norm = apply(norm, m)
    return tuple(jnp.sum(core*a) for a in (norm, slope, axis, hessian))


def expand_tucker_rank(model, rank, *, max_core_entries=2**20):
    """Add physical-L2-orthogonal factor directions without changing the function.

    The old core is embedded in a zero-padded core. Its new entries have
    trainable gradients immediately; no fitted density or noise is added.
    """
    old = model.core.shape
    ranks = np.broadcast_to(rank, (len(old),))
    sizes = [f.core.coeffs.shape[-1] for f in model.factors]
    if np.any(ranks != np.floor(ranks)) or np.any(ranks < old) or np.any(ranks > sizes):
        raise ValueError('Expanded ranks must be integers between existing ranks and polynomial sizes')
    ranks = tuple(map(int, ranks))
    if np.prod(ranks, dtype=object) > max_core_entries:
        raise ValueError('dense Tucker core exceeds max_core_entries')
    factors = []
    for factor, target in zip(model.factors, ranks, strict=True):
        c = np.asarray(factor.core.coeffs[0])
        chol = np.linalg.cholesky(np.asarray(chebyshev_gram_matrix(c.shape[-1], factor.domain)))
        q, _ = np.linalg.qr((c@chol).T, mode='complete')
        extra = np.linalg.solve(chol.T, q[:, c.shape[0]:target]).T
        factors.append(ChebyshevCoeffFactor(jnp.asarray(np.concatenate([c, extra])), factor.domain))
    core = np.zeros(ranks, dtype=np.asarray(model.core).dtype)
    core[tuple(slice(r) for r in old)] = np.asarray(model.core)
    return ChebTucker(jnp.asarray(core), factors, model.domain)


def tucker_to_tt(model):
    """Convert the small dense core by full SVD, without rank truncation.

    Diagnostic/kernel interchange only; training differentiates native
    Tucker contractions and never differentiates an SVD.
    """
    state = np.asarray(model.core)
    cores, left = [], 1
    for k, factor in enumerate(model.factors):
        r = factor.rank
        if k == len(model.factors)-1:
            local = state.reshape(left, r, 1)
        else:
            u, s, vh = np.linalg.svd(state.reshape(left*r, -1), full_matrices=False)
            local = u.reshape(left, r, len(s))
            state = s[:, None]*vh
        coeff = np.einsum('arb,rn->abn', local, np.asarray(factor.core.coeffs[0]))
        cores.append(ChebyshevCoeffCore(jnp.asarray(coeff), factor.domain))
        left = local.shape[-1]
    return ChebTT(cores=tuple(cores), domain=model.domain)


def save_tucker(path, model):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, suffix='.npz', delete=False) as stream:
        temporary = Path(stream.name)
    try:
        np.savez_compressed(temporary, format='tucker', domain=np.asarray(model.domain), core=np.asarray(model.core),
                            **{f'factor_{k}': np.asarray(f.core.coeffs[0]) for k, f in enumerate(model.factors)})
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def load_tucker(path):
    with np.load(path) as data:
        domain = data['domain']
        factors = tuple(ChebyshevCoeffFactor(jnp.asarray(data[f'factor_{k}']), tuple(box)) for k, box in enumerate(domain))
        return ChebTucker(jnp.asarray(data['core']), factors, jnp.asarray(domain))


def squared_marginal_tucker(g, points, keep_modes):
    """Exact normalized squared-density marginal, avoiding square_tucker."""
    keep = tuple(keep_modes)
    if len(set(keep)) != len(keep) or any(k < 0 or k >= len(g.factors) for k in keep):
        raise ValueError('keep_modes must contain distinct valid modes')
    points = jnp.asarray(points)
    if points.ndim != 2 or points.shape[1] != len(keep):
        raise ValueError('points must have shape (batch, len(keep_modes))')
    # Contract retained coordinates first, leaving only integrated ranks.
    state = jnp.broadcast_to(g.core, (len(points),)+g.core.shape)
    remaining = list(range(len(g.factors)))
    for k in sorted(keep, reverse=True):
        pos = remaining.index(k)
        state = jnp.moveaxis(state, pos+1, -1)
        values = g.factors[k](points[:, keep.index(k)])
        state = jnp.einsum('b...r,br->b...', state, values)
        remaining.remove(k)
    transformed = state
    for pos, k in enumerate(remaining):
        f = g.factors[k]
        c = f.core.coeffs[0]
        m = c@chebyshev_gram_matrix(c.shape[-1], f.domain)@c.T
        transformed = _mode_product(transformed, m, pos+1)
    values = jnp.sum((state*transformed).reshape(len(points), -1), axis=1)/squared_integral_tucker(g)
    boxes = g.domain[jnp.asarray(keep, dtype=int)]
    inside = jnp.all((points >= boxes[:, 0]) & (points <= boxes[:, 1]), axis=1)
    return jnp.where(inside, values, 0.)
