"""Exact squared-Tucker convolution without materializing the squared core.

The fixed kernel may use TT or Tucker format. Local polynomial moments act
on one copy of the dense signal core, followed by an inner product with the
other copy. Memory grows with the core size, not its square. This is the
Tucker analogue of the TT double-layer contraction, with no rank truncation.
"""
from dataclasses import dataclass, replace

import jax
import jax.numpy as jnp
import numpy as np
from chebttge.chebtt import ChebTT
from chebttge.cores import ChebyshevCoeffCore
from chebtuckerge.chebtucker import ChebTucker

from .tt_convolution import TTConvolution
from .tucker_density import squared_integral_tucker
from .tucker_operations import tucker_to_tt


def _chebyshev_basis(t, modes):
    """All Chebyshev modes in logarithmic depth, with finite endpoint JVPs.

    T_(2k)=2*T_k²-1 and T_(2k+1)=2*T_k*T_(k+1)-t avoid a long
    accelerator scan over the (often 513) noise-kernel coefficients.
    """
    if modes == 1:
        return jnp.ones((len(t), 1), dtype=t.dtype)
    values = jnp.stack([jnp.ones_like(t), t], axis=-1)
    while values.shape[-1] < modes:
        even = 2*values**2-1
        odd = 2*values[:, :-1]*values[:, 1:]-t[:, None]
        interleaved = jnp.stack([even[:, :-1], odd], axis=-1).reshape(len(t), -1)
        values = jnp.concatenate([interleaved, even[:, -1:]], axis=-1)
    return values[:, :modes]


@dataclass(frozen=True)
class TuckerConvolution:
    moments: TTConvolution
    pair_axes: bool = True
    pair_groups: tuple | None = None
    separable_pairs: bool = True

    @classmethod
    def from_tucker(cls, template, kernel, *, quadrature_order=None):
        if isinstance(kernel, ChebTucker):
            kernel = tucker_to_tt(kernel)
        proxy = ChebTT(cores=tuple(ChebyshevCoeffCore(jnp.ones((1, 1, f.core.coeffs.shape[-1])), f.domain)
                                  for f in template.factors), domain=template.domain)
        return cls(TTConvolution.from_tt(proxy, kernel, quadrature_order=quadrature_order))

    @property
    def safe_domain(self):
        return self.moments.safe_domain

    def prepare(self, observation_domain=None, *, tolerance=0.):
        moments = self.moments.prepare(observation_domain, tolerance=tolerance)
        groups = []
        # Intersect exact structural blocks across each possible coordinate
        # pair. Off-diagonal mixture blocks are identically zero, so retaining
        # them would waste core-sized contractions and reverse-mode storage.
        for k in range(len(moments.sizes)-1):
            blocks = []
            for li, middle_a, ca in moments.moment_groups[k]:
                for middle_b, ri, cb in moments.moment_groups[k+1]:
                    common, ia, ib = np.intersect1d(np.asarray(middle_a), np.asarray(middle_b), return_indices=True)
                    if not len(common):
                        continue
                    na, nb = 2*moments.sizes[k]-1, 2*moments.sizes[k+1]-1
                    a = ca.coeffs.reshape(len(li), len(middle_a), na, -1)[:, ia]
                    b = cb.coeffs.reshape(len(middle_b), len(ri), nb, -1)[ib]
                    ac = ChebyshevCoeffCore(a.reshape(len(li), len(common)*na, -1), ca.domain)
                    bc = ChebyshevCoeffCore(b.reshape(len(common), len(ri)*nb, -1), cb.domain)
                    blocks.append((li, ri, len(common), ac, bc))
            groups.append(tuple(blocks))
        return replace(self, moments=moments, pair_groups=tuple(groups))

    def _factor_moments(self, k, c, y, moment_core=None, shape=None):
        op = self.moments
        if op.moment_cores is None:
            return jnp.einsum('ri,bacij,sj->bacrs', c, op._moments(k, y), c, optimize='optimal')
        n = c.shape[-1]
        moment_core = op.moment_cores[k] if moment_core is None else moment_core
        shape = op.kernel.cores[k].shape if shape is None else shape
        a, b = moment_core.domain
        basis = _chebyshev_basis((2*y-a-b)/(b-a), moment_core.coeffs.shape[-1])
        h = (basis@moment_core.coeffs.reshape(-1, basis.shape[-1]).T).reshape(len(y), *shape, 2*n-1)
        # T_i*T_j=(T_{i+j}+T_|i-j|)/2. Form small factor products,
        # never batch*kernel_bonds*n*n polynomial moment matrices.
        i, j = jnp.indices((n, n))
        outer = (c[:, None, :, None]*c[None, :, None, :]).reshape(c.shape[0], c.shape[0], -1)/2
        products = jnp.zeros((c.shape[0], c.shape[0], 2*n-1), dtype=c.dtype)
        products = products.at[:, :, (i+j).ravel()].add(outer)
        products = products.at[:, :, jnp.abs(i-j).ravel()].add(outer)
        return jnp.einsum('bach,rsh->bacrs', h, products)

    def log_density(self, g, y):
        op = self.moments
        if tuple(f.core.coeffs.shape[-1] for f in g.factors) != op.sizes:
            raise ValueError('operator and Tucker must have the same polynomial sizes')
        domains = tuple(f.domain for f in g.factors)
        concrete = not any(isinstance(v, jax.core.Tracer) for v in jax.tree.leaves(domains))
        if concrete and domains != op.domains:
            raise ValueError('operator and Tucker must have the same domains')
        y = jnp.asarray(y, dtype=jnp.float64)
        single = y.ndim == 1
        if single:
            y = y[None]
        if y.ndim != 2 or y.shape[1] != len(g.factors):
            raise ValueError('y must have shape (d,) or (batch, d)')
        state = jnp.broadcast_to(g.core, (len(y), 1)+g.core.shape)
        k = 0
        while k < len(g.factors):
            factor = g.factors[k]
            c = factor.core.coeffs[0]
            # Eliminate a large internal noise bond before it ever touches
            # the dense signal core. This is particularly effective for
            # prescribed mixtures of independent correlated coordinate pairs.
            # The resulting pair moment is exact; no rank truncation occurs.
            if (self.pair_axes and k+1 < len(g.factors)
                    and op.kernel.cores[k].shape[1] >= max(g.core.shape[k:k+2])
                    and op.kernel.cores[k].shape[0]*op.kernel.cores[k+1].shape[1] <= 16):
                state = jnp.moveaxis(state, (k+2, k+3), (-2, -1))
                shape = state.shape
                state = state.reshape(*shape[:-2], shape[-2]*shape[-1])
                def apply_pair(local_state, a, b, pair_rank=shape[-2]*shape[-1]):
                    if self.separable_pairs and a.shape[2] == 1:
                        # A separable noise block needs two rank-by-rank
                        # actions, not a dense rank²-by-rank² pair action.
                        # Keep both core axes explicit until they are applied.
                        local_state = local_state.reshape(*local_state.shape[:-1], a.shape[-1], b.shape[-1])
                        left = jnp.einsum('bl...su,blmrs->bm...ru', local_state, a)
                        paired = jnp.einsum('bm...ru,bmhtu->bh...rt', left, b)
                        return paired.reshape(*paired.shape[:-2], pair_rank)
                    pair = jnp.einsum('blmrs,bmhtu->blhrtsu', a, b)
                    pair = pair.reshape(*pair.shape[:3], pair_rank, -1)
                    return jnp.einsum('bl...s,blhrs->bh...r', local_state, pair)
                if self.pair_groups is not None and len(self.pair_groups[k]) > 1:
                    next_state = jnp.zeros((len(y), op.kernel.cores[k+1].shape[1])+state.shape[2:], dtype=state.dtype)
                    for li, ri, middle, ca, cb in self.pair_groups[k]:
                        a = self._factor_moments(k, c, y[:, k], ca, (len(li), middle))
                        b = self._factor_moments(k+1, g.factors[k+1].core.coeffs[0], y[:, k+1], cb, (middle, len(ri)))
                        block = apply_pair(jnp.take(state, li, axis=1), a, b)
                        # Several paths may share an output bond (the final
                        # mixture sum in particular), so accumulate, not set.
                        next_state = next_state.at[:, ri].add(block)
                    state = next_state
                else:
                    a = self._factor_moments(k, c, y[:, k])
                    b = self._factor_moments(k+1, g.factors[k+1].core.coeffs[0], y[:, k+1])
                    state = apply_pair(state, a, b)
                state = state.reshape(*state.shape[:-1], *g.core.shape[k:k+2])
                state = jnp.moveaxis(state, (-2, -1), (k+2, k+3))
                k += 2
                continue
            state = jnp.moveaxis(state, k+2, -1)
            if op.moment_groups is not None and len(op.moment_groups[k]) > 1:
                # Reuse TT's exact kernel block structure (e.g. mixtures).
                # Blocks have disjoint output bonds, so no summation or
                # approximation is hidden in the scatter.
                shape = (len(y), op.kernel.cores[k].shape[1])+state.shape[2:-1]+(c.shape[0],)
                next_state = jnp.zeros(shape, dtype=state.dtype)
                for li, ri, moment_core in op.moment_groups[k]:
                    local = self._factor_moments(k, c, y[:, k], moment_core, (len(li), len(ri)))
                    block = jnp.einsum('ba...s,bacrs->bc...r', jnp.take(state, li, axis=1), local)
                    next_state = next_state.at[:, ri].set(block)
                state = next_state
            else:
                local = self._factor_moments(k, c, y[:, k])
                state = jnp.einsum('ba...s,bacrs->bc...r', state, local)
            state = jnp.moveaxis(state, -1, k+2)
            k += 1
        value = jnp.sum((state[:, 0]*g.core).reshape(len(y), -1), axis=1)
        result = jnp.log(value)-jnp.log(squared_integral_tucker(g))
        valid = jnp.all((y >= op.safe_domain[:, 0]) & (y <= op.safe_domain[:, 1]), axis=1)
        valid &= jnp.all(jnp.asarray(domains) == jnp.asarray(op.domains))
        result = jnp.where(valid, result, jnp.nan)
        return result[0] if single else result
