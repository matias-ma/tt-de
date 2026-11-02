"""Direct squared-TT contraction for an arbitrary prescribed Chebyshev TT kernel.

No Gaussian or independence assumption is made. Kernel bond indices travel
alongside the two signal bond indices; the rank-squared signal is never
materialized. Gauss--Legendre quadrature with the degree-derived node count
integrates the represented polynomial product exactly in exact arithmetic.
Approximation of the true noise by the supplied TT remains a separate error.
"""

from dataclasses import dataclass, replace
from typing import Protocol

import jax
import jax.numpy as jnp
import numpy as np
from chebttge.chebtt import ChebTT
from chebttge.cores import ChebyshevCoeffCore
from scipy.fft import dct
from scipy.special import roots_legendre

from .cheb_gram import chebyshev_gram_matrix
from .gaussian_convolution import _rescale, _transfer


class ConvolutionOperator(Protocol):
    def log_density(self, g: ChebTT, y: jnp.ndarray) -> jnp.ndarray: ...


@dataclass(frozen=True)
class TTConvolution:
    """Convolve g**2 with a fixed, not necessarily separable noise TT.

    ``kernel`` is assumed to represent a normalized, nonnegative noise
    density, just as for the legacy convolve_tt interface. No clipping or
    repair of a poor kernel approximation is performed. Valid observations
    satisfy y-x in the kernel box for every x in the signal box; outside
    that safe window this implementation returns NaN, never extrapolates.
    Construct outside jit; signal coefficients remain fully differentiable.
    """

    kernel: ChebTT
    domains: tuple
    sizes: tuple
    nodes: tuple
    weights: tuple
    basis: tuple
    grams: tuple
    orders: tuple
    safe_domain: jnp.ndarray
    moment_cores: tuple | None = None
    moment_groups: tuple | None = None

    def prepare(self, observation_domain=None, *, tolerance=0.):
        """Precompute polynomial integration moments for a fixed noise TT.

        The moment in y has degree at most kernel_degree, so the DCT uses
        the kernel's full coefficient count. No Gaussian formula is used.
        Optional tail truncation bounds each scalar moment's absolute error
        by tolerance times the largest coefficient l1 norm in that core.
        The default retains every coefficient. Build outside jit.
        """
        domain = np.asarray(self.safe_domain if observation_domain is None else observation_domain)
        if domain.shape != (len(self.sizes), 2) or np.any(domain[:, 0] >= domain[:, 1]):
            raise ValueError('observation_domain must contain nonempty intervals')
        if np.any(domain[:, 0] < np.asarray(self.safe_domain)[:, 0]) or np.any(domain[:, 1] > np.asarray(self.safe_domain)[:, 1]):
            raise ValueError('observation_domain must lie inside the kernel safe window')
        if tolerance < 0 or not np.isfinite(tolerance):
            raise ValueError('tolerance must be finite and nonnegative')
        prepared, groups = [], []
        for k, (u, n) in enumerate(zip(self.kernel.cores, self.sizes, strict=True)):
            m = u.coeffs.shape[-1]
            a, b = domain[k]
            ys = a+(b-a)*(np.cos(np.pi*(np.arange(m)+.5)/m)+1)/2
            x = np.asarray(self.nodes[k])
            ga, gb = self.domains[k]
            v = jnp.asarray(np.polynomial.chebyshev.chebvander((2*x-ga-gb)/(gb-ga), 2*n-2))
            def integrate(y, u=u, k=k, v=v):
                values = u((y[:, None]-self.nodes[k][None, :]).reshape(-1))
                values = values.reshape(len(y), self.orders[k], *u.shape)
                return jnp.einsum('btac,th,t->bach', values, v, self.weights[k])
            integrate = jax.jit(integrate)
            vals = np.concatenate([np.asarray(integrate(jnp.asarray(ys[j:j+16])))
                                   for j in range(0, m, 16)], axis=0)
            coeff = dct(vals, type=2, axis=0)/m
            coeff[0] *= .5
            if tolerance:
                magnitudes = np.max(np.abs(coeff).reshape(m, -1), axis=1)
                tails = np.cumsum(magnitudes[::-1])[::-1]
                cutoff = np.flatnonzero(tails <= tolerance*np.max(np.sum(np.abs(coeff), axis=0)))
                if len(cutoff):
                    coeff = coeff[:max(1, int(cutoff[0]))]
            # Pack the moment and right bond into one matrix index.
            coeff = np.moveaxis(coeff, 0, -1)
            prepared.append(ChebyshevCoeffCore(jnp.asarray(coeff.reshape(u.shape[0], u.shape[1]*(2*n-1), -1)), (a, b)))
            # Connected components of the exact structural-zero pattern.
            # Mixture/block kernels often have independent blocks; processing
            # them separately avoids dense arithmetic on their zero entries.
            pattern = np.any(np.asarray(u.coeffs) != 0, axis=-1)
            unseen = set(np.flatnonzero(pattern.any(axis=1)).tolist())
            blocks = []
            while unseen:
                left, right = {min(unseen)}, set()
                while True:
                    new_right = set(np.flatnonzero(pattern[list(left)].any(axis=0)).tolist())
                    new_left = set(np.flatnonzero(pattern[:, list(new_right)].any(axis=1)).tolist())
                    if new_left == left and new_right == right:
                        break
                    left, right = new_left, new_right
                unseen -= left
                li, ri = np.array(sorted(left)), np.array(sorted(right))
                block = coeff[li[:, None], ri[None, :]]
                blocks.append((jnp.asarray(li), jnp.asarray(ri), ChebyshevCoeffCore(
                    jnp.asarray(block.reshape(len(li), len(ri)*(2*n-1), -1)), (a, b))))
            groups.append(tuple(blocks))
        return replace(self, moment_cores=tuple(prepared), moment_groups=tuple(groups), safe_domain=jnp.asarray(domain))

    @staticmethod
    def _prepared_values(core, y, shape):
        a, b = core.domain
        t = (2*y-a-b)/(b-a)
        m = core.coeffs.shape[-1]
        # Shared basis + matrix product avoids Clenshaw's large intermediate.
        if m == 1:
            basis = jnp.ones((len(y), 1))
        else:
            def recurrence(state, unused):
                previous, current = state
                following = 2*t*current-previous
                return (current, following), following
            _, rest = jax.lax.scan(recurrence, (jnp.ones_like(t), t), None, length=m-2)
            basis = jnp.concatenate([jnp.ones_like(t)[None], t[None], rest], axis=0).T
        return (basis@core.coeffs.reshape(-1, m).T).reshape(len(y), *shape)

    @staticmethod
    def _prepared_moments(core, y, n, shape):
        h = TTConvolution._prepared_values(core, y, (*shape, 2*n-1))
        i = jnp.arange(n)
        return .5*(h[..., i[:, None]+i[None, :]]+h[..., jnp.abs(i[:, None]-i[None, :])])

    def _moments(self, k, y):
        u = self.kernel.cores[k]
        if self.moment_cores is not None:
            return self._prepared_moments(self.moment_cores[k], y, self.sizes[k], u.shape)
        z = y[:, None] - self.nodes[k][None, :]
        values = u(z.reshape(-1)).reshape(len(y), self.orders[k], *u.shape)
        weighted = values * self.weights[k][None, :, None, None]
        return jnp.einsum('btac,ti,tj->bacij', weighted, self.basis[k], self.basis[k])

    @classmethod
    def from_tt(cls, template, kernel, *, quadrature_order=None):
        if len(template.cores) != len(kernel.cores):
            raise ValueError("signal and kernel dimensions must match")
        if not all(isinstance(c, ChebyshevCoeffCore) for c in (*template.cores, *kernel.cores)):
            raise TypeError("TTConvolution requires Chebyshev coefficient cores")
        domains = tuple(c.domain for c in template.cores)
        sizes = tuple(c.coeffs.shape[-1] for c in template.cores)
        # 2Q-1 >= 2(n_signal-1)+(n_kernel-1).
        minimum = tuple(n + (u.coeffs.shape[-1]-1)//2
                        for n, u in zip(sizes, kernel.cores, strict=True))
        if quadrature_order is None:
            orders = minimum
        else:
            raw = np.broadcast_to(np.asarray(quadrature_order), (len(sizes),))
            if np.any(~np.isfinite(raw)) or np.any(raw != np.floor(raw)) or np.any(raw < minimum):
                raise ValueError(f"quadrature_order must contain integers >= {minimum}")
            orders = tuple(int(q) for q in raw)
        nodes, weights, basis, grams, safe = [], [], [], [], []
        for n, (a, b), u, q in zip(sizes, domains, kernel.cores, orders, strict=True):
            t, w = roots_legendre(q)
            nodes.append(jnp.asarray(a+(b-a)*(t+1)/2))
            weights.append(jnp.asarray(w*(b-a)/2))
            basis.append(jnp.asarray(np.polynomial.chebyshev.chebvander(t, n-1)))
            grams.append(chebyshev_gram_matrix(n, (a, b)))
            safe.append((u.domain[0]+b, u.domain[1]+a))
        if np.any(np.asarray(safe)[:, 0] > np.asarray(safe)[:, 1]):
            raise ValueError("kernel box is too narrow for fixed-window convolution")
        return cls(kernel, domains, sizes, tuple(nodes), tuple(weights), tuple(basis),
                   tuple(grams), orders, jnp.asarray(safe))

    def log_density(self, g, y):
        domains = tuple(c.domain for c in g.cores)
        concrete = not any(isinstance(v, jax.core.Tracer) for v in jax.tree.leaves(domains))
        if (concrete and domains != self.domains) or tuple(c.coeffs.shape[-1] for c in g.cores) != self.sizes:
            raise ValueError("operator and TT must have the same domains and polynomial sizes")
        y = jnp.asarray(y, dtype=jnp.float64)
        single = y.ndim == 1
        if single:
            y = y[None, :]
        if y.ndim != 2 or y.shape[1] != len(self.sizes):
            raise ValueError("y must have shape (d,) or (batch, d)")
        state = jnp.ones((len(y), 1, 1, 1), dtype=y.dtype)
        norm = jnp.ones((1, 1, 1), dtype=y.dtype)
        log_value, log_norm = jnp.zeros(len(y)), jnp.zeros(1)
        for k, (core, u) in enumerate(zip(g.cores, self.kernel.cores, strict=True)):
            c = core.coeffs
            c = c / jax.lax.stop_gradient(jnp.maximum(jnp.max(jnp.abs(c)), 1e-300))
            def transfer(s, moments, c=c):
                left = jnp.einsum('bpqa,pki->bqkai', s, c)
                left = jnp.einsum('bqkai,bacij->bqkcj', left, moments)
                return jnp.einsum('bqkcj,qlj->bklc', left, c)
            if self.moment_groups is not None and len(self.moment_groups[k]) > 1:
                next_state = jnp.zeros((len(y), c.shape[1], c.shape[1], u.shape[1]))
                for li, ri, moment_core in self.moment_groups[k]:
                    moments = self._prepared_moments(moment_core, y[:, k], self.sizes[k], (len(li), len(ri)))
                    block = transfer(jnp.take(state, li, axis=-1), moments)
                    next_state = next_state.at[:, :, :, ri].set(block)
                state = next_state
            else:
                state = transfer(state, self._moments(k, y[:, k]))
            scale = jax.lax.stop_gradient(jnp.max(jnp.abs(state), axis=(1, 2, 3)))
            scale = jnp.maximum(scale, jnp.finfo(y.dtype).tiny)
            state = state / scale[:, None, None, None]
            log_value = log_value + jnp.log(scale)
            norm = _transfer(norm, c, self.grams[k][None, :, :])
            norm, scale = _rescale(norm)
            log_norm = log_norm + scale
        result = log_value + jnp.log(state[:, 0, 0, 0]) - log_norm - jnp.log(norm[0, 0, 0])
        valid = jnp.all((y >= self.safe_domain[:, 0]) & (y <= self.safe_domain[:, 1]), axis=1)
        if not concrete:
            valid = valid & jnp.all(jnp.asarray(domains) == jnp.asarray(self.domains))
        result = jnp.where(valid, result, jnp.nan)
        return result[0] if single else result
