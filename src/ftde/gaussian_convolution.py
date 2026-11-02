"""Direct Gaussian convolution of a squared TT, without forming its square.

For independent Gaussian noise, each observation supplies a positive weighted
Chebyshev Gram matrix. Contracting these matrices with the two copies of g
costs O(batch * d * (r**3*n + r**2*n**2 + Q*n**2)), rather than storing rank-r**2
convolution cores. Fixed Gauss--Legendre integration uses the analytic noise
kernel; neither the kernel nor the output is interpolated. This is a
convergent quadrature approximation, not an exact Gaussian integral.
"""

from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np
from scipy.special import roots_legendre

from .cheb_gram import chebyshev_gram_matrix


def _rescale(state):
    # Stop-gradient only through the arbitrary scale: the compensating log
    # scale makes this the same derivative as the unscaled contraction.
    scale = jax.lax.stop_gradient(jnp.max(jnp.abs(state), axis=(-2, -1)))
    scale = jnp.maximum(scale, jnp.finfo(state.dtype).tiny)
    return state / scale[..., None, None], jnp.log(scale)


def _transfer(state, coeffs, gram):
    # Explicit contraction order is essential: never create an r**4 tensor.
    left = jnp.einsum("bpq,pki->bqki", state, coeffs)
    left = jnp.einsum("bqki,bij->bqkj", left, gram)
    state = jnp.einsum("bqkj,qlj->bkl", left, coeffs)
    return 0.5 * (state + state.swapaxes(-1, -2))


@dataclass(frozen=True)
class GaussianConvolution:
    """Fixed diagonal-Gaussian forward operator for one TT structure.

    Construct outside jit. ``quadrature_order`` is either a scalar or a
    per-axis sequence (at least the coefficient count on each axis). The
    default accounts for polynomial degree and kernel width; verify
    convergence by doubling it. ``positive``
    uses sums of congruences directly (more work, useful as an independent
    check when high-degree weighted Gram matrices are ill-conditioned).
    The signal is zero outside its box; observations may be anywhere.
    """

    domains: tuple
    sizes: tuple
    std: tuple
    nodes: tuple
    log_weights: tuple
    basis: tuple
    grams: tuple
    orders: tuple
    positive: bool = False

    @classmethod
    def from_tt(cls, template, noise_std, *, quadrature_order=None, positive=False):
        std = np.asarray(noise_std, dtype=np.float64)
        d = len(template.cores)
        if std.shape != (d,) or not np.all(np.isfinite(std) & (std > 0)):
            raise ValueError(f"noise_std must contain {d} finite positive standard deviations")
        domains = tuple(core.domain for core in template.cores)
        sizes = tuple(core.coeffs.shape[-1] for core in template.cores)
        if quadrature_order is None:
            orders = tuple(max(2 * n, n + 16, int(np.ceil(12 * (b - a) / s)))
                           for n, (a, b), s in zip(sizes, domains, std, strict=True))
        else:
            raw = np.broadcast_to(np.asarray(quadrature_order), (d,))
            if np.any(~np.isfinite(raw)) or np.any(raw != np.floor(raw)) or np.any(raw < 2):
                raise ValueError("quadrature_order must contain integers >= 2")
            orders = tuple(int(v) for v in raw)
        if any(q < n for q, n in zip(orders, sizes, strict=True)):
            raise ValueError("quadrature_order must be >= the number of coefficients on each axis")
        nodes, log_weights, basis, grams = [], [], [], []
        for n, (a, b), order in zip(sizes, domains, orders, strict=True):
            if not np.isfinite(a + b) or a >= b:
                raise ValueError("signal domains must be finite increasing intervals")
            t, w = roots_legendre(order)
            nodes.append(jnp.asarray(a + (b - a) * (t + 1) / 2))
            log_weights.append(jnp.asarray(np.log(w * (b - a) / 2)))
            basis.append(jnp.asarray(np.polynomial.chebyshev.chebvander(t, n - 1)))
            grams.append(chebyshev_gram_matrix(n, (a, b)))
        return cls(domains, sizes, tuple(std), tuple(nodes), tuple(log_weights),
                   tuple(basis), tuple(grams), orders, positive)

    def log_density(self, g, y):
        """Differentiable log((Gaussian * g**2)(y) / integral g**2)."""
        domains = tuple(c.domain for c in g.cores)
        concrete_domains = not any(isinstance(v, jax.core.Tracer) for v in jax.tree.leaves(domains))
        if (concrete_domains and domains != self.domains) or tuple(
            c.coeffs.shape[-1] for c in g.cores
        ) != self.sizes:
            raise ValueError("operator and TT must have the same domains and polynomial sizes")
        y = jnp.asarray(y, dtype=jnp.float64)
        single = y.ndim == 1
        if single:
            y = y[None, :]
        if y.ndim != 2 or y.shape[1] != len(self.sizes):
            raise ValueError("y must have shape (d,) or (batch, d)")
        state = jnp.ones((y.shape[0], 1, 1), dtype=y.dtype)
        norm = jnp.ones((1, 1, 1), dtype=y.dtype)
        log_value = jnp.zeros(y.shape[0], dtype=y.dtype)
        log_norm = jnp.zeros(1, dtype=y.dtype)
        for k, core in enumerate(g.cores):
            c = core.coeffs
            c = c / jax.lax.stop_gradient(jnp.maximum(jnp.max(jnp.abs(c)), 1e-300))
            lw = (self.log_weights[k][None, :] - jnp.log(self.std[k] * np.sqrt(2 * np.pi))
                  - 0.5 * ((y[:, k, None] - self.nodes[k]) / self.std[k]) ** 2)
            mass = jax.scipy.special.logsumexp(lw, axis=-1)
            weights = jnp.exp(lw - mass[:, None])
            v = self.basis[k]
            if self.positive:
                values = jnp.einsum("pki,ti->tpk", c, v)
                left = jnp.einsum("bpq,tpk->btqk", state, values)
                left = left * weights[:, :, None, None]
                state = jnp.einsum("btqk,tql->bkl", left, values)
            else:
                weighted = weights[:, :, None] * v[None, :, :]
                gram = jnp.einsum("bti,tj->bij", weighted, v)
                state = _transfer(state, c, gram)
            state, scale = _rescale(state)
            log_value = log_value + mass + scale
            norm = _transfer(norm, c, self.grams[k][None, :, :])
            norm, scale = _rescale(norm)
            log_norm = log_norm + scale
        # Do not silently clip a negative forward value: it signals failed
        # numerical accuracy, and should be caught rather than trained around.
        result = log_value + jnp.log(state[:, 0, 0]) - log_norm - jnp.log(norm[0, 0, 0])
        if not concrete_domains:
            # A whole ChebTT passed as a jit argument traces its domains too.
            # A mismatched box must not silently reuse this operator's nodes.
            matches = jnp.all(jnp.asarray(domains) == jnp.asarray(self.domains))
            result = jnp.where(matches, result, jnp.nan)
        return result[0] if single else result
