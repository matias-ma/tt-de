"""Experimental signed density-TT forward operator, without amplitude squaring.

Positive integral does not make a signed TT a valid density. Callers must
audit negativity and any subsequent clipping separately.
"""

from dataclasses import dataclass

import jax
import jax.numpy as jnp

from .cheb_gram import chebyshev_moment_vector
from .tt_convolution import TTConvolution


def _moment_values(core, y):
    """Evaluate shared Chebyshev basis once, then use a matrix product."""
    a, b = core.domain
    t = (2*y-a-b)/(b-a)
    m = core.coeffs.shape[-1]
    if m == 1:
        basis = jnp.ones((len(y), 1))
    else:
        def recurrence(state, unused):
            previous, current = state
            following = 2*t*current-previous
            return (current, following), following
        _, rest = jax.lax.scan(recurrence, (jnp.ones_like(t), t), None, length=m-2)
        basis = jnp.concatenate([jnp.ones_like(t)[None], t[None], rest], axis=0).T
    return basis@core.coeffs.reshape(-1, m).T


@dataclass(frozen=True)
class LinearTTConvolution:
    prepared: TTConvolution

    @classmethod
    def from_tt(cls, template, kernel, observation_domain, *, quadrature_order=None):
        operator = TTConvolution.from_tt(template, kernel, quadrature_order=quadrature_order)
        return cls(operator.prepare(observation_domain, tolerance=1e-14))

    def density(self, model, points):
        y = jnp.atleast_2d(points)
        if y.ndim != 2 or y.shape[1] != len(self.prepared.sizes):
            raise ValueError('points must have shape (batch, dimension)')
        if tuple(c.coeffs.shape[-1] for c in model.cores) != self.prepared.sizes:
            raise ValueError('model and prepared operator must have matching polynomial sizes')
        domains = jnp.asarray([c.domain for c in model.cores])
        same_domain = jnp.all(domains == jnp.asarray(self.prepared.domains))
        if not isinstance(same_domain, jax.core.Tracer) and not bool(same_domain):
            raise ValueError('model and prepared operator must have matching domains')
        state = jnp.ones((len(y), 1, 1))
        norm = jnp.ones(1)
        log_scale = jnp.zeros(len(y))
        log_norm_scale = jnp.array(0.)
        for k, (core, moment, kernel) in enumerate(zip(model.cores, self.prepared.moment_cores,
                                                       self.prepared.kernel.cores, strict=True)):
            n = core.coeffs.shape[-1]
            c = core.coeffs/jax.lax.stop_gradient(jnp.maximum(jnp.max(jnp.abs(core.coeffs)), 1e-300))
            raw = _moment_values(moment, y[:, k]).reshape(len(y), *kernel.shape, 2*n-1)[..., :n]
            state = jnp.einsum('bla,lri,baci->brc', state, c, raw)
            scale = jax.lax.stop_gradient(jnp.maximum(jnp.max(jnp.abs(state), axis=(1, 2)), 1e-300))
            state, log_scale = state/scale[:, None, None], log_scale+jnp.log(scale)
            norm = norm@jnp.einsum('abi,i->ab', c, chebyshev_moment_vector(n, core.domain))
            scale = jax.lax.stop_gradient(jnp.maximum(jnp.max(jnp.abs(norm)), 1e-300))
            norm, log_norm_scale = norm/scale, log_norm_scale+jnp.log(scale)
        result = jnp.exp(log_scale-log_norm_scale)*state[:, 0, 0]/norm[0]
        inside = jnp.all((y >= self.prepared.safe_domain[:, 0]) & (y <= self.prepared.safe_domain[:, 1]), axis=1)
        return jnp.where(inside & same_domain & (norm[0] > 0), result, jnp.nan)

    def log_density(self, model, points):
        return jnp.log(self.density(model, points))
