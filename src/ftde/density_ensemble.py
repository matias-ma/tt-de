"""Convex mixtures of normalized squared Chebyshev TT densities.

Average densities, not amplitudes or TT cores: signs and bond gauges must
not change an ensemble. Every component remains an optimized ChebTT.
"""

from dataclasses import dataclass

import jax.numpy as jnp
import numpy as np
from jax.scipy.special import logsumexp

from .marginals import squared_pair_grid
from .tt_density import log_density_tt


def normalized_weights(weights, count):
    values = np.ones(count) if weights is None else np.asarray(weights, dtype=float)
    if count == 0 or values.shape != (count,) or not np.all(np.isfinite(values) & (values >= 0)) or values.max() <= 0:
        raise ValueError('Need one finite nonnegative weight per component and a positive total')
    values = values/values.max()
    return values/values.sum()


def mixture_log_density(component_logs, weights=None):
    """Combine component log densities with component index on axis zero."""
    logs = jnp.asarray(component_logs)
    weights = normalized_weights(weights, logs.shape[0])
    shape = (len(weights),)+(1,)*(logs.ndim-1)
    return logsumexp(logs+jnp.log(jnp.asarray(weights)).reshape(shape), axis=0)


@dataclass(frozen=True)
class SquaredTTMixture:
    models: tuple
    weights: tuple | None = None

    def __post_init__(self):
        weights = normalized_weights(self.weights, len(self.models))
        if len({len(g.cores) for g in self.models}) != 1:
            raise ValueError('Components must have the same dimension')
        object.__setattr__(self, 'weights', tuple(weights))

    def log_density(self, x):
        x = jnp.asarray(x)
        logs = []
        for model in self.models:
            domain = jnp.asarray(model.domain)
            inside = jnp.all((x >= domain[:, 0]) & (x <= domain[:, 1]), axis=-1)
            logs.append(jnp.where(inside, log_density_tt(model, x), -jnp.inf))
        return mixture_log_density(jnp.stack(logs), self.weights)

    def convolved_log_density(self, operators, y):
        if len(operators) != len(self.models):
            raise ValueError('Need one compatible convolution operator per component')
        return mixture_log_density(jnp.stack([op.log_density(model, y)
            for op, model in zip(operators, self.models, strict=True)]), self.weights)

    def pair_grid(self, i, j, xs, ys):
        return sum(w*squared_pair_grid(model, i, j, xs, ys)
                   for w, model in zip(self.weights, self.models, strict=True))
