"""Native Tucker counterpart of the direct TT Hessian/likelihood trainer."""
import jax
import jax.numpy as jnp
import numpy as np
from chebttge.chebtt import ChebTT
from chebttge.cores import ChebyshevCoeffCore

from .direct_training import DirectTrainingProblem, chebyshev_features
from .mode_schedule import mode_masks
from .tucker_density import _mode_product, rebuild_tucker
from .tucker_operations import integrated_energies_tucker, load_tucker, save_tucker


class TuckerTrainingProblem(DirectTrainingProblem):
    """Same density, Sobolev geometry, Adam and held-out selection as TT.

    Optimize a dense core and all univariate factors. ``forward`` optionally
    supplies prescribed-noise convolution. The core is never squared.
    """

    def __init__(self, initial, *, forward=None, **kwargs):
        # Reuse the exact per-coordinate Gram/derivative/whitening machinery.
        proxy = ChebTT(cores=tuple(ChebyshevCoeffCore(f.core.coeffs[:, :1], f.domain) for f in initial.factors), domain=initial.domain)
        super().__init__(proxy, **kwargs)
        if self.objective not in ('nll', 'logpen'):
            raise ValueError('Tucker supports nll/logpen and amplitude Sobolev penalties')
        self.initial, self.forward = initial, forward
        self.params = (initial.core,)+tuple(f.core.coeffs[0]@a for f, a in zip(initial.factors, self.factors, strict=True))
        self.energies = jax.jit(lambda p: integrated_energies_tucker(p, self.grams, self.first, self.second))
        self.logs = jax.jit(self._logs)

    save_model = staticmethod(save_tucker)
    load_model = staticmethod(load_tucker)

    def parameter_masks(self, counts):
        return (jnp.asarray(1.),)+mode_masks(self.sizes, counts)

    def inactive_max(self, params, counts):
        return super().inactive_max(params[1:], counts)

    def model(self, params):
        return rebuild_tucker(self.initial, (params[0], [c@inv for c, inv in zip(params[1:], self.inverse, strict=True)]))

    def features(self, observations):
        if self.forward is not None:
            observations = np.asarray(observations)
            if observations.ndim != 2 or observations.shape[1] != len(self.domain) or not np.isfinite(observations).all():
                raise ValueError('Invalid noisy observations')
            return (jnp.asarray(observations),)
        return tuple(jnp.asarray(a@inv.T) for a, inv in zip(
            chebyshev_features(observations, self.domain, self.sizes), self.inverse, strict=True))

    def _logs(self, params, features):
        if self.forward is not None:
            return self.forward.log_density(self.model(params), features[0])
        state = params[0]
        for k, (c, basis) in enumerate(zip(params[1:], features, strict=True)):
            values = basis@c.T
            state = jnp.einsum('br,r...->b...', values, state) if k == 0 else jnp.einsum('br,br...->b...', values, state)
        return jnp.log(state**2+1e-300)-jnp.log(self.energies(params)[0])

    def marginal_penalty(self, params):
        if not self.marginal_strength:
            return jnp.asarray(0.)
        core, *factors = params
        norm = self.energies(params)[0]
        grams = [c@g@c.T for c, g in zip(factors, self.grams, strict=True)]
        result = jnp.asarray(0.)
        for k, (c, (b, db, ddb, w)) in enumerate(zip(factors, self.marginal_geometry, strict=True)):
            transformed = core
            for j, gram in enumerate(grams):
                if j != k:
                    transformed = _mode_product(transformed, gram, j)
            left = jnp.moveaxis(core, k, 0).reshape(core.shape[k], -1)
            right = jnp.moveaxis(transformed, k, 0).reshape(core.shape[k], -1)
            h = c.T@(left@right.T)@c
            second = 2*(jnp.sum((ddb@h)*b, axis=1)+jnp.sum((db@h)*db, axis=1))/norm
            result += jnp.sum(w*second**2)
        return self.marginal_strength*result

    def loss(self, params, features):
        norm, h1, axis, hessian = self.energies(params)
        penalty = {'h1': h1, 'axis_h2': axis, 'hessian': hessian}[self.regularizer]/norm
        regularizer = jnp.log1p(self.strength*penalty) if self.objective == 'logpen' else self.strength*penalty
        return -jnp.mean(self._logs(params, features))+regularizer+self.marginal_penalty(params)
