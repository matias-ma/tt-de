"""Direct density TTs with joint clipping *inside* the likelihood.

The model is max(f(x), 0) / integral max(f, 0). Clipping destroys the
separable TT integration formula. This reference implementation uses full
four-dimensional Gauss--Legendre cubature, and therefore deliberately does
not pretend to scale to high dimension. Increase the order to audit accuracy.
"""

from dataclasses import dataclass

import jax.numpy as jnp
import numpy as np
from scipy.special import roots_legendre


@dataclass(frozen=True, eq=False)
class ClippedTTQuadrature4D:
    domain: np.ndarray
    nodes: tuple
    weights: tuple
    bases: tuple
    pair_nodes: tuple
    pair_weights: tuple
    covariances: jnp.ndarray
    mixture_weights: jnp.ndarray

    @classmethod
    def from_tt(cls, template, covariances, mixture_weights, *, order=65):
        domain = np.asarray(template.domain)
        if domain.shape != (4, 2):
            raise ValueError('Full clipped cubature currently requires four dimensions')
        covs, mix = np.asarray(covariances), np.asarray(mixture_weights)
        if covs.shape != (len(mix), 4, 4) or np.any(mix < 0) or not np.isclose(mix.sum(), 1):
            raise ValueError('Expected normalized weights and (components, 4, 4) covariances')
        if not np.allclose(covs[:, :2, 2:], 0) or not np.allclose(covs[:, 2:, :2], 0):
            raise ValueError('Noise must be block diagonal in pairs (0,1), (2,3)')
        if not np.allclose(covs, covs.transpose(0, 2, 1)) or np.any(np.linalg.eigvalsh(covs) <= 0):
            raise ValueError('Noise covariance matrices must be symmetric positive definite')
        if order < max(c.coeffs.shape[-1] for c in template.cores):
            raise ValueError('Use at least as many quadrature nodes as polynomial modes')
        t, w = roots_legendre(order)
        nodes = tuple(jnp.asarray(a+(b-a)*(t+1)/2) for a, b in domain)
        weights = tuple(jnp.asarray(w*(b-a)/2) for a, b in domain)
        bases = tuple(jnp.asarray(np.polynomial.chebyshev.chebvander(t, c.coeffs.shape[-1]-1))
                      for c in template.cores)
        pairs = tuple(jnp.stack(jnp.meshgrid(nodes[k], nodes[k+1], indexing='ij'), axis=-1).reshape(-1, 2)
                      for k in (0, 2))
        pair_weights = tuple((weights[k][:, None]*weights[k+1][None]).reshape(-1) for k in (0, 2))
        return cls(domain, nodes, weights, bases, pairs, pair_weights, jnp.asarray(covs), jnp.asarray(mix))

    def raw_grid(self, model):
        values = [jnp.einsum('tn,abn->tab', basis, core.coeffs)
                  for basis, core in zip(self.bases, model.cores, strict=True)]
        left = jnp.einsum('ia,jab->ijb', values[0][:, 0], values[1]).reshape(-1, values[1].shape[-1])
        right = jnp.einsum('kab,lb->akl', values[2], values[3][:, :, 0]).reshape(values[2].shape[1], -1)
        return left@right

    def weighted_grid(self, model):
        raw = self.raw_grid(model)
        weights = self.pair_weights[0][:, None]*self.pair_weights[1][None]
        positive = jnp.maximum(raw, 0)*weights
        mass = jnp.sum(positive)
        negative_mass = -jnp.sum(jnp.minimum(raw, 0)*weights)
        return positive/mass, mass, negative_mass

    def noise_factors(self, observations):
        y = jnp.atleast_2d(observations)
        factors = []
        for k, nodes in zip((0, 2), self.pair_nodes, strict=True):
            cov = self.covariances[:, k:k+2, k:k+2]
            delta = y[None, :, None, k:k+2]-nodes[None, None]
            exponent = jnp.einsum('cbni,cij,cbnj->cbn', delta, jnp.linalg.inv(cov), delta)
            factors.append(jnp.exp(-.5*exponent)/(2*jnp.pi*jnp.sqrt(jnp.linalg.det(cov)))[:, None, None])
        return tuple(factors)

    def convolve_weighted(self, weighted, factors):
        left, right = factors
        # Each component shares its label across both pairs (not independent
        # marginal noise mixtures). Contraction clips the JOINT density first.
        return sum(w*jnp.sum((a@weighted)*b, axis=1)
                   for w, a, b in zip(self.mixture_weights, left, right, strict=True))

    def observed_density(self, model, observations):
        weighted, _, _ = self.weighted_grid(model)
        return self.convolve_weighted(weighted, self.noise_factors(observations))

    def density(self, model, points):
        _, mass, _ = self.weighted_grid(model)
        x = jnp.atleast_2d(points)
        inside = jnp.all((x >= self.domain[:, 0]) & (x <= self.domain[:, 1]), axis=1)
        return jnp.where(inside, jnp.maximum(model(x), 0)/mass, 0.)

    def pair_marginals(self, model):
        weighted, _, _ = self.weighted_grid(model)
        n = len(self.nodes[0])
        return tuple((jnp.sum(weighted, axis=axis)/pw).reshape(n, n).T
                     for axis, pw in zip((1, 0), self.pair_weights, strict=True))
