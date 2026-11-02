"""Conditional inverse-CDF sampling of normalized squared Chebyshev TTs.

Every conditional is a univariate polynomial. Integrating its Chebyshev
coefficients avoids a discretized histogram or a learned sampling proposal.
"""

import jax
import jax.numpy as jnp
import numpy as np

from .cheb_gram import chebyshev_gram_matrix
from .gaussian_convolution import _transfer


def make_squared_tt_moments(template):
    """Return exact polynomial marginal means/variances for sampler checks."""
    moments = []
    for core in template.cores:
        n = core.coeffs.shape[-1]
        nodes, weights = np.polynomial.legendre.leggauss(n+1)
        a, b = core.domain
        physical = a+(nodes+1)*(b-a)/2
        weights = weights*(b-a)/2
        basis = np.polynomial.chebyshev.chebvander(nodes, n-1)
        moments.append(tuple(jnp.asarray(basis.T@((weights*physical**power)[:, None]*basis))
                             for power in range(3)))

    @jax.jit
    def evaluate(params):
        cores = tuple(c/jnp.max(jnp.abs(c)) for c in params)
        right = [None]*(len(cores)+1)
        right[-1] = jnp.ones((1, 1))
        for k in range(len(cores)-1, -1, -1):
            value = jnp.einsum("abn,nm,bd,cdm->ac", cores[k], moments[k][0], right[k+1], cores[k],
                               optimize="optimal")
            right[k] = value/jnp.max(jnp.abs(value))
        left = jnp.ones((1, 1, 1))
        means, variances = [], []
        for k, core in enumerate(cores):
            states = [_transfer(left, core, gram[None]) for gram in moments[k]]
            integrals = [jnp.sum(state[0]*right[k+1]) for state in states]
            mean = integrals[1]/integrals[0]
            means.append(mean)
            variances.append(integrals[2]/integrals[0]-mean**2)
            left = states[0]/jnp.max(jnp.abs(states[0]))
        return jnp.stack(means), jnp.stack(variances)

    return evaluate


def make_squared_tt_sampler(template, *, bisection_steps=36):
    """Return a JIT sampler ``sample(coefficients, uniforms) -> points``.

Uniforms must be in (0, 1), with shape (batch, dimension). The box and core
shapes are fixed by template. Bisection error is at most box width / 2**steps.
No target density or training observations enter this sampler.
"""
    if bisection_steps < 20:
        raise ValueError("at least 20 bisection iterations are required")
    grams, integrals, sums, differences = [], [], [], []
    for core in template.cores:
        n = core.coeffs.shape[-1]
        grams.append(chebyshev_gram_matrix(n, core.domain))
        integrals.append(jnp.asarray(np.polynomial.chebyshev.chebint(
            np.eye(2*n-1), axis=0, lbnd=-1)))
        index = np.arange(n)
        sums.append(jnp.asarray((index[:, None]+index).ravel()))
        differences.append(jnp.asarray(np.abs(index[:, None]-index).ravel()))

    @jax.jit
    def sample(params, uniforms):
        if uniforms.ndim != 2 or uniforms.shape[1] != len(params):
            raise ValueError("uniforms must have shape (batch, dimension)")
        cores = tuple(c/jnp.max(jnp.abs(c)) for c in params)
        right = [None]*(len(cores)+1)
        right[-1] = jnp.ones((1, 1))
        for k in range(len(cores)-1, -1, -1):
            c = cores[k]
            value = jnp.einsum("abn,nm,bd,cdm->ac", c, grams[k], right[k+1], c,
                               optimize="optimal")
            right[k] = value/jnp.max(jnp.abs(value))
        prefix = jnp.ones((len(uniforms), 1))
        columns = []
        for k, c in enumerate(cores):
            n = c.shape[-1]
            conditional = jnp.einsum("ba,arn->brn", prefix, c)
            products = jnp.einsum("bri,rs,bsj->bij", conditional, right[k+1], conditional)
            flat = products.reshape(len(uniforms), -1)
            coefficients = jnp.zeros((len(uniforms), 2*n-1))
            coefficients = coefficients.at[:, sums[k]].add(.5*flat)
            coefficients = coefficients.at[:, differences[k]].add(.5*flat)
            integrated = coefficients @ integrals[k].T
            mass = integrated.sum(axis=1)
            integrated = integrated/mass[:, None]

            def cdf(x, n=n, integrated=integrated):
                # Chebyshev basis evaluation is stable on [-1, 1].
                basis = jnp.cos(jnp.arccos(x)[:, None]*jnp.arange(2*n)[None])
                return jnp.sum(integrated*basis, axis=1)

            def bisect(_, bounds, cdf=cdf, k=k):
                lo, hi = bounds
                mid = .5*(lo+hi)
                left = cdf(mid) < uniforms[:, k]
                return jnp.where(left, mid, lo), jnp.where(left, hi, mid)

            lo, hi = jax.lax.fori_loop(0, bisection_steps, bisect,
                                       (-jnp.ones(len(uniforms)), jnp.ones(len(uniforms))))
            t = .5*(lo+hi)
            basis = jnp.cos(jnp.arccos(t)[:, None]*jnp.arange(n)[None])
            prefix = jnp.einsum("brn,bn->br", conditional, basis)
            prefix = prefix/jnp.maximum(jnp.max(jnp.abs(prefix), axis=1, keepdims=True), 1e-300)
            a, b = template.cores[k].domain
            columns.append(a+(t+1)*(b-a)/2)
        return jnp.stack(columns, axis=1)

    return sample
