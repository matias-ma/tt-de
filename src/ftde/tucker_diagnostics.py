"""Memory-bounded polynomial integrals for higher-rank Tucker diagnostics."""
import jax
import jax.numpy as jnp
import numpy as np
from scipy.special import roots_legendre

from .cheb_gram import chebyshev_gram_matrix
from .tucker_density import _mode_product, squared_integral_tucker


def squared_pair_grid_tucker(model, i, j, xs, ys):
    """Exact pair marginal via a small pair-rank covariance matrix."""
    if not 0 <= i < j < len(model.factors):
        raise ValueError('require 0 <= i < j < dimension')
    transformed = model.core
    for k, f in enumerate(model.factors):
        if k not in (i, j):
            c = f.core.coeffs[0]
            gram = c@chebyshev_gram_matrix(c.shape[-1], f.domain)@c.T
            transformed = _mode_product(transformed, gram, k)
    order = (i, j)+tuple(k for k in range(len(model.factors)) if k not in (i, j))
    ranks = model.core.shape[i], model.core.shape[j]
    left = jnp.transpose(model.core, order).reshape(np.prod(ranks), -1)
    right = jnp.transpose(transformed, order).reshape(np.prod(ranks), -1)
    covariance = (left@right.T).reshape(*ranks, *ranks)
    vx = model.factors[i](jnp.clip(jnp.asarray(xs), *model.domain[i]))
    vy = model.factors[j](jnp.clip(jnp.asarray(ys), *model.domain[j]))
    grid = jnp.einsum('xa,xc,abcd,yb,yd->yx', vx, vx, covariance, vy, vy, optimize='optimal')
    inside_x = (np.asarray(xs) >= model.domain[i, 0]) & (np.asarray(xs) <= model.domain[i, 1])
    inside_y = (np.asarray(ys) >= model.domain[j, 0]) & (np.asarray(ys) <= model.domain[j, 1])
    return np.asarray(jnp.where(inside_y[:, None] & inside_x[None], grid/squared_integral_tucker(model), 0.))


def fourth_power_integral_4d(model, *, block_size=128):
    """Exact integral of the fourth power, using blocked tensor quadrature.

    A degree n-1 amplitude needs 2*n-1 Gauss nodes per axis. Pairing axes
    turns the evaluation into matrix products. Unlike a converted TT's
    quartic environment, memory does not grow as the eighth power of rank.
    """
    if len(model.factors) != 4 or block_size < 1:
        raise ValueError('require a four-dimensional model and positive block_size')
    values, weights = [], []
    for factor in model.factors:
        coeff = np.asarray(factor.core.coeffs[0])
        nodes, w = roots_legendre(2*coeff.shape[-1]-1)
        values.append(jnp.asarray(np.polynomial.chebyshev.chebvander(nodes, coeff.shape[-1]-1)@coeff.T))
        weights.append(jnp.asarray(w*np.ptp(factor.domain)/2))
    left = jnp.einsum('ia,jb->ijab', *values[:2]).reshape(len(values[0])*len(values[1]), -1)
    right = jnp.einsum('ia,jb->ijab', *values[2:]).reshape(len(values[2])*len(values[3]), -1)
    wl = jnp.outer(*weights[:2]).ravel()
    wr = jnp.outer(*weights[2:]).ravel()
    middle = model.core.reshape(left.shape[1], right.shape[1])@right.T

    @jax.jit
    def integrate(a, w):
        return jnp.sum(w[:, None]*(a@middle)**4*wr[None, :])

    return sum(float(integrate(left[start:start+block_size], wl[start:start+block_size]))
               for start in range(0, len(left), block_size))


def product_measure_integral(model, nodes, weights, *, batch_size=16):
    """Integrate amplitude squared against batched product quadrature rules.

    nodes and weights have shape (measure, dimension, quadrature_order).
    Applies each factor Gram matrix to one dense core; no TT conversion.
    """
    nodes, weights = np.asarray(nodes), np.asarray(weights)
    if nodes.shape != weights.shape or nodes.ndim != 3 or nodes.shape[1] != len(model.factors):
        raise ValueError('nodes/weights must have matching (measure, dimension, order) shapes')

    @jax.jit
    def integrate(grams):
        state = jnp.broadcast_to(model.core, (grams[0].shape[0],)+model.core.shape)
        for k, gram in enumerate(grams):
            state = jnp.moveaxis(state, k+1, -1)
            state = jnp.einsum('b...s,brs->b...r', state, gram)
            state = jnp.moveaxis(state, -1, k+1)
        return jnp.sum((state*model.core).reshape(len(state), -1), axis=1)

    result = []
    for start in range(0, len(nodes), batch_size):
        grams = []
        for k, f in enumerate(model.factors):
            a, b = f.domain
            x = nodes[start:start+batch_size, k]
            c = np.asarray(f.core.coeffs[0])
            basis = np.polynomial.chebyshev.chebvander(np.clip(2*(x-a)/(b-a)-1, -1, 1), c.shape[-1]-1)
            v = basis@c.T
            w = weights[start:start+batch_size, k]*((x >= a) & (x <= b))
            grams.append(jnp.asarray(np.einsum('bqr,bq,bqs->brs', v, w, v)))
        result.append(np.asarray(integrate(tuple(grams))))
    return np.concatenate(result)


def density_squared_integral_qmc(model, observations, *, power=17, replicates=8,
                                 centers_count=512, bandwidth=.25, batch_size=256):
    """Randomized importance-QMC estimate of integral (g² / Z)².

    The proposal is 95% Gaussian KDE of observations and 5% uniform on the
    model box. The uniform component covers every point of the support.
    Independent scrambled Sobol replicates supply an empirical standard
    error; this is a numerical estimate, not exact polynomial quadrature.
    """
    from scipy.special import ndtri
    from scipy.stats import qmc
    if replicates < 2 or bandwidth <= 0:
        raise ValueError('require at least two replicates and positive bandwidth')
    observations = np.asarray(observations)
    centers = observations[np.random.default_rng(728).choice(len(observations), min(centers_count, len(observations)), replace=False)]
    box = np.asarray(model.domain)
    dim = len(box)
    volume = np.prod(np.ptp(box, axis=1))
    z = squared_integral_tucker(model)
    c = jnp.asarray(centers)

    @jax.jit
    def contributions(x):
        inside = jnp.all((x >= box[:, 0]) & (x <= box[:, 1]), axis=1)
        difference = (x[:, None, :]-c[None, :, :])/bandwidth
        kde = jnp.mean(jnp.exp(-.5*jnp.sum(difference**2, axis=-1)), axis=1)/(2*jnp.pi*bandwidth**2)**(dim/2)
        proposal = .95*kde+.05*inside/volume
        # Outside the box the model density is zero. Clip before evaluating
        # the polynomial to avoid extrapolation overflow in Gaussian tails.
        density = jnp.where(inside, model(jnp.clip(x, box[:, 0], box[:, 1]))**2/z, 0.)
        return density**2/proposal

    estimates, half = [], []
    for seed in range(replicates):
        u = qmc.Sobol(dim+2, scramble=True, seed=9472+seed).random_base2(power)
        indices = np.minimum((u[:, 1]*len(centers)).astype(int), len(centers)-1)
        gaussian = centers[indices]+bandwidth*ndtri(np.clip(u[:, 2:], 1e-15, 1-1e-15))
        uniform = box[:, 0]+u[:, 2:]*np.ptp(box, axis=1)
        points = np.where((u[:, 0] < .05)[:, None], uniform, gaussian)
        values = np.concatenate([np.asarray(contributions(jnp.asarray(points[start:start+batch_size])))
                                 for start in range(0, len(points), batch_size)])
        estimates.append(float(values.mean()))
        half.append(float(values[:len(values)//2].mean()))
    return {'value': float(np.mean(estimates)), 'standard_error': float(np.std(estimates, ddof=1)/np.sqrt(replicates)),
            'replicate_estimates': estimates, 'half_sample_value': float(np.mean(half)),
            'samples': replicates*2**power, 'method': 'scrambled Sobol importance sampling; 95% observed KDE + 5% uniform',
            'bandwidth': bandwidth, 'centers': len(centers)}
