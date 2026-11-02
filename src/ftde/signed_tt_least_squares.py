"""Direct signed Chebyshev-TT fitting by alternating quadratic minimization.

The current solver is explicitly four-dimensional with prescribed Gaussian
noise block diagonal in coordinate pairs. No signal density is fitted before
TT optimization. It minimizes

    integral (U*f)^2 - 2 mean_data(U*f) + lambda sum_k s_k^4 integral(f_kk)^2

subject to integral(f)=1 and a fixed TT rank across the middle bond. Negative
values are allowed throughout. Final clipping is a separate measured change
of estimator, not hidden in this objective. Internal linear coordinates
whiten the Chebyshev mass matrix and diagonalize the derivative matrix;
saved/evaluated TTs use the original Chebyshev coefficient basis.
"""

from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np
from chebttge.chebtt import ChebTT
from chebttge.cores import ChebyshevCoeffCore
from scipy.special import roots_legendre

from .cheb_gram import chebyshev_gram_matrix, chebyshev_moment_vector
from .linear_tt_convolution import LinearTTConvolution, _moment_values


def _coordinates(domain, modes, scales, order):
    transforms, moments, derivatives, bases, nodes, quadrature = [], [], [], [], [], []
    t, w = roots_legendre(order)
    for box, scale in zip(domain, scales, strict=True):
        gram = np.asarray(chebyshev_gram_matrix(modes, box))
        derivative = np.zeros((modes, modes))
        for k in range(2, modes):
            v = np.polynomial.chebyshev.chebder(np.eye(modes)[k], m=2)*(2/np.ptp(box))**2
            derivative[k, :len(v)] = v
        inv = np.linalg.inv(np.linalg.cholesky(gram)).T
        d = (inv.T@(derivative@gram@derivative.T)@inv)*scale**4
        ev, vectors = np.linalg.eigh((d+d.T)/2)
        transform = inv@vectors
        transforms.append(jnp.asarray(transform))
        moments.append(jnp.asarray(np.asarray(chebyshev_moment_vector(modes, box))@transform))
        derivatives.append(jnp.asarray(np.maximum(ev, 0)))
        bases.append(jnp.asarray(np.polynomial.chebyshev.chebvander(t, modes-1)@transform))
        nodes.append(box[0]+np.ptp(box)*(t+1)/2)
        quadrature.append(w*np.ptp(box)/2)
    return tuple(transforms), tuple(moments), tuple(derivatives), bases, nodes, quadrature


@dataclass(frozen=True)
class SignedTTLeastSquares4D:
    domain: np.ndarray
    transforms: tuple
    moments: tuple
    derivatives: tuple
    left_grams: jax.Array
    right_grams: jax.Array
    overlap_weights: jax.Array
    quadrature_order: int

    @classmethod
    def from_noise(cls, domain, modes, covariances, weights, *, order=65):
        domain, covs, weights = np.asarray(domain), np.asarray(covariances), np.asarray(weights)
        if domain.shape != (4, 2) or covs.shape != (len(weights), 4, 4):
            raise ValueError('requires four dimensions and Gaussian-mixture noise')
        if not np.isfinite(domain).all() or np.any(domain[:, 0] >= domain[:, 1]) or modes < 2:
            raise ValueError('finite increasing domains and at least two modes required')
        if not np.isfinite(covs).all() or not np.allclose(covs, covs.transpose(0, 2, 1)):
            raise ValueError('noise covariances must be finite and symmetric')
        if not np.allclose(covs[:, :2, 2:], 0) or not np.allclose(covs[:, 2:, :2], 0):
            raise ValueError('noise must be block diagonal in pairs')
        if np.any(weights < 0) or not np.isclose(weights.sum(), 1):
            raise ValueError('noise weights must be nonnegative and sum to one')
        if np.any(np.linalg.eigvalsh(covs) <= 0) or order < modes:
            raise ValueError('positive definite noise and sufficient quadrature required')
        scales = np.sqrt(np.einsum('c,cii->i', weights, covs))
        transforms, moments, derivatives, bases, nodes, quadrature = _coordinates(domain, modes, scales, order)
        pair_grams, overlap_weights = [], []
        for pair in (0, 2):
            points = jnp.asarray(np.stack(np.meshgrid(nodes[pair], nodes[pair+1], indexing='ij'), axis=-1).reshape(-1, 2))
            v = jnp.einsum('ai,bj->abij', bases[pair], bases[pair+1]).reshape(order**2, modes**2)
            qw = jnp.asarray(np.outer(quadrature[pair], quadrature[pair+1]).ravel())
            v = v*qw[:, None]

            @jax.jit
            def projected(cov, points=points, v=v):
                precision = jnp.linalg.inv(cov)
                squared = jnp.sum((points@precision)*points, axis=1)
                exponent = squared[:, None]+squared[None]-2*points@precision@points.T
                kernel = jnp.exp(-.5*jnp.maximum(exponent, 0))/(2*jnp.pi*jnp.sqrt(jnp.linalg.det(cov)))
                result = v.T@(kernel@v)
                return (result+result.T)/2

            grams = []
            for i in range(len(weights)):
                for j in range(i, len(weights)):
                    grams.append(projected(jnp.asarray(covs[i, pair:pair+2, pair:pair+2]+covs[j, pair:pair+2, pair:pair+2])))
                    if pair == 0:
                        overlap_weights.append(weights[i]*weights[j]*(1 if i == j else 2))
            pair_grams.append(jnp.stack(grams))
        return cls(domain, tuple(transforms), tuple(moments), tuple(derivatives),
                   *pair_grams, jnp.asarray(overlap_weights), order)

    @classmethod
    def without_noise(cls, domain, modes, *, length_scales=None):
        """Direct density least squares with exact physical L2 mass matrices.

        Default curvature scales are 5% of each box width, independent of the
        unknown density. They set units; validation still chooses the strength.
        """
        domain = np.asarray(domain, dtype=float)
        if domain.shape != (4, 2) or not np.isfinite(domain).all() or np.any(np.diff(domain) <= 0) or modes < 2:
            raise ValueError('requires a finite four-dimensional box and at least two modes')
        scales = .05*np.ptp(domain, axis=1) if length_scales is None else np.asarray(length_scales)
        if scales.shape != (4,) or not np.isfinite(scales).all() or np.any(scales <= 0):
            raise ValueError('four positive finite curvature scales required')
        transforms, moments, derivatives, _, _, _ = _coordinates(domain, modes, scales, modes)
        identity = jnp.eye(modes*modes)[None]
        return cls(domain, transforms, moments, derivatives, identity, identity, jnp.ones(1), modes)

    def empirical_term_without_noise(self, observations, batch=512):
        """Sample-average Chebyshev evaluation functional, without density fitting."""
        observations = np.asarray(observations)
        if observations.ndim != 2 or observations.shape[1] != 4 or not len(observations):
            raise ValueError('nonempty four-dimensional observations required')
        if not np.all((observations >= self.domain[:, 0]) & (observations <= self.domain[:, 1])):
            raise ValueError('noise-free observations must lie inside the domain')
        @jax.jit
        def contribution(features, mask):
            left = jnp.einsum('bi,bj->bij', features[0], features[1]).reshape(batch, -1)
            right = jnp.einsum('bi,bj->bij', features[2], features[3]).reshape(batch, -1)
            return (left*mask[:, None]).T@right
        total = jnp.zeros((self.modes**2, self.modes**2))
        for start in range(0, len(observations), batch):
            points = observations[start:start+batch]
            count = len(points)
            t = 2*(points-self.domain[:, 0])/np.ptp(self.domain, axis=1)-1
            features = tuple(jnp.asarray(np.pad(np.polynomial.chebyshev.chebvander(t[:, k], self.modes-1),
                                  ((0, batch-count), (0, 0))))@self.transforms[k] for k in range(4))
            total += contribution(features, jnp.asarray(np.arange(batch) < count))
        return total/len(observations)

    @property
    def modes(self):
        return self.transforms[0].shape[0]

    @property
    def pair_moments(self):
        return tuple(jnp.outer(self.moments[k], self.moments[k+1]).ravel() for k in (0, 2))

    @property
    def pair_derivatives(self):
        return tuple((self.derivatives[k][:, None]+self.derivatives[k+1][None]).ravel() for k in (0, 2))

    def initial_factors(self, rank, seed=0, perturbation=.001):
        """Uniform density plus small random directions; no observations."""
        if rank < 1 or rank > self.modes**2:
            raise ValueError('invalid middle rank')
        rng = np.random.default_rng(seed)
        ml, mr = map(np.asarray, self.pair_moments)
        left = rng.normal(size=(len(ml), rank))
        right = rng.normal(size=(rank, len(mr)))
        dl, dr = map(np.asarray, self.pair_derivatives)
        left /= np.sqrt(1+dl[:, None])
        right /= np.sqrt(1+dr[None])
        left[:, 0] = ml/(ml@ml)
        right[0] = mr/(mr@mr)
        left[:, 1:] *= perturbation/np.sqrt(rank)
        right[1:] *= perturbation/np.sqrt(rank)
        left /= (ml@left@right@mr)
        return jnp.asarray(left), jnp.asarray(right)

    def model(self, left, right):
        """An ordinary four-core Chebyshev TT with ranks (modes, rank, modes)."""
        n, r = self.modes, left.shape[1]
        cores = (
            self.transforms[0].T[None],
            jnp.einsum('ijr,kj->irk', left.reshape(n, n, r), self.transforms[1]),
            jnp.einsum('rij,ki->rjk', right.reshape(r, n, n), self.transforms[2]),
            self.transforms[3].T[:, None],
        )
        return ChebTT(cores=tuple(ChebyshevCoeffCore(c, tuple(box)) for c, box in zip(cores, self.domain, strict=True)),
                     domain=jnp.asarray(self.domain))

    def empirical_term(self, operator: LinearTTConvolution, observations, batch=512):
        """Exact sample average of the linear forward functional, in TT coordinates.

        This is an objective coefficient, not an independently fitted density.
        It uses the prescribed polynomial noise kernel and every observation.
        """
        prepared, n = operator.prepared, self.modes

        @jax.jit
        def contribution(y, mask):
            raw = []
            for k, (moment, kernel) in enumerate(zip(prepared.moment_cores, prepared.kernel.cores, strict=True)):
                v = _moment_values(moment, y[:, k]).reshape(len(y), *kernel.shape, 2*n-1)[..., :n]
                raw.append(v@self.transforms[k])
            a = jnp.einsum('bai,bacj->bcij', raw[0][:, 0], raw[1]).reshape(len(y), -1, n*n)
            b = jnp.einsum('bcai,baj->bcij', raw[2], raw[3][:, :, 0]).reshape(len(y), -1, n*n)
            return jnp.einsum('bci,bcj,b->ij', a, b, mask)

        total = jnp.zeros((n*n, n*n))
        for start in range(0, len(observations), batch):
            y = np.asarray(observations[start:start+batch])
            count = len(y)
            y = np.pad(y, ((0, batch-count), (0, 0)))
            total += contribution(jnp.asarray(y), jnp.asarray(np.arange(batch) < count))
        return total/len(observations)

    def terms(self, left, right, empirical):
        lp = jnp.einsum('ir,cij,js->crs', left, self.left_grams, left)
        rp = jnp.einsum('ri,cij,sj->crs', right, self.right_grams, right)
        norm = jnp.einsum('c,crs,crs->', self.overlap_weights, lp, rp)
        cross = jnp.sum(left*(empirical@right.T))
        dl, dr = self.pair_derivatives
        roughness = jnp.sum((left.T@(dl[:, None]*left))*(right@right.T))
        roughness += jnp.sum((left.T@left)*((right*dr[None])@right.T))
        return norm-2*cross, roughness

    def penalty_bases(self, order):
        """Weighted tensor quadrature features for a squared negative-part penalty."""
        if order < self.modes:
            raise ValueError('penalty quadrature must resolve the polynomial')
        t, w = roots_legendre(order)
        bases = []
        for transform, box in zip(self.transforms, self.domain, strict=True):
            v = np.polynomial.chebyshev.chebvander(t, self.modes-1)@np.asarray(transform)
            bases.append(jnp.asarray(v*np.sqrt(w*np.ptp(box)/2)[:, None]))
        return tuple(jnp.einsum('ai,bj->abij', bases[k], bases[k+1]).reshape(order**2, self.modes**2)
                     for k in (0, 2))

    def make_half_solver(self, *, strength, ridge=1e-10, tolerance=1e-7, maxiter=1000, reverse=False,
                         negative_strength=0., penalty_bases=None):
        """Constrained convex core solve, with nonvanishing gradients at f<0.

        Each half updates one middle core after a QR gauge change. Two CG
        solves enforce the mass constraint exactly via its Schur complement.
        A block-Jacobi preconditioner includes the physical curvature penalty.
        """
        a, b = (self.right_grams, self.left_grams) if reverse else (self.left_grams, self.right_grams)
        if strength < 0 or negative_strength < 0 or ridge <= 0 or tolerance <= 0 or maxiter < 1:
            raise ValueError('nonnegative penalties and positive solver controls required')
        dl, dr = self.pair_derivatives[::-1] if reverse else self.pair_derivatives
        ml, mr = self.pair_moments[::-1] if reverse else self.pair_moments
        weights = self.overlap_weights
        if negative_strength:
            if penalty_bases is None:
                raise ValueError('negative-part penalty requires quadrature features')
            vl, vr = penalty_bases[::-1] if reverse else penalty_bases
        else:
            vl, vr = None, None
        matrices = (a, b, dl, dr, ml, mr, weights, vl, vr)

        @jax.jit
        def solve(left, right, empirical, matrices):
            # Large quadrature/overlap arrays are runtime inputs, not embedded
            # compiler constants (which can exceed the 2-GiB cache limit).
            a, b, dl, dr, ml, mr, weights, vl, vr = matrices
            q, triangular = jnp.linalg.qr(right.T, mode='reduced')
            left, right = left@triangular.T, q.T
            small = jnp.einsum('ri,cij,sj->crs', right, b, right)
            identity = jnp.eye(right.shape[0])
            derivative = (right*dr[None])@right.T
            if negative_strength:
                samples_right = right@vr.T
                negative = (vl@left)@samples_right < 0

            def base_action(x):
                result = sum(weights[k]*(a[k]@x@small[k]) for k in range(len(weights)))
                return result+strength*(dl[:, None]*x+x@derivative)+ridge*x

            def action(x):
                result = base_action(x)
                if negative_strength:
                    delta = (vl@x)@samples_right
                    result += negative_strength*(vl.T@(negative*delta)@samples_right.T)
                return result

            diagonal = jnp.einsum('c,ci,crs->irs', weights, jnp.diagonal(a, axis1=1, axis2=2), small)
            diagonal += strength*(dl[:, None, None]*identity[None]+derivative[None])+ridge*identity[None]
            if negative_strength:
                # Exact diagonal blocks of the active negative-part Hessian.
                # Adding eta*I even when every sampled value is positive
                # severely slows CG at strong penalties.
                products = jnp.einsum('rb,sb->brs', samples_right, samples_right).reshape(samples_right.shape[1], -1)
                active_products = negative.astype(left.dtype)@products
                blocks = ((vl*vl).T@active_products).reshape(diagonal.shape)
                diagonal += negative_strength*blocks
            inverse = jnp.linalg.inv(diagonal)
            precondition = lambda x: jnp.einsum('irs,is->ir', inverse, x)
            rhs = empirical@right.T
            mass = ml[:, None]*(right@mr)[None]
            unconstrained, _ = jax.scipy.sparse.linalg.cg(action, rhs, x0=left, tol=tolerance, maxiter=maxiter, M=precondition)
            direction, _ = jax.scipy.sparse.linalg.cg(action, mass, tol=tolerance, maxiter=maxiter, M=precondition)
            multiplier = (1-jnp.sum(mass*unconstrained))/jnp.sum(mass*direction)
            result = unconstrained+multiplier*direction
            residual = jnp.linalg.norm(action(result)-rhs-multiplier*mass)/jnp.maximum(jnp.linalg.norm(rhs), 1e-30)
            if negative_strength:
                # One semismooth Newton core step; a mass-preserving line
                # search evaluates the ACTUAL negative-part penalty, so a
                # changing active set cannot silently increase the objective.
                def objective(x):
                    weighted = (vl@x)@samples_right
                    return jnp.sum(x*base_action(x))-2*jnp.sum(x*rhs)+negative_strength*jnp.sum(jnp.minimum(weighted, 0)**2)
                initial_value = objective(left)
                delta = result-left
                def backtrack(state):
                    k, chosen, accepted = state
                    candidate = left+(0.5**k)*delta
                    valid = objective(candidate) <= initial_value+1e-14
                    return k+1, jnp.where(valid, candidate, chosen), accepted | valid
                _, result, _ = jax.lax.while_loop(lambda state: (state[0] < 14) & ~state[2],
                        backtrack, (jnp.array(0), left, jnp.array(False)))
            return result, right, residual
        return lambda left, right, empirical: solve(left, right, empirical, matrices)
