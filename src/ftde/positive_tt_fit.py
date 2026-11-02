"""Alternating convex core solves for a positive Chebyshev density TT."""
import jax
import jax.numpy as jnp
import numpy as np
from jax.scipy.linalg import solve_triangular
from jax.scipy.sparse.linalg import cg

from .linear_tt_convolution import LinearTTConvolution, _moment_values
from .positive_chebyshev import ChebyshevCoreCoordinates, PositiveChebyshevTT
from .tt_density import rebuild_tt


class PositiveTTForward:
    """Reuse the unsquared Chebyshev/noise-TT convolution moments."""
    def __init__(self, model, observation_domain=None, kernel=None):
        self.template = model.tt
        self.coordinates = tuple(ChebyshevCoreCoordinates(c.coeffs.shape[-1], model.subdivisions) for c in model.tt.cores)
        self.operator = None if kernel is None else LinearTTConvolution.from_tt(model.tt, kernel, observation_domain)
        self.features = []
        for k, (c, coord) in enumerate(zip(model.tt.cores, self.coordinates, strict=True)):
            def feature(x, k=k, c=c, coord=coord):
                if self.operator is None:
                    # Shared recurrence also used by the prepared convolution.
                    from chebttge.cores import ChebyshevCoeffCore
                    identity = ChebyshevCoeffCore(jnp.eye(coord.modes)[None], c.domain)
                    h = _moment_values(identity, x).reshape(len(x), 1, 1, coord.modes)
                else:
                    prepared = self.operator.prepared
                    h = _moment_values(prepared.moment_cores[k], x).reshape(len(x), *prepared.kernel.cores[k].shape, 2*coord.modes-1)[..., :coord.modes]
                return h@jnp.asarray(coord.inverse)/(c.shape[1]*(c.domain[1]-c.domain[0]))
            self.features.append(jax.jit(feature))

    def parameters(self, model):
        return tuple(coord.parameters(c.coeffs, c.domain) for coord, c in zip(self.coordinates, model.tt.cores, strict=True))

    def model(self, params):
        coeff = tuple(coord.coefficients(p, c.domain) for coord, p, c in zip(self.coordinates, params, self.template.cores, strict=True))
        return PositiveChebyshevTT(rebuild_tt(self.template, coeff), self.coordinates[0].subdivisions)

    def log_density(self, params, points):
        model = self.model(params)
        return model.log_density(points) if self.operator is None else self.operator.log_density(model.tt, points)


@jax.jit
def transfer_left(state, u, h):
    result = jnp.einsum('nla,lrm,nacm->nrc', state, u, h, optimize='optimal')
    return result/jnp.maximum(jnp.max(jnp.abs(result), axis=(1, 2), keepdims=True), 1e-300)


@jax.jit
def transfer_right(state, u, h):
    result = jnp.einsum('nrc,lrm,nacm->nla', state, u, h, optimize='optimal')
    return result/jnp.maximum(jnp.max(jnp.abs(result), axis=(1, 2), keepdims=True), 1e-300)


@jax.jit
def core_design(left, right, h, u):
    design = jnp.einsum('nla,nrc,nacm->nlrm', left, right, h, optimize='optimal')
    q = jnp.einsum('nlrm,lrm->n', design, u)
    return design/q[:, None, None, None]


def make_core_solver(coord, width, *, cg_iterations=60):
    """Matrix-free damped Newton-CG; linear equality/positivity constraints.

    Bernstein bounds give a fraction-to-boundary step. Armijo backtracking
    checks the actual objective. Every accepted step is feasible and lowers
    the block objective. The quadratic roughness is on the core polynomial,
    NOT the old amplitude-relative (nonconvex) Sobolev quotient.
    """
    bounds = jnp.asarray(coord.bounds)
    # A tiny interior margin in dimensionless core-value coordinates protects
    # the certificate during roundoff in nodal <-> Chebyshev conversion.
    margin = 1e-8
    roughness = jnp.asarray(coord.curvature/width**4)
    roughness_factor = jnp.asarray(coord.curvature_factor/width**2)
    w = jnp.asarray(coord.weights)

    @jax.jit
    def step(u, design, barrier, strength):
        shape = u.shape
        d = design.reshape(len(design), -1)
        n = len(d)
        edges = shape[0]*shape[1]
        certificate_count = edges*bounds.shape[0]

        def project(v):
            return coord.project(v.reshape(shape)).reshape(-1)

        def objective(v):
            p = d@v.ravel()
            z = v@bounds.T-margin
            return -jnp.mean(jnp.log(p))-barrier*jnp.mean(jnp.log(z))+strength*jnp.sum((v@roughness)*v)/edges

        q, z = d@u.ravel(), u@bounds.T-margin
        grad = (-d.T@(1/q)/n).reshape(shape)-barrier*(1/z)@bounds/certificate_count+2*strength*(u@roughness)/edges
        likelihood_diag = jnp.sum((d/q[:, None])**2, axis=0).reshape(shape)/n
        # QR whitening avoids forming/inverting a badly conditioned barrier
        # Gram matrix. Each small edge block gets its own stable factor.
        damping = 1e-7
        stacked = jnp.concatenate([
            jnp.sqrt(barrier/certificate_count)*bounds[None, None]/z[..., None],
            jnp.eye(shape[-1])*jnp.sqrt(likelihood_diag+damping)[..., None, :],
            jnp.broadcast_to(jnp.sqrt(2*strength/edges)*roughness_factor, (*shape[:2], *roughness_factor.shape))], axis=-2)
        r = jnp.linalg.qr(stacked, mode='r')

        def inverse(v, transpose=False):
            return solve_triangular(r, v[..., None], lower=False, trans='T' if transpose else 'N')[..., 0]

        iw = inverse(jnp.broadcast_to(w, shape), transpose=True)
        denom = jnp.sum(iw*iw, axis=(1, 2), keepdims=True)

        def white_project(v):
            x = v.reshape(shape)
            return (x-iw*jnp.sum(x*iw, axis=(1, 2), keepdims=True)/denom).ravel()

        g = white_project(inverse(grad, transpose=True))

        def hessian(v):
            x = inverse(white_project(v).reshape(shape))
            v = x.ravel()
            result = (d.T@((d@v)/q**2)/n).reshape(shape)
            result += barrier*((x@bounds.T)/z**2)@bounds/certificate_count
            result += 2*strength*(x@roughness)/edges+damping*x
            return white_project(inverse(result, transpose=True))

        direction, _ = cg(hessian, -g, tol=1e-3, maxiter=cg_iterations)
        descent = jnp.vdot(g, direction)
        direction = jnp.where(jnp.isfinite(descent) & (descent < 0), direction, -g)
        direction = project(inverse(white_project(direction).reshape(shape))).reshape(shape)
        dz, dq = direction@bounds.T, d@direction.ravel()
        maximum = jnp.minimum(jnp.min(jnp.where(dz < 0, -.9*z/dz, jnp.inf)),
                              jnp.min(jnp.where(dq < 0, -.9*q/dq, jnp.inf)))
        alpha = jnp.minimum(1., maximum)
        before = objective(u)
        slope = jnp.sum(grad*direction)

        def condition(state):
            alpha, count = state
            candidate = objective(u+alpha*direction)
            return ((~jnp.isfinite(candidate)) | (candidate > before+1e-4*alpha*slope)) & (count < 24)

        alpha, count = jax.lax.while_loop(condition, lambda state: (state[0]/2, state[1]+1), (alpha, 0))
        candidate = u+alpha*direction
        after = objective(candidate)
        accepted = jnp.isfinite(after) & (after <= before+1e-4*alpha*slope) & (jnp.min(candidate@bounds.T) > margin)
        result = jnp.where(accepted, candidate, u)
        return result, jnp.array([before, jnp.where(accepted, after, before), alpha, jnp.linalg.norm(g), count, accepted])
    return step


def sweep(params, features, solvers, *, barrier, strength, inner_steps=3, reverse=False):
    """One sweep uses a fixed observed-data batch; truth is never involved."""
    params = list(params)
    count = len(params)
    boundary = jnp.ones((len(features[0]), 1, 1))
    cached = [None]*(count+1)
    if reverse:
        cached[0] = boundary
        for k in range(count):
            cached[k+1] = transfer_left(cached[k], params[k], features[k])
    else:
        cached[count] = boundary
        for k in reversed(range(count)):
            cached[k] = transfer_right(cached[k+1], params[k], features[k])
    active, diagnostics = boundary, []
    for k in reversed(range(count)) if reverse else range(count):
        left, right = (cached[k], active) if reverse else (active, cached[k+1])
        design = core_design(left, right, features[k], params[k])
        if not bool(jnp.all(jnp.isfinite(design))):
            raise FloatingPointError('Nonpositive/invalid convolution at block initialization')
        for _ in range(inner_steps):
            params[k], info = solvers[k](params[k], design, barrier, strength)
            diagnostics.append(np.asarray(info).tolist())
        active = transfer_right(active, params[k], features[k]) if reverse else transfer_left(active, params[k], features[k])
    return tuple(params), diagnostics
