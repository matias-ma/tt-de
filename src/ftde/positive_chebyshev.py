"""Direct density TTs with certified positive Chebyshev polynomial cores.

The density is the TT contraction itself. Bernstein coefficients on small
subintervals are used ONLY as linear positivity inequalities on Chebyshev
coefficients; neither the model basis nor the convolution basis changes.
No clipping, squaring, exponential link, or SVD rounding is used.
"""
from dataclasses import dataclass
from functools import lru_cache
from itertools import pairwise

import jax.numpy as jnp
import numpy as np
from chebttge.chebtt import ChebTT
from chebttge.cores import ChebyshevCoeffCore
from scipy.fft import dct
from scipy.special import roots_legendre

from .cheb_gram import chebyshev_moment_vector


def _elevate(b):
    n = len(b)
    result = np.zeros(n+1)
    result[:-1] += (1-np.arange(n)/n)*b
    result[1:] += np.arange(1, n+1)/n*b
    return result


@lru_cache(maxsize=32)
def positivity_certificate(modes, subdivisions=16):
    """A @ c are Bernstein bounds for a Chebyshev polynomial on [-1, 1].

    If every component is positive, the polynomial is positive everywhere,
    including endpoints. Subdivision makes this sufficient condition less
    restrictive. Recurrence in the local Bernstein basis avoids conversion
    through ill-conditioned global power coefficients at high degrees.
    """
    if modes < 1 or subdivisions < 1:
        raise ValueError('modes and subdivisions must be positive')
    blocks = []
    edges = np.linspace(-1., 1., subdivisions+1)
    for lo, hi in pairwise(edges):
        polynomials = [np.ones(1)]
        if modes > 1:
            polynomials.append(np.array([lo, hi]))
        for n in range(1, modes-1):
            b = polynomials[-1]
            product = np.zeros(n+2)
            product[:-1] += lo*(1-np.arange(n+1)/(n+1))*b
            product[1:] += hi*np.arange(1, n+2)/(n+1)*b
            polynomials.append(2*product-_elevate(_elevate(polynomials[-2])))
        elevated = []
        for b in polynomials:
            while len(b) < modes:
                b = _elevate(b)
            elevated.append(b)
        blocks.append(np.stack(elevated, axis=1))
    result = np.concatenate(blocks)
    result.setflags(write=False)
    return result


@dataclass(frozen=True)
class ChebyshevCoreCoordinates:
    """Nodal preconditioning of Chebyshev coefficients, not a new basis.

    u stores core values at Gauss nodes, divided by 1/(right_rank*width).
    Row normalization is the linear equality sum_{right,node} w[node]*u
    = right_rank, with w summing to one. Thus integral G_k is stochastic.
    """
    modes: int
    subdivisions: int = 16

    def __post_init__(self):
        nodes, weights = roots_legendre(self.modes)
        inverse = np.linalg.inv(np.polynomial.chebyshev.chebvander(nodes, self.modes-1))
        certificate = positivity_certificate(self.modes, self.subdivisions)
        object.__setattr__(self, 'nodes', nodes)
        object.__setattr__(self, 'weights', weights/2)
        object.__setattr__(self, 'inverse', inverse)
        object.__setattr__(self, 'certificate', certificate)
        object.__setattr__(self, 'bounds', certificate@inverse)
        # Integrated squared physical second derivative, divided by width^4
        # later. This is a fixed convex quadratic in an active core.
        derivative = np.polynomial.chebyshev.chebder(inverse, m=2, axis=0)
        values = np.polynomial.chebyshev.chebvander(nodes, max(self.modes-3, 0))@derivative if self.modes > 2 else np.zeros_like(inverse)
        object.__setattr__(self, 'curvature_factor', np.sqrt(weights[:, None]/2)*values*4)
        object.__setattr__(self, 'curvature', values.T@(weights[:, None]/2*values)*16)

    def coefficients(self, u, domain):
        return u@jnp.asarray(self.inverse.T)/(u.shape[1]*(domain[1]-domain[0]))

    def parameters(self, c, domain):
        vand = np.polynomial.chebyshev.chebvander(self.nodes, self.modes-1)
        return jnp.asarray(np.asarray(c)@vand.T*c.shape[1]*(domain[1]-domain[0]))

    def project(self, u):
        """Orthogonal projection onto the equality-constraint tangent."""
        w = jnp.asarray(self.weights)
        return u-jnp.sum(u*w, axis=(1, 2), keepdims=True)*w/(u.shape[1]*jnp.sum(w*w))


@dataclass(frozen=True)
class PositiveChebyshevTT:
    tt: ChebTT
    subdivisions: int = 16

    def audit(self):
        minima, row_errors = [], []
        for c in self.tt.cores:
            coeff = np.asarray(c.coeffs)
            bounds = coeff@positivity_certificate(coeff.shape[-1], self.subdivisions).T
            minima.append(float(bounds.min()))
            masses = coeff@np.asarray(chebyshev_moment_vector(coeff.shape[-1], c.domain))
            row_errors.append(float(np.max(np.abs(masses.sum(axis=1)-1))))
        return {'minimum_core_certificates': minima, 'maximum_row_mass_errors': row_errors,
                'integral': float(self.integral())}

    def validate(self):
        report = self.audit()
        if min(report['minimum_core_certificates']) <= 0:
            raise ValueError('Chebyshev core does not have a strictly positive certificate')
        if max(report['maximum_row_mass_errors']) > 1e-9:
            raise ValueError('Core rows must integrate to one')
        return report

    def integral(self):
        state = jnp.ones(1)
        for core in self.tt.cores:
            state = state@(core.coeffs@chebyshev_moment_vector(core.coeffs.shape[-1], core.domain))
        return state[0]

    def log_density(self, points):
        x = jnp.atleast_2d(points)
        if x.ndim != 2 or x.shape[1] != len(self.tt.cores):
            raise ValueError('points must have shape (dimension,) or (batch, dimension)')
        domain = jnp.asarray(self.tt.domain)
        state, logscale = jnp.ones((len(x), 1)), jnp.zeros(len(x))
        for k, c in enumerate(self.tt.cores):
            values = c(jnp.clip(x[:, k], *c.domain))
            state = jnp.einsum('na,nab->nb', state, values)
            scale = jnp.max(state, axis=1)
            state = state/scale[:, None]
            logscale += jnp.log(scale)
        inside = jnp.all((x >= domain[:, 0]) & (x <= domain[:, 1]), axis=1)
        result = jnp.where(inside, logscale+jnp.log(state[:, 0]), -jnp.inf)
        return result[0] if jnp.ndim(points) == 1 else result

    def density(self, points):
        return jnp.exp(self.log_density(points))

    def pair_grid(self, i, j, xs, ys):
        if not 0 <= i < j < len(self.tt.cores):
            raise ValueError('Require 0 <= i < j < dimension')
        xx, yy = np.meshgrid(xs, ys)
        coordinates = {i: xx.ravel(), j: yy.ravel()}
        state = np.ones((xx.size, 1))
        inside = np.ones(xx.size, dtype=bool)
        for k, core in enumerate(self.tt.cores):
            if k in coordinates:
                x = coordinates[k]
                inside &= (x >= core.domain[0]) & (x <= core.domain[1])
                values = np.asarray(core(jnp.asarray(np.clip(x, *core.domain))))
                state = np.einsum('na,nab->nb', state, values)
            else:
                state = state@(np.asarray(core.coeffs)@np.asarray(chebyshev_moment_vector(core.coeffs.shape[-1], core.domain)))
        return np.where(inside, state[:, 0], 0.).reshape(xx.shape)

    def save(self, path):
        self.validate()
        np.savez(path, kind='positive_chebyshev_density_tt_v1', subdivisions=self.subdivisions,
                 domain=np.asarray(self.tt.domain), **{f'core_{k}': np.asarray(c.coeffs) for k, c in enumerate(self.tt.cores)})

    @classmethod
    def load(cls, path):
        with np.load(path) as data:
            if str(data['kind']) != 'positive_chebyshev_density_tt_v1':
                raise ValueError('This is not a direct positive Chebyshev density TT')
            domains = data['domain']
            cores = tuple(ChebyshevCoeffCore(jnp.asarray(data[f'core_{k}']), tuple(box)) for k, box in enumerate(domains))
            result = cls(ChebTT(cores=cores, domain=jnp.asarray(domains), ranks=tuple(c.shape[1] for c in cores[:-1])), int(data['subdivisions']))
        result.validate()
        return result


def initialize_positive_tt(train, domain, *, rank=16, modes=49, subdivisions=16, seed=0, width=.4):
    """Data-only start from representative observations, then free TT cores.

    k-means supplies locations only (no signal-mixture likelihood/EM or
    deconvolution is fitted). All transition polynomials are independently
    trainable. A small positive transition floor prevents disconnected states.
    """
    from scipy.cluster.vq import kmeans2
    if rank < 1 or modes < 3 or width <= 0 or len(domain) < 2:
        raise ValueError('Require rank >= 1, modes >= 3, width > 0, dimension >= 2')
    rng = np.random.default_rng(seed)
    subset = np.asarray(train)[rng.choice(len(train), min(20000, len(train)), replace=False)]
    centers, labels = kmeans2(subset, rank, iter=20, minit='++', seed=seed)
    frequencies = np.bincount(labels, minlength=rank)+1.
    frequencies /= frequencies.sum()
    cert = positivity_certificate(modes, subdivisions)
    cores = []
    for k, (a, b) in enumerate(domain):
        t = np.cos(np.pi*(np.arange(modes)+.5)/modes)
        x = a+(b-a)*(t+1)/2
        samples = np.exp(-.5*((x[None]-centers[:, k, None])/width)**2)
        coeff = dct(samples, type=2, axis=-1)/modes
        coeff[:, 0] *= .5
        coeff[:, 0] += np.maximum(0., 1e-5-(coeff@cert.T).min(axis=1))
        masses = coeff@np.asarray(chebyshev_moment_vector(modes, (a, b)))
        coeff /= masses[:, None]
        if k == 0:
            c = (frequencies[:, None]*coeff)[None]
        elif k == len(domain)-1:
            c = coeff[:, None, :]
        else:
            transitions = .02*np.ones((rank, rank))/rank+.98*np.eye(rank)
            c = transitions[:, :, None]*coeff[None, :, :]
        cores.append(ChebyshevCoeffCore(jnp.asarray(c), (float(a), float(b))))
    model = PositiveChebyshevTT(ChebTT(cores=tuple(cores), domain=jnp.asarray(domain), ranks=(rank,)*(len(domain)-1)), subdivisions)
    model.validate()
    return model


def expand_positive_modes(model, modes):
    """Exact polynomial-degree continuation; no SVD or function change."""
    cores = []
    for core in model.tt.cores:
        old = core.coeffs.shape[-1]
        if modes < old:
            raise ValueError('Mode continuation cannot truncate a positive polynomial')
        coeff = jnp.pad(core.coeffs, ((0, 0), (0, 0), (0, modes-old)))
        cores.append(ChebyshevCoeffCore(coeff, core.domain))
    result = PositiveChebyshevTT(ChebTT(cores=tuple(cores), domain=model.tt.domain, ranks=model.tt.ranks), model.subdivisions)
    result.validate()
    return result
