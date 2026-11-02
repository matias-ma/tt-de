"""Per-observation prescribed TT kernels, including compact Gaussian kernels.

Kernel fitting uses chebdGE, never a pilot for the unknown signal. The kernel
is zero outside its box. Observation-specific polynomial moments are prepared
once; training differentiates only the signal coefficients. Full covariance
enters the Gaussian target, including every off-diagonal entry.
"""
from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np
from scipy.special import roots_legendre, ndtr
from chebttge import ChebTTOptions, tt_qtensge, integrate
from chebttge.chebtt import ChebTT
from chebttge.cores import ChebyshevCoeffCore

from .cheb_gram import chebyshev_gram_matrix
from .gaussian_convolution import _transfer, _rescale
from .direct_training import NoisyDirectTrainingProblem


def gaussian_peak(x, precision):
    return jnp.exp(-.5*jnp.einsum('...i,ij,...j->...', x, precision, x))


def validate_covariances(covariance):
    c = np.asarray(covariance, dtype=np.float64)
    if c.ndim < 2 or c.shape[-1] != c.shape[-2] or not np.isfinite(c).all():
        raise ValueError('covariance must contain finite square matrices')
    sd = np.sqrt(np.maximum(c.diagonal(axis1=-2, axis2=-1), 0))
    if np.any(sd <= 0):
        raise ValueError('covariance must be positive definite')
    corr = c/sd[..., :, None]/sd[..., None, :]
    if not np.allclose(corr, corr.swapaxes(-1,-2), atol=1e-7, rtol=0):
        raise ValueError('covariance must be symmetric')
    # Symmetrize floating-point inversion noise only; never diagonalize or add jitter.
    c = (c+c.swapaxes(-1,-2))/2
    try:
        np.linalg.cholesky(c)
    except np.linalg.LinAlgError as e:
        raise ValueError('covariance must be positive definite') from e
    return c


def fit_gaussian_kernel(covariance, *, sigmas=8., tolerance=1e-12,
                        max_rank=96, seed=0, n_starts=256, gd_steps=0):
    """Fit on [-1,1]^d after axis scaling by sigmas*sqrt(diag(cov)).

    This is axis scaling, not whitening: all correlations remain in the
    fitted target. The Gaussian mass discarded outside the physical box
    is at most 2*d*Phi(-sigmas), independently of its correlations. Normalize
    the represented compact kernel and return construction diagnostics.
    The chebdGE stopping tolerance is NOT a global accuracy guarantee.
    """
    c = validate_covariances(covariance)
    if c.ndim != 2 or not np.isfinite(sigmas) or sigmas <= 0:
        raise ValueError('expected one covariance and positive finite sigmas')
    d = len(c); width = sigmas*np.sqrt(c.diagonal())
    precision = np.linalg.inv(c/width[:,None]/width[None,:])
    opts = ChebTTOptions(domain=jnp.array([[-1.,1.]]*d), tol=tolerance,
        cheb_tol=max(tolerance*.01, 2e-14), max_rank=max_rank,
        core_representation='coeff_tensor', n_starts=n_starts, gd_steps=gd_steps,
        pivot_strategy='sample_cplu', random_seed=seed, validation_samples=1024)
    fitted = tt_qtensge(gaussian_peak, opts, d=d, target_params=jnp.asarray(precision))
    box = np.stack([-width,width],axis=1)
    cores = [ChebyshevCoeffCore(core.coeffs,tuple(bounds))
             for core,bounds in zip(fitted.cores,box,strict=True)]
    physical = ChebTT(cores=cores,domain=box)
    mass = float(integrate(physical))
    if not np.isfinite(mass) or mass <= 0:
        raise FloatingPointError('nonpositive fitted kernel integral')
    cores[0] = ChebyshevCoeffCore(cores[0].coeffs/mass,cores[0].domain)
    kernel = ChebTT(cores=cores,domain=box)
    return kernel, fitted, dict(ranks=list(fitted.ranks),
        modes=[c.coeffs.shape[-1] for c in fitted.cores],
        tail_mass_bound=float(2*d*ndtr(-sigmas)),
        relative_mass_error=float(mass/((2*np.pi)**(d/2)*np.sqrt(np.linalg.det(c)))-1))


def kernel_moments(kernel, point, domain, sizes):
    """h_k[a,b,l] = integral U_k(y_k-x) T_l(x) dx on the box overlap.

    Degree-derived Gauss quadrature is exact for the supplied polynomials.
    Moments through degree 2*n-2 suffice via T_i*T_j=(T_{i+j}+T_{|i-j|})/2.
    """
    moments=[]
    for core,y,(a,b),n in zip(kernel.cores,point,domain,sizes,strict=True):
        lo,hi=max(a,y-core.domain[1]),min(b,y-core.domain[0])
        q=n+(core.coeffs.shape[-1]-1)//2
        t,w=roots_legendre(q)
        if hi <= lo:
            raise ValueError('observation is outside the compact convolution support')
        x=lo+(hi-lo)*(t+1)/2
        v=np.polynomial.chebyshev.chebvander((2*x-a-b)/(b-a),2*n-2)
        z=(2*(y-x)-sum(core.domain))/(core.domain[1]-core.domain[0])
        u=np.polynomial.chebyshev.chebvander(z,core.coeffs.shape[-1]-1) @ np.asarray(core.coeffs).reshape(-1,core.coeffs.shape[-1]).T
        h=(u.T@(w[:,None]*(hi-lo)/2*v)).reshape(*core.shape,2*n-1)
        moments.append(h)
    return tuple(moments)


def round_moments(cores, tolerance=1e-12):
    """Standard right-orthogonalization then TT-SVD of a moment tensor.

    The relative Frobenius error bound is tolerance (in moment space),
    separate from the kernel fit error. No rank cap silently truncates it.
    """
    cs=[np.array(c,copy=True) for c in cores]
    for k in range(len(cs)-1,0,-1):
        a,b,n=cs[k].shape
        q,r=np.linalg.qr(cs[k].reshape(a,b*n).T,mode='reduced')
        cs[k]=q.T.reshape(-1,b,n)
        cs[k-1]=np.einsum('abn,bc->acn',cs[k-1],r.T)
    norm=np.linalg.norm(cs[0]); delta=tolerance*norm/np.sqrt(max(1,len(cs)-1))
    for k in range(len(cs)-1):
        a,b,n=cs[k].shape
        u,s,v=np.linalg.svd(cs[k].transpose(0,2,1).reshape(a*n,b),full_matrices=False)
        tails=np.r_[np.cumsum(s[::-1]**2)[::-1],0.]
        keep=max(1,int(np.flatnonzero(tails <= delta**2)[0]))
        cs[k]=u[:,:keep].reshape(a,n,keep).transpose(0,2,1)
        cs[k+1]=np.einsum('ab,bcn->acn',s[:keep,None]*v[:keep],cs[k+1])
    return tuple(cs)


@dataclass(frozen=True)
class ParticleTTConvolution:
    """Contraction against per-observation TT polynomial moments."""
    domains: tuple
    sizes: tuple
    shapes: tuple
    grams: tuple

    @classmethod
    def from_tt(cls, template, shapes):
        domains=tuple(c.domain for c in template.cores)
        sizes=tuple(c.coeffs.shape[-1] for c in template.cores)
        shapes=tuple(tuple(s) for s in shapes)
        if len(shapes)!=len(sizes) or any(s[2]!=2*n-1 for s,n in zip(shapes,sizes)):
            raise ValueError('moments must have degree 2*n-2 for every signal core')
        if shapes[0][0]!=1 or shapes[-1][1]!=1 or any(a[1]!=b[0] for a,b in zip(shapes,shapes[1:])):
            raise ValueError('moment TT ranks do not match')
        return cls(domains,sizes,shapes,tuple(chebyshev_gram_matrix(n,box) for n,box in zip(sizes,domains)))

    def log_density(self,g,features):
        domains=tuple(c.domain for c in g.cores)
        concrete=not any(isinstance(v,jax.core.Tracer) for v in jax.tree.leaves(domains))
        if (concrete and domains!=self.domains) or tuple(c.coeffs.shape[-1] for c in g.cores)!=self.sizes:
            raise ValueError('operator and TT must have the same domains and polynomial sizes')
        if len(features)!=len(self.sizes):
            raise ValueError('one moment array per coordinate is required')
        batch=len(features[0]);state=jnp.ones((batch,1,1,1));norm=jnp.ones((1,1,1))
        lv=jnp.zeros(batch);ln=jnp.zeros(1)
        for k,core in enumerate(g.cores):
            h=jnp.asarray(features[k]).reshape(batch,*self.shapes[k])
            c=core.coeffs/jax.lax.stop_gradient(jnp.maximum(jnp.max(jnp.abs(core.coeffs)),1e-300))
            i=jnp.arange(self.sizes[k]);m=.5*(h[...,i[:,None]+i[None,:]]+h[...,jnp.abs(i[:,None]-i[None,:])])
            left=jnp.einsum('bpqa,pki->bqkai',state,c)
            left=jnp.einsum('bqkai,bacij->bqkcj',left,m)
            state=jnp.einsum('bqkcj,qlj->bklc',left,c)
            scale=jax.lax.stop_gradient(jnp.maximum(jnp.max(jnp.abs(state),axis=(1,2,3)),jnp.finfo(state.dtype).tiny))
            state=state/scale[:,None,None,None];lv=lv+jnp.log(scale)
            norm=_transfer(norm,c,self.grams[k][None]);norm,s=_rescale(norm);ln=ln+s
        value=lv+jnp.log(state[:,0,0,0])-ln-jnp.log(norm[0,0,0])
        if not concrete:value=jnp.where(jnp.all(jnp.asarray(domains)==jnp.asarray(self.domains)),value,jnp.nan)
        return value


class ParticleTTProblem(NoisyDirectTrainingProblem):
    """Train/validation inputs are row indices into a fixed moment cache."""
    def __init__(self,initial,moments,*,stream=False,**kwargs):
        self.moments=tuple(m if isinstance(m,RaggedMoments) else np.asarray(m) for m in moments)
        self.stream=stream
        if any(len(m)!=len(self.moments[0]) for m in self.moments):
            raise ValueError('moment arrays must have matching row counts')
        super().__init__(initial,ParticleTTConvolution.from_tt(initial,[m.shape[1:] for m in self.moments]),**kwargs)

    def features(self,observations):
        ids=np.asarray(observations)
        if ids.ndim!=1 or ids.dtype.kind not in 'iu' or np.any(ids<0) or np.any(ids>=len(self.moments[0])):
            raise ValueError('observations must be valid integer moment row indices')
        if self.stream:
            return (jnp.asarray(ids[:,None]),)
        return tuple(jnp.asarray(m[ids].reshape(len(ids),-1)) for m in self.moments)

    def _logs(self,params,features):
        if self.stream:
            ids=features[0].reshape(-1)
            shapes=tuple(jax.ShapeDtypeStruct((len(ids),int(np.prod(m.shape[1:]))),jnp.float64)
                         for m in self.moments)
            def read(rows):
                values=tuple(m[np.asarray(rows)].reshape(len(rows),-1) for m in self.moments)
                if any(not np.isfinite(v).all() for v in values):
                    raise FloatingPointError('nonfinite cached moments')
                return values
            # Moments depend only on integer particle IDs, never on the
            # trainable coefficients; no derivative crosses this host read.
            features=jax.pure_callback(read,shapes,ids)
        return self.forward.log_density(self.model(params),features)


class RaggedMoments:
    """Read variable-rank particle cores from contiguous memory-mapped storage.

    Padding happens only for the requested minibatch, so one unusually
    correlated particle does not inflate every on-disk particle record.
    """
    def __init__(self,values,shapes):
        self.values=values
        self.row_shapes=np.asarray(shapes,dtype=int)
        if self.row_shapes.ndim!=2 or self.row_shapes.shape[1]!=3 or np.any(self.row_shapes<=0):
            raise ValueError('ragged moments require positive three-dimensional core shapes')
        sizes=np.prod(self.row_shapes,axis=1)
        self.offsets=np.r_[0,np.cumsum(sizes)]
        if len(values)!=self.offsets[-1]:raise ValueError('ragged moment storage length mismatch')
        self.shape=(len(shapes),*self.row_shapes.max(axis=0))

    def __len__(self):return self.shape[0]

    def __getitem__(self,indices):
        ids=np.asarray(indices)
        if ids.ndim!=1 or ids.dtype.kind not in 'iu' or np.any(ids<0) or np.any(ids>=len(self)):
            raise ValueError('invalid ragged moment row indices')
        result=np.zeros((len(ids),*self.shape[1:]),dtype=np.float64)
        for j,i in enumerate(ids):
            a,b,n=self.row_shapes[i]
            result[j,:a,:b,:n]=self.values[self.offsets[i]:self.offsets[i+1]].reshape(a,b,n)
        return result
