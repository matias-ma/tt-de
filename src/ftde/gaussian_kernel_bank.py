"""Reuse chebdGE cross interpolation points for nearby Gaussian covariances.

Every reuse resamples the Gaussian with the NEW full covariance and solves
its cross matrices; no kernel or covariance averaging occurs. Independent
checks of the resulting Chebyshev TT and its mass decide acceptance. Failed
reuse triggers a fresh adaptive chebdGE fit. These sampled checks are not
rigorous uniform-error bounds. Use direct fits to audit likelihood accuracy.
"""
import warnings
import numpy as np
from scipy.fft import dct
from scipy.linalg import solve, LinAlgWarning
from chebttge.chebtt import ChebTT
from chebttge.cores import ChebyshevCoeffCore
from .particle_convolution import fit_gaussian_kernel, validate_covariances


def evaluate(cores,x):
    v=np.ones((len(x),1))
    for k,c in enumerate(cores):
        b=np.polynomial.chebyshev.chebvander(x[:,k],c.shape[-1]-1)
        local=(b@c.reshape(-1,c.shape[-1]).T).reshape(len(x),*c.shape[:2])
        v=np.einsum('bi,bij->bj',v,local)
    return v[:,0]


def integral(cores):
    state=np.ones(1)
    for c in cores:
        n=np.arange(c.shape[-1]);w=np.zeros(len(n));w[::2]=2/(1-n[::2]**2)
        state=state@np.einsum('abn,n->ab',c,w)
    return float(state[0])


def coefficients(values):
    c=dct(values,type=1,axis=-1)/(values.shape[-1]-1)
    c[...,0]*=.5;c[...,-1]*=.5
    return c


def interpolate(states,precision,modes=129):
    d=len(precision);t=np.cos(np.pi*np.arange(modes)/(modes-1));cores=[]
    def f(x):return np.exp(-.5*np.einsum('...i,ij,...j->...',x,precision,x))
    if d==1:
        return (coefficients(f(t[:,None])[None,None,:]),)
    for k,s in enumerate(states):
        pre=np.asarray(s.prefixes);cols=np.asarray(s.col_ys)
        rows=np.column_stack([pre[np.asarray(s.row_left_indices)],np.asarray(s.row_xs)])
        a,b=len(pre),len(cols)
        grid=np.concatenate([np.broadcast_to(pre[:,None,None,:],(a,b,modes,k)),
            np.broadcast_to(t[None,None,:,None],(a,b,modes,1)),
            np.broadcast_to(cols[None,:,None,:],(a,b,modes,d-k-1))],axis=-1)
        matrix=f(np.concatenate([np.broadcast_to(rows[:,None,:],(b,b,k+1)),
            np.broadcast_to(cols[None,:,:],(b,b,d-k-1))],axis=-1))
        vals=f(grid).transpose(0,2,1)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore',LinAlgWarning)
            # Never form an explicit inverse. A poorly conditioned reuse must
            # still pass the independent target and mass checks below.
            vals=solve(matrix.T,vals.reshape(-1,b).T).T.reshape(a,modes,b).transpose(0,2,1)
        cores.append(coefficients(vals))
    vals=f(np.concatenate([np.broadcast_to(rows[:,None,:],(len(rows),modes,d-1)),
        np.broadcast_to(t[None,:,None],(len(rows),modes,1))],axis=-1))
    cores.append(coefficients(vals[:,None,:]))
    return tuple(cores)


class GaussianKernelBank:
    def __init__(self, *, tolerance=1e-8, fit_tolerance=1e-12, sigmas=8., seed=20261002):
        if not 0<tolerance<1 or not 0<fit_tolerance<1 or sigmas<=0:
            raise ValueError('invalid kernel fitting tolerances or support')
        self.tolerance=tolerance;self.fit_tolerance=fit_tolerance
        self.sigmas=sigmas;self.seed=seed;self.templates=[]

    def fit(self,covariance):
        c=validate_covariances(covariance);d=len(c)
        sd=np.sqrt(c.diagonal());r=c/sd[:,None]/sd[None,:]
        width=self.sigmas*sd;precision=np.linalg.inv(r)*self.sigmas**2
        rng=np.random.default_rng(self.seed)
        x=np.concatenate([np.zeros((1,d)),rng.normal(size=(512,d))@np.linalg.cholesky(r).T/self.sigmas,
                          rng.uniform(-1,1,(256,d))])
        x=x[np.all(abs(x)<=1,axis=1)]
        truth=np.exp(-.5*np.einsum('bi,ij,bj->b',x,precision,x))
        expected=(2*np.pi)**(d/2)*np.sqrt(np.linalg.det(r))/self.sigmas**d
        def check(cores):
            error=float(np.max(abs(evaluate(cores,x)-truth)))
            mass=integral(cores);mass_error=abs(mass/expected-1)
            okay=np.isfinite(error+mass_error) and error<=self.tolerance and mass_error<=self.tolerance*10
            return okay,dict(peak_absolute_error=error,relative_mass_error=mass_error),mass
        # A nearby low-rank cross may miss a direction that a more distant,
        # richer cross resolves. Trying a few more validated reuses costs much
        # less than rebuilding that cross, without weakening either check.
        order=sorted(range(len(self.templates)),key=lambda j:np.linalg.norm(r-self.templates[j][0]))[:16]
        for j in order:
            try:
                cores=interpolate(self.templates[j][1],precision)
                okay,record,mass=check(cores)
            except (np.linalg.LinAlgError,ValueError,FloatingPointError):
                continue
            if okay:break
        else:
            _,fitted,diag=fit_gaussian_kernel(c,sigmas=self.sigmas,tolerance=self.fit_tolerance,seed=self.seed)
            cores=tuple(np.asarray(a.coeffs) for a in fitted.cores)
            okay,record,mass=check(cores)
            if not okay:
                _,fitted,diag=fit_gaussian_kernel(c,sigmas=self.sigmas,tolerance=self.fit_tolerance,
                    max_rank=192,n_starts=512,seed=self.seed+1)
                cores=tuple(np.asarray(a.coeffs) for a in fitted.cores)
                okay,record,mass=check(cores)
            if not okay:
                _,fitted,diag=fit_gaussian_kernel(c,sigmas=self.sigmas,tolerance=self.fit_tolerance,
                    max_rank=256,n_starts=1024,seed=self.seed+1)
                cores=tuple(np.asarray(a.coeffs) for a in fitted.cores)
                okay,record,mass=check(cores)
            if not okay:
                raise FloatingPointError(f'adaptive Gaussian kernel failed validation: {record}, {diag}')
            j=len(self.templates);self.templates.append((r,fitted.cross_states))
            record['adaptive_fit']=diag
        # Normalize in scaled coordinates; physical volume is prod(width).
        cores=list(cores);cores[0]=cores[0]/(mass*np.prod(width))
        box=np.stack([-width,width],axis=1)
        kernel=ChebTT(cores=tuple(ChebyshevCoeffCore(a,tuple(b)) for a,b in zip(cores,box)),domain=box)
        record.update(template=j,n_templates=len(self.templates),ranks=list(kernel.ranks))
        return kernel,record
