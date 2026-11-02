import argparse,json,time
import jax.numpy as jnp
import numpy as np
from chebttge.chebtt import ChebTT
from chebttge.cores import ChebyshevCoeffCore
from chebttge.computations import convolve_tt
from ftde.cheb_product import square_tt
from ftde.tt_convolution import TTConvolution
from ftde.tt_density import squared_integral_tt
p=argparse.ArgumentParser();p.add_argument('--rank',type=int,default=256);p.add_argument('--sketch',type=int);p.add_argument('--tol',type=float,default=1e-10);p.add_argument('--right-kernel',action='store_true');p.add_argument('--right-signal',action='store_true');a=p.parse_args()
s=np.load('artifacts/gl_rsi_trial/paired_setup.npz')
def load(label,domain):
 return ChebTT(cores=tuple(ChebyshevCoeffCore(jnp.asarray(s[f'{label}_{k}']),tuple(bounds)) for k,bounds in enumerate(domain)),domain=jnp.asarray(domain))
def right(tt):
 arrays=[np.asarray(c.coeffs).copy() for c in tt.cores]
 for k in range(len(arrays)-1,0,-1):
  left,r,n=arrays[k].shape
  q,rr=np.linalg.qr(arrays[k].reshape(left,-1).T,mode='reduced')
  arrays[k]=q.T.reshape(-1,r,n)
  arrays[k-1]=np.einsum('ain,ij->ajn',arrays[k-1],rr.T)
 return ChebTT(cores=tuple(ChebyshevCoeffCore(jnp.asarray(c),old.domain) for c,old in zip(arrays,tt.cores)),domain=tt.domain)
g=load('signal',s['domain']);u=load('kernel',s['kernel_domain'])
if a.right_kernel:u=right(u)
if a.right_signal:g=right(g)
start=time.monotonic()
r=convolve_tt(square_tt(g),u,output_domain=s['observation_domain'],max_rank=a.rank,sketch_size=a.sketch,tol=a.tol,validation_samples=64)
print('setup',time.monotonic()-start,r.ranks,r.diagnostics['validation_relative_rms_error'],flush=True)
op=TTConvolution.from_tt(g,u).prepare(s['observation_domain'],tolerance=1e-14)
exact=np.asarray(op.log_density(g,jnp.asarray(s['probe'])))
actual=np.asarray(r(jnp.asarray(s['probe'])))/float(squared_integral_tt(g))
print('log max',np.max(np.abs(np.log(np.maximum(actual,1e-300))-exact)),'relative density',np.linalg.norm(actual-np.exp(exact))/np.linalg.norm(np.exp(exact)),flush=True)
for st in r.diagnostics['stages']:
 print({k:st[k] for k in ['mode','rank','id_relative_residual','id_condition','local_tolerance_met']},flush=True)
