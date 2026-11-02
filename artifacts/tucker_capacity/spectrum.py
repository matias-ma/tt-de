import numpy as np
from ftde.tucker_operations import load_tucker
from ftde.cheb_gram import chebyshev_gram_matrix
from ftde.tucker_density import _mode_product
m=load_tucker('artifacts/tucker_capacity/r8/swirl6_free/strong/fit.npz')
c=m.core
for k,f in enumerate(m.factors):
 a=np.asarray(f.core.coeffs[0]); chol=np.linalg.cholesky(np.asarray(chebyshev_gram_matrix(a.shape[1],f.domain)))
 q,r=np.linalg.qr((a@chol).T)
 c=_mode_product(c,r,k)
a=np.asarray(c).reshape(512,512); s=np.linalg.svd(a,compute_uv=False)
for eps in [1e-2,1e-3,1e-4,1e-5,1e-6]:
 tails=np.sqrt(np.cumsum(s[::-1]**2)[::-1])/np.linalg.norm(s)
 print(eps,np.sum(tails>eps))
print(s[:20],s[::32])
