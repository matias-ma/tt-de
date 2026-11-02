from itertools import product
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from scipy.special import roots_legendre
from test_gaussian_convolution import random_tt
from test_tt_convolution import correlated_polynomial_kernel
from ftde.particle_convolution import (fit_gaussian_kernel, kernel_moments, round_moments,
    ParticleTTConvolution, ParticleTTProblem, validate_covariances)
from ftde.tt_convolution import TTConvolution
from ftde.tt_density import tt_params, rebuild_tt, squared_integral_tt


def test_particle_moments_match_direct_operator_and_gradient():
    g=random_tt(d=3,n=5,rank=2);u=correlated_polynomial_kernel()
    y=np.array([[.2,-.3,.1],[-.9,.5,1.1]])
    moments=[round_moments(kernel_moments(u,x,g.domain,[5]*3),1e-13) for x in y]
    ranks=[max(h[k].shape[1] for h in moments) for k in range(3)]
    shapes=[(1 if k==0 else ranks[k-1],ranks[k],9) for k in range(3)]
    features=[]
    for k,shape in enumerate(shapes):
        a=np.zeros((len(y),*shape))
        for i,h in enumerate(moments):a[i,:h[k].shape[0],:h[k].shape[1]]=h[k]
        features.append(jnp.asarray(a.reshape(len(y),-1)))
    op=ParticleTTConvolution.from_tt(g,shapes)
    direct=TTConvolution.from_tt(g,u,zero_extension=True)
    np.testing.assert_allclose(op.log_density(g,features),direct.log_density(g,y),atol=2e-12)
    p=tt_params(g)
    a=jax.jit(jax.grad(lambda p:op.log_density(rebuild_tt(g,p),features).sum()))(p)
    b=jax.jit(jax.grad(lambda p:direct.log_density(rebuild_tt(g,p),y).sum()))(p)
    for aa,bb in zip(a,b):np.testing.assert_allclose(aa,bb,atol=2e-11)
    problem=ParticleTTProblem(g,[f.reshape(len(y),*s) for f,s in zip(features,shapes)],strength=.001)
    np.testing.assert_allclose(problem.logs(problem.params,problem.features(np.array([1,0]))),direct.log_density(g,y)[::-1],atol=2e-12)
    streamed=ParticleTTProblem(g,[f.reshape(len(y),*s) for f,s in zip(features,shapes)],stream=True,strength=.001)
    ids=np.array([1,0])
    np.testing.assert_allclose(streamed.logs(streamed.params,streamed.features(ids)),problem.logs(problem.params,problem.features(ids)),atol=2e-12)
    a=jax.jit(jax.grad(streamed.loss))(streamed.params,streamed.features(ids))
    b=jax.jit(jax.grad(problem.loss))(problem.params,problem.features(ids))
    for aa,bb in zip(a,b):np.testing.assert_allclose(aa,bb,atol=2e-11)
    with pytest.raises(ValueError,match='indices'):problem.features(np.array([-1]))


def test_ragged_moments_padding_and_row_order():
    from ftde.particle_convolution import RaggedMoments
    a=np.arange(12.).reshape(1,3,4);b=np.arange(8.).reshape(1,2,4)
    cache=RaggedMoments(np.r_[a.ravel(),b.ravel()],np.array([a.shape,b.shape]))
    selected=cache[np.array([1,0,1])]
    np.testing.assert_array_equal(selected[0,:,:2],b)
    np.testing.assert_array_equal(selected[1],a)
    np.testing.assert_array_equal(selected[2,:,2],0)
    with pytest.raises(ValueError,match='length'):RaggedMoments(np.zeros(3),np.array([a.shape]))


def test_full_gaussian_against_independent_dense_integral():
    g=random_tt(d=2,n=5,rank=2)
    covariance=np.array([[.16,.084],[.084,.09]])
    kernel,_,record=fit_gaussian_kernel(covariance,tolerance=1e-12)
    assert record['tail_mass_bound'] < 3e-15
    assert abs(record['relative_mass_error']) < 1e-8
    y=np.array([[.3,.2],[-.8,.9]])
    op=TTConvolution.from_tt(g,kernel,zero_extension=True)
    t,w=roots_legendre(48);ix=np.array(list(product(range(48),repeat=2)))
    x=t[ix];weight=np.prod(w[ix],axis=1)*np.asarray(g(x))**2
    delta=y[:,None,:]-x
    normal=np.exp(-.5*np.einsum('bni,ij,bnj->bn',delta,np.linalg.inv(covariance),delta))/(2*np.pi*np.sqrt(np.linalg.det(covariance)))
    expected=np.log((normal*weight).sum(1)/float(squared_integral_tt(g)))
    np.testing.assert_allclose(jax.jit(op.log_density)(g,y),expected,atol=2e-8)
    diag=np.diag(np.diag(covariance))
    diagonal=np.exp(-.5*np.einsum('bni,ij,bnj->bn',delta,np.linalg.inv(diag),delta))/(2*np.pi*np.sqrt(np.linalg.det(diag)))
    assert np.max(abs(expected-np.log((diagonal*weight).sum(1)/float(squared_integral_tt(g)))))>.01


def test_diagonal_limit_matches_existing_gaussian_operator():
    from ftde.gaussian_convolution import GaussianConvolution
    g=random_tt(d=2,n=7,rank=3)
    std=np.array([.25,.35]);points=np.array([[.3,.2],[-.8,.9],[1.2,-.2]])
    kernel,_,_=fit_gaussian_kernel(np.diag(std**2),tolerance=1e-13)
    general=TTConvolution.from_tt(g,kernel,zero_extension=True)
    diagonal=GaussianConvolution.from_tt(g,std)
    np.testing.assert_allclose(general.log_density(g,points),diagonal.log_density(g,points),atol=2e-10)


@pytest.mark.parametrize('covariance',[np.array([[1.,2.],[2.,1.]]),np.array([[1.,.2],[.1,1.]]),np.array([[np.nan]])])
def test_invalid_covariance(covariance):
    with pytest.raises(ValueError):validate_covariances(covariance)


def test_reused_cross_points_preserve_each_full_covariance():
    from ftde.gaussian_kernel_bank import GaussianKernelBank
    bank=GaussianKernelBank(tolerance=1e-8)
    points=np.random.default_rng(342).normal(0,.3,(100,2))
    for c in [np.array([[.16,.04],[.04,.09]]),np.array([[.25,.055],[.055,.12]])]:
        kernel,record=bank.fit(c)
        exact=np.exp(-.5*np.einsum('bi,ij,bj->b',points,np.linalg.inv(c),points))/(2*np.pi*np.sqrt(np.linalg.det(c)))
        np.testing.assert_allclose(kernel(points),exact,atol=1e-7)
        assert record['peak_absolute_error']<1e-8
