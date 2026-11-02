"""Check conditional TT sampling against independent polynomial quadrature."""

import jax
import jax.numpy as jnp
import numpy as np
from chebttge.chebtt import ChebTT
from chebttge.cores import ChebyshevCoeffCore
from scipy.stats import kstest

from ftde.tt_density import density_tt, tt_params
from ftde.tt_sampling import make_squared_tt_moments, make_squared_tt_sampler

jax.config.update("jax_enable_x64", True)


def test_polynomial_inverse_cdf_and_correlated_moments():
    # Amplitude 1 + .4*x + .3*x*y on [-1,1]^2: a non-product density.
    a = np.zeros((1, 2, 3))
    a[0, 0, :2] = [1., .4]
    a[0, 1, 1] = .3
    b = np.zeros((2, 1, 3))
    b[0, 0, 0] = b[1, 0, 1] = 1.
    model = ChebTT(cores=tuple(ChebyshevCoeffCore(jnp.asarray(c), (-1., 1.)) for c in (a, b)),
                   domain=jnp.asarray([[-1., 1.], [-1., 1.]]))
    uniforms = np.random.default_rng(72).uniform(size=(16384, 2))
    draws = np.asarray(make_squared_tt_sampler(model)(tt_params(model), uniforms))
    # Conditional density in y is (A+B*y)^2. Its CDF is analytic.
    x, y = draws.T
    aa, bb = 1+.4*x, .3*x
    cdf = (aa*aa*(y+1)+aa*bb*(y*y-1)+bb*bb*(y**3+1)/3)/(2*aa*aa+2*bb*bb/3)
    np.testing.assert_allclose(cdf, uniforms[:, 1], atol=3e-11)
    assert kstest(cdf, "uniform").statistic < .015
    nodes, weights = np.polynomial.legendre.leggauss(20)
    xx, yy = np.meshgrid(nodes, nodes)
    points = np.stack([xx.ravel(), yy.ravel()], axis=1)
    pdf = np.asarray(density_tt(model, points)).reshape(xx.shape)
    w = weights[:, None]*weights[None]
    expected = np.array([np.sum(pdf*w*xx), np.sum(pdf*w*yy), np.sum(pdf*w*xx*yy)])
    means, variances = make_squared_tt_moments(model)(tt_params(model))
    np.testing.assert_allclose(means, expected[:2], atol=1e-13)
    np.testing.assert_allclose(variances, [np.sum(pdf*w*xx**2)-expected[0]**2,
                                         np.sum(pdf*w*yy**2)-expected[1]**2], atol=1e-13)
    actual = np.array([x.mean(), y.mean(), (x*y).mean()])
    np.testing.assert_allclose(actual, expected, atol=.013)


def test_constant_density_nonunit_box():
    box = np.array([[-3., 2.], [4., 7.]])
    model = ChebTT(cores=tuple(ChebyshevCoeffCore(jnp.ones((1, 1, 1)), tuple(b)) for b in box),
                   domain=jnp.asarray(box))
    u = np.random.default_rng(19).uniform(size=(40, 2))
    samples = make_squared_tt_sampler(model)(tt_params(model), u)
    np.testing.assert_allclose(samples, box[:, 0]+u*np.ptp(box, axis=1), atol=1e-10)


def test_high_degree_inverse_cdf_against_numpy_antiderivative():
    rng = np.random.default_rng(32)
    coefficients = rng.normal(size=49)/(1+np.arange(49))
    core = ChebyshevCoeffCore(jnp.asarray(coefficients[None, None]), (-2., 3.))
    model = ChebTT(cores=(core,), domain=jnp.asarray([[-2., 3.]]))
    uniforms = np.concatenate([[1e-9], rng.uniform(size=126), [1-1e-9]])[:, None]
    draws = np.asarray(make_squared_tt_sampler(model)(tt_params(model), uniforms))[:, 0]
    squared = np.polynomial.chebyshev.chebmul(coefficients, coefficients)
    integral = np.polynomial.chebyshev.chebint(squared, lbnd=-1)
    cdf = np.polynomial.chebyshev.chebval((2*draws-1)/5, integral)/np.polynomial.chebyshev.chebval(1., integral)
    np.testing.assert_allclose(cdf, uniforms[:, 0], atol=2e-9)
