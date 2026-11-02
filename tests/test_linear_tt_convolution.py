"""Validate the experimental unsquared forward operator against cubature."""

import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from chebttge.chebtt import ChebTT
from chebttge.computations import integrate
from chebttge.cores import ChebyshevCoeffCore
from scipy.special import roots_legendre

from ftde.linear_tt_convolution import LinearTTConvolution
from ftde.noise_kernel import build_diagonal_gaussian_noise_kernel_tt

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
from test_cryoem_unsquared import clipping_audit


def test_linear_convolution_and_gradient_match_independent_cubature():
    domain = jnp.array([[-1., 1.], [-1., 1.]])
    cores = tuple(ChebyshevCoeffCore(jnp.array([[[1., .1, .2]]]), tuple(box)) for box in domain)
    model = ChebTT(cores=cores, domain=domain, ranks=(1,))
    kernel = build_diagonal_gaussian_noise_kernel_tt(jnp.array([.2, .3]), domain, n_coeffs=(129,))
    operator = LinearTTConvolution.from_tt(model, kernel, np.array([[-2., 2.], [-2., 2.]]))
    nodes, weights = roots_legendre(80)
    xx, yy = np.meshgrid(nodes, nodes, indexing='ij')
    points = jnp.array(np.stack([xx.ravel(), yy.ravel()], axis=1))
    ww = jnp.array((weights[:, None]*weights[None, :]).ravel())
    observation = jnp.array([[.3, -.2], [-.4, .6]])
    def rebuilt(t):
        c = model.cores[0].coeffs.at[0, 0, 1].set(t)
        return ChebTT(cores=(ChebyshevCoeffCore(c, (-1., 1.)), model.cores[1]), domain=domain, ranks=(1,))
    def direct(t):
        f = rebuilt(t)
        return jnp.stack([jnp.sum(ww*f(points)*kernel(y-points))/integrate(f) for y in observation])
    np.testing.assert_allclose(operator.density(model, observation), direct(.1), atol=2e-12)
    np.testing.assert_allclose(jax.jacfwd(lambda t: operator.density(rebuilt(t), observation))(.1),
                               jax.jacfwd(direct)(.1), atol=2e-12)


def test_joint_clipping_mass_and_marginal_normalization(tmp_path):
    # Integral is one, but the negative part of (1+2*x0)/16 has mass 1/8.
    domain = jnp.tile(jnp.array([-1., 1.]), (4, 1))
    cores = [ChebyshevCoeffCore(jnp.array([[[1., 2.]]])/16, (-1., 1.))]
    cores.extend(ChebyshevCoeffCore(jnp.ones((1, 1, 1)), (-1., 1.)) for _ in range(3))
    model = ChebTT(cores=tuple(cores), domain=domain, ranks=(1, 1, 1))
    kernel = build_diagonal_gaussian_noise_kernel_tt(jnp.full(4, .3), domain, n_coeffs=(129,))
    operator = LinearTTConvolution.from_tt(model, kernel, np.tile([-1., 1.], (4, 1)))
    observations = np.zeros((8, 4))
    np.save(tmp_path/'validation.npy', np.log(operator.density(model, observations)))
    panels = clipping_audit(model, kernel, {'valid': observations}, tmp_path, 17)
    import json
    record = json.loads((tmp_path/'clipping_order17_n8.json').read_text())
    np.testing.assert_allclose(record['signed_integral'], 1., atol=1e-12)
    np.testing.assert_allclose(record['negative_mass'], .125, atol=.004)
    np.testing.assert_allclose(record['positive_mass'], 1+record['negative_mass'], atol=1e-12)
    _, weights = roots_legendre(17)
    for panel in panels.values():
        np.testing.assert_allclose(weights@panel['clipped']@weights, 1., atol=1e-12)
    np.testing.assert_allclose(panels['23']['clipped'], .25, atol=1e-12)
