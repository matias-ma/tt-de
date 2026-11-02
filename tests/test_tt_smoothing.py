import jax.numpy as jnp
import numpy as np
from chebttge.chebtt import ChebTT
from chebttge.cores import ChebyshevCoeffCore

from ftde.cheb_gram import chebyshev_gram_matrix
from ftde.simple_initialization import init_tt_constant
from ftde.sobolev import AmplitudeSobolevPenalty
from ftde.tt_smoothing import heat_smooth_tt


def integral(model):
    value = np.ones(1)
    for core in model.cores:
        moments = np.asarray(chebyshev_gram_matrix(core.coeffs.shape[-1], core.domain))[:, 0]
        value = value@np.einsum('abn,n->ab', np.asarray(core.coeffs), moments)
    return value.item()


def test_diffusion_preserves_mass_and_constant_on_nonsymmetric_box():
    box = np.array([[-2., 3.], [.5, 2.]])
    for perturbation in (0., .3):
        model = init_tt_constant(box, rank=3, modes=17, perturbation=perturbation)
        smooth = heat_smooth_tt(model, [.3, .2])
        np.testing.assert_allclose(integral(smooth), integral(model), rtol=2e-12)
        if perturbation == 0:
            points = jnp.array([[-1.3, .8], [2., 1.7]])
            np.testing.assert_allclose(smooth(points), model(points), rtol=2e-12)


def test_diffusion_matches_known_neumann_cosine_decay():
    n, sigma, a, b = 49, .19, -.7, 2.3
    t = np.cos(np.pi*np.arange(n)/(n-1))
    x = a+(t+1)*(b-a)/2
    values = 1+.6*np.cos(3*np.pi*(x-a)/(b-a))
    coeff = np.polynomial.chebyshev.chebfit(t, values, n-1)
    model = ChebTT(cores=(ChebyshevCoeffCore(jnp.asarray(coeff[None, None]), (a, b)),),
                   domain=jnp.array([[a, b]]))
    smooth = heat_smooth_tt(model, [sigma])
    points = np.linspace(a, b, 101)
    expected = 1+.6*np.exp(-.5*(sigma*3*np.pi/(b-a))**2)*np.cos(3*np.pi*(points-a)/(b-a))
    np.testing.assert_allclose(smooth(jnp.asarray(points[:, None])), expected, rtol=2e-10, atol=2e-10)


def test_diffusion_composes_and_reduces_unscaled_gradient_energy():
    model = init_tt_constant(np.array([[-1., 1.], [-2., 2.]]), modes=17, rank=4, perturbation=.5)
    first, second = np.array([.08, .17]), np.array([.12, .09])
    combined = heat_smooth_tt(model, np.sqrt(first**2+second**2))
    sequential = heat_smooth_tt(heat_smooth_tt(model, first), second)
    points = jnp.asarray(np.random.default_rng(7).uniform([-1., -2.], [1., 2.], (29, 2)))
    np.testing.assert_allclose(combined(points), sequential(points), rtol=2e-11, atol=2e-11)
    from ftde.tt_density import squared_integral_tt
    energy = AmplitudeSobolevPenalty.from_tt(model, order=1)
    assert float(energy(combined)*squared_integral_tt(combined)) < float(energy(model)*squared_integral_tt(model))
