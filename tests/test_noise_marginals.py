import jax.numpy as jnp
import numpy as np
from chebttge.chebtt import ChebTT
from chebttge.cores import ChebyshevCoeffCore
from scipy.integrate import quad
from scipy.special import roots_legendre
from test_gaussian_convolution import random_tt

from ftde.fisher_visibility import FisherVisibilityPenalty, marginal_fisher_filter
from ftde.marginal_reference import fit_noisy_marginal_reference
from ftde.noise_marginals import PolynomialNoiseMarginal, noise_marginals_from_tt


def correlated_polynomial_noise():
    c1, c2 = np.array([1., .25]), np.array([1., -.4])
    first = np.stack([np.pad(c1, (0, 1)), np.polynomial.chebyshev.chebmul(c1, [0., 1.])])[None]
    last = np.stack([np.pad(c2, (0, 1)), .6*np.polynomial.chebyshev.chebmul(c2, [0., 1.])])[:, None]
    boxes = [(-.7, 1.1), (-1.5, .5)]
    return ChebTT(cores=tuple(ChebyshevCoeffCore(jnp.asarray(c), box) for c, box in zip([first, last], boxes, strict=True)),
                  domain=jnp.asarray(boxes))


def test_correlated_non_gaussian_marginals_against_cubature():
    noise = correlated_polynomial_noise()
    marginals = noise_marginals_from_tt(noise)
    t, w = roots_legendre(20)
    for k, marginal in enumerate(marginals):
        a, b = marginal.domain
        other_a, other_b = noise.cores[1-k].domain
        x, y = a+(t+1)*(b-a)/2, other_a+(t+1)*(other_b-other_a)/2
        xx, yy = np.meshgrid(x, y, indexing='ij')
        points = np.stack([xx.ravel(), yy.ravel()], axis=1)
        if k == 1:
            points = points[:, ::-1]
        values = np.asarray(noise(jnp.asarray(points))).reshape(20, 20)
        expected = values@(w*(other_b-other_a)/2)
        expected /= np.sum(expected*w*(b-a)/2)
        np.testing.assert_allclose(marginal.pdf(x), expected, atol=1e-13)
        for z in np.linspace(a, b, 9):
            np.testing.assert_allclose(marginal.cdf(z), quad(marginal.pdf, a, z)[0], atol=1e-13)
        mean, std = marginal.moments()
        np.testing.assert_allclose(mean, np.sum(x*expected*w*(b-a)/2), atol=1e-13)
        np.testing.assert_allclose(std**2, np.sum((x-mean)**2*expected*w*(b-a)/2), atol=1e-13)


def test_uniform_noise_information_oracle_and_refinement():
    noise = PolynomialNoiseMarginal(np.array([.5]), (-1., 1.))
    # X and noise uniform(-1,1): E[X|Y]=Y/2 and Var(Y)=2 Var(X),
    # hence exactly half of the linear-score information is retained.
    _, spectrum, diagnostic = marginal_fisher_filter(2, (-1., 1.), [0.], noise_marginal=noise, order=64)
    np.testing.assert_allclose(spectrum, [.5], atol=1e-13)
    np.testing.assert_allclose(diagnostic['observed_reference_mass'], 1., atol=1e-13)
    first, _, _ = marginal_fisher_filter(9, (-1., 1.), [0.], noise_marginal=noise, order=64)
    refined, _, _ = marginal_fisher_filter(9, (-1., 1.), [0.], noise_marginal=noise, order=128)
    np.testing.assert_allclose(first, refined, atol=1e-11)


def test_tt_noise_filter_matches_gaussian_shortcut():
    g = random_tt(d=2, rank=2, n=7)
    std = [.5, .8]
    # Use eight-sigma support: the filter can amplify six-sigma truncation
    # discrepancies by up to 1 / visibility_floor.
    boxes = [(-8*s, 8*s) for s in std]
    cores = [ChebyshevCoeffCore(jnp.asarray(np.polynomial.chebyshev.chebinterpolate(
        lambda t, s=s: np.exp(-.5*(8*t)**2)/(s*np.sqrt(2*np.pi)), 96))[None, None], box)
        for s, box in zip(std, boxes, strict=True)]
    noise = ChebTT(cores=tuple(cores), domain=jnp.asarray(boxes))
    references = [[0., .1, -.4], [0., -.2, -.6]]
    generic = FisherVisibilityPenalty.from_tt(g, references, noise_tt=noise, order=128)
    gaussian = FisherVisibilityPenalty.from_tt(g, references, std, order=128)
    for a, b in zip(generic.derivatives, gaussian.derivatives, strict=True):
        np.testing.assert_allclose(a, b, atol=1e-8)


def test_non_gaussian_pilot_predictions_against_adaptive_quadrature():
    rng = np.random.default_rng(17)
    train, valid = rng.uniform(-1., 1.5, (1000, 1)), rng.uniform(-1., 1.5, (12, 1))
    noise = ChebTT(cores=(ChebyshevCoeffCore(jnp.array([[[1., .3]]]), (-.4, .8)),),
                   domain=jnp.array([[-.4, .8]]))
    coefficients, logs, diagnostic = fit_noisy_marginal_reference(
        train, valid, [[-1., 1.]], degree=2, noise_tt=noise, order=48, bins=60)
    assert diagnostic['success']
    marginal = noise_marginals_from_tt(noise)[0]
    rho = lambda x: np.exp(np.polynomial.chebyshev.chebval(x, coefficients[0]))
    normalization = quad(rho, -1., 1., epsabs=1e-11)[0]
    expected = [np.log(quad(lambda x, y=y: rho(x)*marginal.pdf(y-x),
                           max(-1., y-.8), min(1., y+.4), epsabs=1e-11)[0]/normalization)
                for y in valid[:, 0]]
    np.testing.assert_allclose(logs, expected, atol=2e-11)
