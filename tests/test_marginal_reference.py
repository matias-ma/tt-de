import numpy as np
import pytest
from scipy.integrate import quad

from ftde.marginal_reference import fit_noisy_marginal_reference


def test_marginal_pilot_predictions_against_independent_integral():
    # These arrays are the observed data. No latent samples or true density
    # are supplied to the pilot or to a selection decision.
    rng = np.random.default_rng(15)
    train = rng.normal(size=(3000, 2))*[1., 1.3]+[.2, -.1]
    valid = rng.normal(size=(12, 2))*[1., 1.3]+[.2, -.1]
    domain, std = np.array([[-3., 3.], [-4., 3.]]), np.array([.5, .7])
    coefficients, logs, diagnostic = fit_noisy_marginal_reference(
        train, valid, domain, std, degree=2, shared=False, order=96, bins=200)
    assert diagnostic['success']
    independent = []
    for k, (a, b) in enumerate(domain):
        def rho(x, a=a, b=b, c=coefficients[k]):
            return np.exp(np.polynomial.chebyshev.chebval(2*(x-a)/(b-a)-1, c))
        normalization = quad(rho, a, b, epsabs=1e-11)[0]
        independent.append([np.log(quad(
            lambda x, y=y, s=std[k]: rho(x)*np.exp(-.5*((y-x)/s)**2)/(s*np.sqrt(2*np.pi)),
            a, b, epsabs=1e-11)[0]/normalization) for y in valid[:, k]])
    np.testing.assert_allclose(logs, np.mean(independent, axis=0), atol=2e-11)


def test_marginal_pilot_rejects_invalid_noise():
    with pytest.raises(ValueError, match='noise scales positive'):
        fit_noisy_marginal_reference(np.ones((3, 1)), np.ones((2, 1)),
                                    [[-1., 1.]], [0.], degree=2, shared=True)
