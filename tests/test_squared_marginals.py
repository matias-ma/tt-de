"""Compare efficient squared-TT marginals with explicit density products."""

import sys
from pathlib import Path

import numpy as np
from test_gaussian_convolution import random_tt

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))

from ftde.cheb_product import square_tt
from ftde.marginals import (
    relative_pair_hessian_energy,
    squared_pair_coefficients,
    squared_pair_grid,
)
from scripts.refresh_density_figures import evaluate_pair, pair_coefficients


def test_nonadjacent_pair_matches_explicit_square_and_handles_support():
    model = random_tt(d=4, rank=3, n=5)
    xs, ys = np.linspace(-1.2, 1.2, 35), np.linspace(-1, 1, 29)
    for i, j in [(0, 1), (0, 3), (1, 3)]:
        coeff, box = pair_coefficients(square_tt(model), i, j)
        expected = evaluate_pair(coeff, box, xs, ys)
        np.testing.assert_allclose(squared_pair_grid(model, i, j, xs, ys), expected, atol=2e-12)


def test_pair_coefficients_and_hessian_energy_against_quadrature():
    from scipy.special import roots_legendre

    model = random_tt(d=3, rank=2, n=4)
    coefficients, box = squared_pair_coefficients(model, 0, 2)
    expected, _ = pair_coefficients(square_tt(model), 0, 2)
    xs = np.linspace(-1, 1, 15)
    np.testing.assert_allclose(np.polynomial.chebyshev.chebgrid2d(xs, xs, coefficients),
                               evaluate_pair(expected, box, xs, xs), atol=2e-12)
    t, w = roots_legendre(12)
    weights = w[:, None]*w[None, :]
    # Independent physical quadrature of analytic polynomial derivatives.
    def norm(c):
        return np.sum(np.polynomial.chebyshev.chebgrid2d(t, t, c)**2*weights)
    xx = np.polynomial.chebyshev.chebder(coefficients, m=2, axis=1)
    yy = np.polynomial.chebyshev.chebder(coefficients, m=2, axis=0)
    xy = np.polynomial.chebyshev.chebder(np.polynomial.chebyshev.chebder(coefficients, axis=0), axis=1)
    scales = (.2, .4)
    value = (scales[0]**4*norm(xx)+2*np.prod(scales)**2*norm(xy)+scales[1]**4*norm(yy))/norm(coefficients)
    np.testing.assert_allclose(relative_pair_hessian_energy(coefficients, box, scales), value, rtol=2e-12)
