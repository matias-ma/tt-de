"""Independent checks of the gallery's marginal and convolution calculations."""

import sys
from pathlib import Path

import jax.numpy as jnp
import numpy as np
import pytest
from scipy.special import roots_legendre
from test_gaussian_convolution import random_tt

from ftde.cheb_product import square_tt
from ftde.ginzburg_landau import ginzburg_landau_log_normalizer
from ftde.noise_kernel import build_diagonal_gaussian_noise_kernel_tt
from ftde.tt_density import squared_integral_tt

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
from refresh_density_figures import (
    convolved_grids,
    evaluate_pair,
    histogram,
    pair_coefficients,
    physical_noise_grid,
)


def test_pair_coefficients_against_direct_cubature():
    signal = random_tt(d=3, rank=2, n=5)
    coeff, box = pair_coefficients(square_tt(signal), 0, 2)
    xs, ys = np.array([-.7, .2]), np.array([-.4, .3, .8])
    nodes, weights = roots_legendre(12)
    expected = np.empty((len(ys), len(xs)))
    for iy, y in enumerate(ys):
        for ix, x in enumerate(xs):
            grid = jnp.asarray(np.column_stack([np.full_like(nodes, x), nodes, np.full_like(nodes, y)]))
            expected[iy, ix] = weights @ np.asarray(signal(grid))**2 / float(squared_integral_tt(signal))
    np.testing.assert_allclose(evaluate_pair(coeff, box, xs, ys), expected, atol=2e-13)
    np.testing.assert_array_equal(evaluate_pair(coeff, box, [-2., 2.], ys), 0.)


def test_histogram_crop_preserves_probability_mass():
    points = np.array([[0., 0.], [.1, .1], [4., 4.], [5., 5.]])
    density = histogram(points, 0, 1, (-1., 1., -1., 1.), bins=10)
    np.testing.assert_allclose(density.sum()*.2**2, .5)


def test_noise_panel_uses_physical_displacements_and_density_jacobian():
    std, scale = np.array([.5, .4]), np.array([3., 5.])
    kernel = build_diagonal_gaussian_noise_kernel_tt(
        jnp.asarray(std**2), jnp.asarray([[-2., 2.], [-2., 2.]]), n_coeffs=(129,))
    # An asymmetric viewing window must not translate the zero-mean kernel.
    xs, ys = np.array([-2., 0., 3.]), np.array([-1., 0., 4.])
    expected_x = np.exp(-.5*(xs/(std[0]*scale[0]))**2)/(np.sqrt(2*np.pi)*std[0]*scale[0])
    expected_y = np.exp(-.5*(ys/(std[1]*scale[1]))**2)/(np.sqrt(2*np.pi)*std[1]*scale[1])
    actual = physical_noise_grid(kernel, 0, 1, xs, ys, scale)
    np.testing.assert_allclose(actual, np.outer(expected_y, expected_x), atol=1e-11)
    assert np.unravel_index(np.argmax(actual), actual.shape) == (1, 1)


def test_plot_fft_against_analytic_gaussian_convolution():
    def gaussian_coeff(std, radius, count):
        t = np.cos(np.pi*(np.arange(count)+.5)/count)
        values = np.exp(-.5*(t*radius/std)**2)/(std*np.sqrt(2*np.pi))
        c = np.polynomial.chebyshev.chebfit(t, values, count-1)
        return np.outer(c, c)

    p = gaussian_coeff(.4, 3., 129)
    u = gaussian_coeff(.3, 6., 257)
    xs = np.linspace(-2.5, 2.5, 41)
    _, observed = convolved_grids(2, 0, 1, p, np.tile([-3., 3.], (2, 1)),
        u, np.tile([-6., 6.], (2, 1)), xs, ginzburg_landau_log_normalizer(2), .0125)
    marginal = np.exp(-.5*(xs/.5)**2)/(.5*np.sqrt(2*np.pi))
    np.testing.assert_allclose(observed, np.outer(marginal, marginal), atol=1e-10)


def test_plot_rejects_small_negative_amplitude_with_large_negative_mass():
    # A low pointwise error spread over a large area is not harmless.
    with pytest.raises(FloatingPointError, match='too much negative mass'):
        convolved_grids(2, 0, 1, np.array([[.25]]), np.tile([-1., 1.], (2, 1)),
            np.array([[-1e-6]]), np.tile([-6., 6.], (2, 1)), np.linspace(-2.5, 2.5, 11),
            ginzburg_landau_log_normalizer(2), .025)
