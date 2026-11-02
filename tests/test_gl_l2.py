"""Check post-fit GL error integrals against independent tensor cubature."""

import jax.numpy as jnp
import numpy as np
import pytest
from scipy.special import roots_legendre
from test_gaussian_convolution import random_tt

from ftde.ginzburg_landau import ginzburg_landau_density, ginzburg_landau_log_normalizer
from scripts.evaluate_gl_l2 import integrated_terms, relative_l2_from_terms


def test_integrated_error_terms_against_tensor_cubature():
    g = random_tt(d=2, rank=2, n=5)
    nodes, weights = roots_legendre(48)
    x, y = np.meshgrid(nodes, nodes, indexing='ij')
    grid = jnp.asarray(np.stack([x.ravel(), y.ravel()], axis=1))
    w = (weights[:, None]*weights[None, :]).ravel()
    raw = np.asarray(g(grid))**2
    q = raw/np.sum(w*raw)
    log_z = ginzburg_landau_log_normalizer(2)
    p = np.asarray(ginzburg_landau_density(grid, log_z=log_z))
    expected = [np.sum(w*q*q), np.sum(w*p*q)]
    np.testing.assert_allclose(integrated_terms(g, 64, log_z), expected, rtol=2e-12)


def test_gl_normalizer_uses_exact_zero_boundary_on_even_grids():
    even = ginzburg_landau_log_normalizer(10, n_grid=400)
    odd = ginzburg_landau_log_normalizer(10, n_grid=401)
    np.testing.assert_allclose(even, odd, atol=1e-12, rtol=0)


def test_invalid_integrals_cannot_report_perfect_accuracy():
    with pytest.raises(FloatingPointError):
        relative_l2_from_terms(np.nan, 1., 1.)
    with pytest.raises(FloatingPointError):
        relative_l2_from_terms(1., 2., 1.)
