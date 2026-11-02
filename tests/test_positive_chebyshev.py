"""Algebraic and optimizer checks independent of benchmark target densities."""
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from scipy.special import comb, roots_legendre

from ftde.positive_chebyshev import (
    ChebyshevCoreCoordinates,
    PositiveChebyshevTT,
    expand_positive_modes,
    initialize_positive_tt,
    positivity_certificate,
)
from ftde.positive_tt_fit import (
    PositiveTTForward,
    core_design,
    make_core_solver,
    transfer_right,
)


def test_certificate_is_polynomial_identity_and_bounds_between_nodes():
    rng = np.random.default_rng(34)
    for modes in (1, 9, 49, 81, 97):
        c = rng.normal(size=modes)
        a = positivity_certificate(modes, 16).reshape(16, modes, modes)
        s = np.linspace(0, 1, 27)
        degree = modes-1
        bernstein = np.array([comb(degree, j)*s**j*(1-s)**(degree-j) for j in range(modes)]).T
        for k in (0, 7, 15):
            t = -1+(k+s)/8
            np.testing.assert_allclose(bernstein@(a[k]@c), np.polynomial.chebyshev.chebval(t, c), atol=2e-11)
    # Positive at both endpoints, negative at the middle: cannot pass.
    assert np.min(positivity_certificate(3)@np.array([.4, 0., .5])) < 0


def model_fixture():
    x = np.random.default_rng(27).normal(size=(300, 2))*.4
    return initialize_positive_tt(x, np.array([[-2., 2.], [-2., 2.]]), rank=3, modes=17, width=.5)


def test_mass_support_marginal_and_serialization(tmp_path):
    model = model_fixture()
    report = model.validate()
    assert min(report['minimum_core_certificates']) > 0
    np.testing.assert_allclose(report['integral'], 1., atol=1e-12)
    t, w = roots_legendre(24)
    x, w = t*2, w*2
    xx, yy = np.meshgrid(x, x)
    p = np.asarray(model.density(np.stack([xx.ravel(), yy.ravel()], axis=1))).reshape(xx.shape)
    np.testing.assert_allclose(w@p@w, 1., atol=1e-12)
    np.testing.assert_allclose(model.pair_grid(0, 1, x, x), p, atol=1e-12)
    assert model.density(jnp.array([3., 0.])) == 0
    with pytest.raises(ValueError, match='points must have shape'):
        model.density(jnp.zeros((3, 3)))
    model.save(tmp_path/'model.npz')
    restored = PositiveChebyshevTT.load(tmp_path/'model.npz')
    np.testing.assert_allclose(restored.density(jnp.zeros((3, 2))), model.density(jnp.zeros((3, 2))))
    expanded = expand_positive_modes(model, 33)
    np.testing.assert_allclose(expanded.density(jnp.asarray(np.stack([xx.ravel(), yy.ravel()], axis=1))), p.ravel(), atol=1e-13)
    expanded.validate()


def test_convolution_matches_independent_integration_and_has_gradient():
    from ftde.noise_kernel import build_diagonal_gaussian_noise_kernel_tt
    model = model_fixture()
    kernel = build_diagonal_gaussian_noise_kernel_tt(jnp.array([.4**2, .6**2]), jnp.asarray(model.tt.domain), n_coeffs=(129,))
    forward = PositiveTTForward(model, np.array([[-2., 2.], [-2., 2.]]), kernel)
    params = forward.parameters(model)
    points = jnp.array([[.2, -.3], [-.5, .7]])
    q = np.exp(np.asarray(forward.log_density(params, points)))
    t, w = roots_legendre(100)
    x, w = t*2, w*2
    xx, yy = np.meshgrid(x, x)
    grid = np.stack([xx.ravel(), yy.ravel()], axis=1)
    p = np.asarray(model.density(jnp.asarray(grid)))
    sigma = np.array([.4, .6])
    noise = np.prod(np.exp(-.5*((np.asarray(points)[:, None]-grid)/sigma)**2)/(np.sqrt(2*np.pi)*sigma), axis=-1)
    expected = noise@(p*np.outer(w, w).ravel())
    np.testing.assert_allclose(q, expected, rtol=2e-8, atol=1e-10)
    g = jax.grad(lambda u: -forward.log_density((u, params[1]), points).mean())(params[0])
    assert np.all(np.isfinite(g)) and np.linalg.norm(g) > 0


def test_middle_core_and_nonadjacent_marginal_integrate_correctly():
    from ftde.tt_density import squared_integral_tt
    data = np.random.default_rng(31).normal(size=(200, 3))*.4
    model = initialize_positive_tt(data, np.array([[-2., 2.]]*3), rank=3, modes=9, width=.6)
    t, w = roots_legendre(12)
    x, w = t*2, w*2
    points = np.stack(np.meshgrid(x, x, x, indexing='ij'), axis=-1).reshape(-1, 3)
    p = np.asarray(model.density(jnp.asarray(points))).reshape(12, 12, 12)
    np.testing.assert_allclose(np.einsum('ijk,i,j,k->', p, w, w, w), 1., atol=2e-13)
    np.testing.assert_allclose(np.einsum('ijk,j->ki', p, w), model.pair_grid(0, 2, x, x), atol=2e-13)
    np.testing.assert_allclose(np.einsum('ijk,i,j,k->', p*p, w, w, w), squared_integral_tt(model.tt), atol=2e-13)


def test_convex_block_update_decreases_loss_and_preserves_certificate_and_mass():
    model = model_fixture()
    forward = PositiveTTForward(model)
    params = forward.parameters(model)
    x = jnp.asarray(np.random.default_rng(1).uniform(-1, 1, (64, 2)))
    h = [f(x[:, k]) for k, f in enumerate(forward.features)]
    boundary = jnp.ones((len(x), 1, 1))
    right = transfer_right(boundary, params[1], h[1])
    design = core_design(boundary, right, h[0], params[0])
    coord = ChebyshevCoreCoordinates(17)
    solve = make_core_solver(coord, 4.)
    u, info = solve(params[0], design, .001, 1e-7)
    assert info[5] == 1 and info[1] < info[0]
    assert np.min(np.asarray(u)@coord.bounds.T) > 0
    np.testing.assert_allclose(np.sum(np.asarray(u)*coord.weights, axis=(1, 2)), u.shape[1], atol=1e-11)
    # Verify normalized convolution is linear along the equality tangent.
    direction = coord.project(jnp.asarray(np.random.default_rng(2).normal(size=u.shape)))
    eps = 1e-8
    q0 = np.exp(np.asarray(forward.log_density(params, x)))
    q1 = np.exp(np.asarray(forward.log_density((params[0]+eps*direction, params[1]), x)))
    prediction = np.einsum('nabi,abi->n', np.asarray(design), np.asarray(direction))
    np.testing.assert_allclose((q1/q0-1)/eps, prediction, rtol=2e-5, atol=1e-6)


def test_correlated_noise_design_matches_full_forward_and_direct_quadrature():
    from chebttge.chebtt import ChebTT
    from chebttge.cores import ChebyshevCoeffCore
    model = model_fixture()
    # A normalized, positive, nonseparable polynomial noise on [-6, 6]^2.
    a = np.zeros((1, 2, 2))
    b = np.zeros((2, 1, 2))
    a[0, 0, 0], a[0, 1, 1] = 1., 1.
    b[0, 0, 0], b[1, 0, 1] = 1/144, .7/144
    kernel = ChebTT(cores=(ChebyshevCoeffCore(jnp.asarray(a), (-6., 6.)),
                          ChebyshevCoeffCore(jnp.asarray(b), (-6., 6.))),
                    domain=jnp.array([[-6., 6.], [-6., 6.]]), ranks=(2,))
    forward = PositiveTTForward(model, np.array([[-2., 2.], [-2., 2.]]), kernel)
    params = forward.parameters(model)
    y = jnp.array([[.2, -.3], [-.5, .7], [1., -1.]])
    features = [f(y[:, k]) for k, f in enumerate(forward.features)]
    boundary = jnp.ones((len(y), 1, 1))
    right = transfer_right(boundary, params[1], features[1])
    design = core_design(boundary, right, features[0], params[0])
    coord = forward.coordinates[0]
    direction = coord.project(jnp.asarray(np.random.default_rng(91).normal(size=params[0].shape)))
    eps = 1e-7
    q0 = np.exp(np.asarray(forward.log_density(params, y)))
    q1 = np.exp(np.asarray(forward.log_density((params[0]+eps*direction, params[1]), y)))
    prediction = np.einsum('nabi,abi->n', np.asarray(design), np.asarray(direction))
    np.testing.assert_allclose((q1/q0-1)/eps, prediction, rtol=2e-4, atol=2e-7)
    t, w = roots_legendre(24)
    xx, yy = np.meshgrid(t*2, t*2)
    grid = np.stack([xx.ravel(), yy.ravel()], axis=1)
    z = np.asarray(y)[:, None]-grid
    noise = (1+.7*z[..., 0]*z[..., 1]/36)/144
    expected = noise@(np.asarray(model.density(jnp.asarray(grid)))*np.outer(w*2, w*2).ravel())
    np.testing.assert_allclose(q0, expected, atol=1e-13)
