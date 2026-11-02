"""Independent checks of elementary starts and the signed quadratic objective."""

import jax
import jax.numpy as jnp
import numpy as np
from chebttge.computations import integrate
from scipy.special import ndtr, roots_legendre

from ftde.linear_tt_convolution import LinearTTConvolution
from ftde.noise_kernel import (
    build_diagonal_gaussian_noise_kernel_tt,
    build_paired_gaussian_mixture_noise_kernel_tt,
)
from ftde.signed_tt_least_squares import SignedTTLeastSquares4D
from ftde.simple_initialization import init_tt_constant
from ftde.tt_density import squared_integral_tt


def test_constant_initializer_has_no_density_fitting_and_active_ranks():
    box = np.tile([-1., 1.], (4, 1))
    exact = init_tt_constant(box, modes=5, rank=3, perturbation=0.)
    points = jnp.asarray(np.random.default_rng(0).uniform(-1, 1, (32, 4)))
    np.testing.assert_allclose(exact(points), .25, atol=1e-14)
    perturbed = init_tt_constant(box, modes=5, rank=3)
    np.testing.assert_allclose(squared_integral_tt(perturbed), 1, atol=1e-13)
    assert max(np.asarray(c.coeffs).size for c in perturbed.cores) > 5
    assert np.std(perturbed(points)) > 0


def test_uniform_convolved_squared_norm_matches_independent_normal_cdf_integrals():
    box = np.tile([-1., 1.], (4, 1))
    sigmas = np.array([.3, .4, .25, .35])
    geometry = SignedTTLeastSquares4D.from_noise(box, 3, np.diag(sigmas**2)[None], [1.], order=25)
    left, right = geometry.initial_factors(1, perturbation=0.)
    value, roughness = geometry.terms(left, right, jnp.zeros((9, 9)))
    t, w = roots_legendre(240)
    y, ww = 4*t, 4*w
    norm = np.prod([np.sum(ww*((ndtr((y+1)/s)-ndtr((y-1)/s))/2)**2) for s in sigmas])
    np.testing.assert_allclose(value, norm, rtol=2e-11, atol=1e-13)
    np.testing.assert_allclose(roughness, 0., atol=1e-13)
    np.testing.assert_allclose(integrate(geometry.model(left, right)), 1., atol=1e-13)


def test_empirical_linear_term_matches_forward_and_signed_gradient():
    box = np.tile([-1., 1.], (4, 1))
    geometry = SignedTTLeastSquares4D.from_noise(box, 3, np.eye(4)[None]*.3**2, [1.], order=21)
    left, right = geometry.initial_factors(2, seed=42, perturbation=.4)
    model = geometry.model(left, right)
    kernel = build_diagonal_gaussian_noise_kernel_tt(jnp.full(4, .3), jnp.asarray(box), n_coeffs=(129,))
    operator = LinearTTConvolution.from_tt(model, kernel, np.tile([-2., 2.], (4, 1)))
    observations = np.random.default_rng(52).normal(size=(15, 4))*.3
    empirical = geometry.empirical_term(operator, observations, batch=8)
    np.testing.assert_allclose(jnp.sum(left*(empirical@right.T)),
                              jnp.mean(operator.density(model, jnp.asarray(observations))), rtol=2e-11)
    direction = jnp.asarray(np.random.default_rng(3).normal(size=left.shape))
    loss = lambda t: geometry.terms(left+t*direction, right, empirical)[0]
    # The quadratic loss has an ordinary derivative even at negative density.
    derivative = float(jax.grad(loss)(-2.))
    finite_difference = float((loss(-2.+1e-5)-loss(-2.-1e-5))/2e-5)
    np.testing.assert_allclose(derivative, finite_difference, rtol=1e-7)
    assert abs(derivative) > 1e-5


def test_constrained_half_solve_matches_dense_kkt_solution():
    box = np.tile([-1., 1.], (4, 1))
    geometry = SignedTTLeastSquares4D.from_noise(box, 3, np.eye(4)[None]*.3**2, [1.], order=21)
    left, right = geometry.initial_factors(2, seed=42, perturbation=.1)
    empirical = jnp.asarray(np.random.default_rng(91).normal(size=(9, 9))*.01)
    strength, ridge = .01, 1e-7
    solver = geometry.make_half_solver(strength=strength, ridge=ridge, tolerance=1e-11)
    fitted, new_right, residual = solver(left, right, empirical)
    ml, mr = geometry.pair_moments
    np.testing.assert_allclose(ml@fitted@new_right@mr, 1., atol=1e-11)
    assert float(residual) < 1e-9
    def loss(flat):
        x = flat.reshape(fitted.shape)
        a, b = geometry.terms(x, new_right, empirical)
        return a+strength*b+ridge*jnp.sum(x*x)
    zero = jnp.zeros(fitted.size)
    hessian = np.asarray(jax.hessian(loss)(zero))
    rhs = -np.asarray(jax.grad(loss)(zero))
    mass = np.asarray(ml[:, None]*(new_right@mr)[None]).ravel()
    kkt = np.block([[hessian, mass[:, None]], [mass[None], np.zeros((1, 1))]])
    expected = np.linalg.solve(kkt, np.r_[rhs, 1.])[:-1]
    np.testing.assert_allclose(fitted.ravel(), expected, rtol=2e-8, atol=2e-9)


def test_soft_negative_penalty_core_step_preserves_mass_and_decreases_true_loss():
    box = np.tile([-1., 1.], (4, 1))
    geometry = SignedTTLeastSquares4D.from_noise(box, 3, np.eye(4)[None]*.3**2, [1.], order=21)
    left, right = geometry.initial_factors(2, seed=42, perturbation=.1)
    empirical = jnp.asarray(np.random.default_rng(91).normal(size=(9, 9))*.1)
    bases = geometry.penalty_bases(9)
    ml, mr = geometry.pair_moments
    direction = jnp.arange(9.)-4
    direction -= ml*jnp.dot(ml, direction)/jnp.dot(ml, ml)
    left = left.at[:, 0].add(3*direction)
    assert jnp.min((bases[0]@left)@(right@bases[1].T)) < 0
    solver = geometry.make_half_solver(strength=.001, negative_strength=1., penalty_bases=bases)
    def loss(l, r):
        a, b = geometry.terms(l, r, empirical)
        negative = jnp.minimum((bases[0]@l)@(r@bases[1].T), 0)
        return a+.001*b+jnp.sum(negative**2)+1e-10*jnp.sum((l.T@l)*(r@r.T))
    for _ in range(3):
        before = float(loss(left, right))
        fitted, new_right, residual = solver(left, right, empirical)
        assert float(loss(fitted, new_right)) < before
        np.testing.assert_allclose(ml@fitted@new_right@mr, 1., atol=1e-10)
        assert float(residual) < 1e-5
        left, right = fitted, new_right


def test_correlated_noise_empirical_term_matches_independent_joint_cubature():
    from scipy.stats import multivariate_normal
    box = np.tile([-1., 1.], (4, 1))
    covs = np.stack([np.eye(4)*.7**2, np.eye(4)*.9**2])
    covs[1, 0, 1] = covs[1, 1, 0] = .2
    covs[1, 2, 3] = covs[1, 3, 2] = -.2
    weights = np.array([.6, .4])
    geometry = SignedTTLeastSquares4D.from_noise(box, 3, covs, weights, order=21)
    left, right = geometry.initial_factors(2, seed=8, perturbation=.5)
    model = geometry.model(left, right)
    kernel = build_paired_gaussian_mixture_noise_kernel_tt(covs, weights, box, n_coeffs=129)
    operator = LinearTTConvolution.from_tt(model, kernel, np.tile([-2., 2.], (4, 1)))
    observations = np.array([[.3, -.1, .2, .5], [-.6, .7, -.4, .1]])
    empirical = geometry.empirical_term(operator, observations, batch=2)
    t, w = roots_legendre(17)
    points = np.stack(np.meshgrid(*([t]*4), indexing='ij'), axis=-1).reshape(-1, 4)
    qw = np.prod(np.stack(np.meshgrid(*([w]*4), indexing='ij'), axis=-1), axis=-1).ravel()
    density = np.asarray(model(jnp.asarray(points)))
    independent = [sum(a*np.dot(qw*density, multivariate_normal.pdf(y-points, cov=c))
                       for a, c in zip(weights, covs, strict=True)) for y in observations]
    np.testing.assert_allclose(jnp.sum(left*(empirical@right.T)), np.mean(independent), rtol=1e-9)


def test_full_batch_lbfgs_chunking_preserves_objective_and_updates():
    from ftde.full_batch_optimization import fit_noisy_tt_lbfgs
    from ftde.sobolev import AmplitudeSobolevPenalty
    box = np.array([[-2., 2.]])
    initial = init_tt_constant(box, rank=1, modes=5)
    kernel = build_diagonal_gaussian_noise_kernel_tt(jnp.array([.4]), jnp.asarray(box), n_coeffs=(129,))
    penalty = AmplitudeSobolevPenalty.from_tt(initial, order=2, length_scales=[.4])
    train = np.linspace(-.4, .7, 11)[:, None]
    valid = np.linspace(-.3, .6, 7)[:, None]
    runs = [fit_noisy_tt_lbfgs(initial, train, valid, kernel, penalty, iterations=4,
                             batch_size=b, checkpoint_every=1) for b in [4, 11]]
    for _, _, _, result in runs:
        assert result['evaluations'][-1]['value'] < result['evaluations'][0]['value']
        assert result['best_step'] > 0
    np.testing.assert_allclose(runs[0][1].cores[0].coeffs, runs[1][1].cores[0].coeffs, rtol=1e-8, atol=1e-10)


def test_noise_free_least_squares_matches_direct_integral_and_sample_average():
    box = np.array([[-1., 2.], [-2., 1.], [-1., 1.], [-3., 2.]])
    geometry = SignedTTLeastSquares4D.without_noise(box, 5)
    left, right = geometry.initial_factors(3, seed=7, perturbation=.3)
    model = geometry.model(left, right)
    points = np.random.default_rng(83).uniform(box[:, 0], box[:, 1], (23, 4))
    empirical = geometry.empirical_term_without_noise(points, batch=8)
    direct = float(squared_integral_tt(model)-2*jnp.mean(model(jnp.asarray(points))))
    np.testing.assert_allclose(geometry.terms(left, right, empirical)[0], direct, atol=1e-12)
    solver = geometry.make_half_solver(strength=.001, tolerance=1e-10)
    fitted, new_right, residual = solver(left, right, empirical)
    ml, mr = geometry.pair_moments
    np.testing.assert_allclose(ml@fitted@new_right@mr, 1., atol=1e-12)
    assert float(residual) < 1e-8
    a, b = geometry.terms(left, right, empirical)
    c, d = geometry.terms(fitted, new_right, empirical)
    assert float(c+.001*d) < float(a+.001*b)


def test_noise_free_lbfgs_improves_from_uniform_with_padding():
    from ftde.full_batch_optimization import fit_noisy_tt_lbfgs
    from ftde.sobolev import AmplitudeSobolevPenalty
    # Zero padding would extrapolate far outside this shifted box and overflow.
    offset = 1e8
    initial = init_tt_constant(np.array([[offset-2., offset+2.]]), modes=25, rank=1)
    penalty = AmplitudeSobolevPenalty.from_tt(initial, order=2, length_scales=[.2])
    _, _, _, result = fit_noisy_tt_lbfgs(initial, offset+np.linspace(-.3, .5, 19)[:, None],
        offset+np.linspace(-.25, .45, 11)[:, None], None, penalty, iterations=6, batch_size=8, checkpoint_every=1)
    assert result['best_step'] > 0
    assert result['best_nll'] < result['history'][0]['nll']
    assert result['evaluations'][-1]['value'] < result['evaluations'][0]['value']
