import jax
import jax.numpy as jnp
import numpy as np
from scipy.special import roots_legendre

from ftde.direct_training import (
    DirectTrainingProblem,
    conditional_log_densities,
    fourth_integral_4d,
)
from ftde.simple_initialization import init_tt_constant
from ftde.tt_density import log_density_tt


def test_direct_features_and_normalization_match_original_tt():
    box = np.array([[-2., 1.], [.3, 2.], [-1., 1.], [-2., 2.]])
    initial = init_tt_constant(box, rank=3, modes=5, perturbation=.5)
    points = np.random.default_rng(2).uniform(box[:, 0], box[:, 1], (31, 4))
    for penalty in ('h1', 'axis_h2', 'hessian'):
        problem = DirectTrainingProblem(initial, regularizer=penalty, strength=.003)
        np.testing.assert_allclose(problem.logs(problem.params, problem.features(points)),
                                   log_density_tt(initial, jnp.asarray(points)), rtol=1e-10, atol=1e-11)


def test_full_hessian_matches_independent_autodiff_cubature():
    box = np.array([[-1., 2.], [-2., 1.]])
    initial = init_tt_constant(box, rank=3, modes=4, perturbation=1.)
    scales = np.array([.3, .2])
    problem = DirectTrainingProblem(initial, regularizer='hessian', length_scales=scales)
    t, w = roots_legendre(7)
    nodes = [(a+b)/2+(b-a)*t/2 for a, b in box]
    points = np.stack(np.meshgrid(*nodes, indexing='ij'), axis=-1).reshape(-1, 2)
    weights = np.outer(w*1.5, w*1.5).ravel()
    hessians = np.asarray(jax.vmap(jax.hessian(lambda x: initial(x[None])[0]))(jnp.asarray(points)))
    expected = np.dot(weights, np.sum((hessians*scales[None, :, None]*scales[None, None, :])**2, axis=(1, 2)))
    z, _, axis, full = problem.energies(problem.params)
    np.testing.assert_allclose(full, expected, rtol=1e-10, atol=1e-12)
    assert full > axis and z > 0


def test_conditional_density_integrates_to_one_and_has_correct_ratio():
    box = np.array([[-2., 2.], [-1., 2.]])
    initial = init_tt_constant(box, rank=3, modes=5, perturbation=.8)
    problem = DirectTrainingProblem(initial, objective='composite')
    t, w = roots_legendre(15)
    points = np.column_stack([2*t, np.full(len(t), .3)])
    features = problem.features(points)
    logs = np.asarray(conditional_log_densities(problem.params, features, problem.grams))[:, 0]
    np.testing.assert_allclose(np.dot(2*w, np.exp(logs)), 1., atol=2e-12)
    joint = np.asarray(initial(jnp.asarray(points)))**2
    np.testing.assert_allclose(np.exp(logs), joint/np.dot(2*w, joint), rtol=1e-11)


def test_quartic_integral_and_loss_gradient_match_independent_grid():
    box = np.tile([-1., 1.], (4, 1))
    initial = init_tt_constant(box, rank=2, modes=3, perturbation=1.)
    problem = DirectTrainingProblem(initial, objective='l2', regularizer='hessian')
    t, w = roots_legendre(7)
    points = np.stack(np.meshgrid(*([t]*4), indexing='ij'), axis=-1).reshape(-1, 4)
    weights = np.prod(np.stack(np.meshgrid(*([w]*4), indexing='ij'), axis=-1), axis=-1).ravel()
    expected = np.dot(weights, np.asarray(initial(jnp.asarray(points)))**4)
    actual = fourth_integral_4d(problem.params, problem.quad_bases, problem.quad_weights)
    np.testing.assert_allclose(actual, expected, rtol=2e-12)
    data = np.random.default_rng(1).uniform(-1, 1, (19, 4))
    features = problem.features(data)
    direction = jnp.asarray(np.random.default_rng(3).normal(size=problem.params[1].shape))
    def line(t):
        p = list(problem.params)
        p[1] = p[1]+t*direction
        return problem.loss(tuple(p), features)
    np.testing.assert_allclose(jax.grad(line)(0.), (line(1e-5)-line(-1e-5))/2e-5, rtol=2e-6, atol=1e-9)


def test_training_learns_from_elementary_start(tmp_path):
    initial = init_tt_constant(np.array([[-2., 2.]]), rank=1, modes=7)
    problem = DirectTrainingProblem(initial, regularizer='h1')
    train, valid = np.linspace(-.2, .4, 50)[:, None], np.linspace(-.15, .35, 25)[:, None]
    _, result = problem.fit_adam(train, valid, tmp_path, steps=40, batch_size=32,
                                learning_rate=.03, checkpoint_every=10)
    assert result['best_step'] > 0
    assert result['best_nll'] < result['history'][0]['validation_nll']-.1


def test_full_batch_lbfgs_learns_and_is_chunk_invariant(tmp_path):
    from ftde.direct_lbfgs import fit_lbfgs
    initial = init_tt_constant(np.array([[3., 7.]]), rank=1, modes=7)
    problem = DirectTrainingProblem(initial, objective='logpen', regularizer='hessian', strength=.01)
    train, valid = np.linspace(4.7, 5.3, 43)[:, None], np.linspace(4.75, 5.25, 19)[:, None]
    a, ra = fit_lbfgs(problem, train, valid, tmp_path/'a', iterations=20, batch_size=16, checkpoint_every=5)
    b, rb = fit_lbfgs(problem, train, valid, tmp_path/'b', iterations=20, batch_size=43, checkpoint_every=5)
    assert ra['best_nll'] < ra['history'][0]['validation_nll']-.5
    np.testing.assert_allclose(ra['best_nll'], rb['best_nll'], atol=1e-7)
    np.testing.assert_allclose(a.cores[0].coeffs, b.cores[0].coeffs, rtol=1e-6, atol=1e-7)


def test_noisy_direct_objective_matches_fixed_convolution():
    from ftde.direct_training import NoisyDirectTrainingProblem
    from ftde.gaussian_convolution import GaussianConvolution
    initial = init_tt_constant(np.tile([-1., 1.], (2, 1)), rank=2, modes=5, perturbation=.2)
    forward = GaussianConvolution.from_tt(initial, [.2, .3], quadrature_order=40)
    problem = NoisyDirectTrainingProblem(initial, forward, regularizer='hessian', strength=.01)
    data = np.array([[.1, .2], [-.3, .1], [1.1, -.3]])
    logs = problem.logs(problem.params, problem.features(data))
    np.testing.assert_allclose(logs, forward.log_density(initial, data), atol=1e-10)
    e = problem.energies(problem.params)
    np.testing.assert_allclose(problem.loss(problem.params, problem.features(data)), -logs.mean()+.01*e[3]/e[0], atol=1e-10)
    gradients = jax.grad(problem.loss)(problem.params, problem.features(data))
    assert all(np.isfinite(g).all() for g in gradients)
