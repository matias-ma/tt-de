import jax
import jax.numpy as jnp
import numpy as np
from scipy.special import roots_legendre

from ftde.direct_training import DirectTrainingProblem
from ftde.kernel_score import FourierKernelScore, gaussian_mixture_characteristic
from ftde.simple_initialization import init_tt_constant


def test_characteristic_matches_independent_cubature_and_refinement():
    model = init_tt_constant(np.array([[-2., 1.], [-1., 3.]]), rank=3, modes=5, perturbation=.8)
    problem = DirectTrainingProblem(model)
    frequencies = np.array([[0., 0.], [1.3, -.8], [-1.3, .8], [12., -8.]])
    score = FourierKernelScore(problem, frequencies)
    actual = score.predict(problem.params)
    t, w = roots_legendre(100)
    points = np.stack(np.meshgrid(1.5*t-.5, 2*t+1, indexing='ij'), axis=-1).reshape(-1, 2)
    weights = np.outer(1.5*w, 2*w).ravel()
    density = np.asarray(model(jnp.asarray(points)))**2
    expected = (weights*density)@np.exp(1j*points@frequencies.T)
    np.testing.assert_allclose(actual, expected, atol=2e-12)
    np.testing.assert_allclose(actual[0], 1., atol=1e-12)
    np.testing.assert_allclose(actual[1], np.conj(actual[2]), atol=1e-12)
    np.testing.assert_allclose(actual, FourierKernelScore(problem, frequencies, order_multiplier=2).predict(problem.params), atol=2e-12)


def test_score_gradient_and_observation_identity():
    initial = init_tt_constant(np.array([[-2., 2.]]), rank=1, modes=6, perturbation=.4)
    problem = DirectTrainingProblem(initial)
    frequencies = np.array([[.3], [1.2], [3.5]])
    score = FourierKernelScore(problem, frequencies)
    data = np.array([[-.4], [.2], [.7]])
    target = jnp.asarray(np.exp(1j*data@frequencies.T).mean(axis=0))
    indices = jnp.arange(3)
    direction = jnp.asarray(np.random.default_rng(3).normal(size=problem.params[0].shape))
    def line(t):
        return score.discrepancy((problem.params[0]+t*direction,), target, indices)
    np.testing.assert_allclose(jax.grad(line)(0.), (line(1e-5)-line(-1e-5))/2e-5, atol=1e-8)
    expected = score.observation_scores(score.predict(problem.params), data).mean()+np.mean(np.abs(target)**2)
    np.testing.assert_allclose(line(0.), expected, atol=1e-12)


def test_noise_characteristic_is_forward_multiplication():
    model = init_tt_constant(np.tile([-1., 1.], (2, 1)), rank=2, modes=4)
    problem = DirectTrainingProblem(model)
    frequencies = np.array([[0., 0.], [.3, -.8], [2., 3.]])
    covariance = np.array([[[.5, .2], [.2, .7]], [[1., 0.], [0., 2.]]])
    weights = np.array([.7, .3])
    noise = gaussian_mixture_characteristic(frequencies, weights, covariance)
    expected = sum(w*np.exp(-.5*np.sum((frequencies@c)*frequencies, axis=1)) for w, c in zip(weights, covariance))
    np.testing.assert_allclose(noise, expected)
    clean = FourierKernelScore(problem, frequencies).predict(problem.params)
    noisy = FourierKernelScore(problem, frequencies, noise_characteristic=noise).predict(problem.params)
    np.testing.assert_allclose(noisy, clean*noise)


def test_kernel_objective_learns_from_constant_start(tmp_path):
    from ftde.kernel_score import draw_frequencies
    from ftde.scored_training import fit_scored
    initial = init_tt_constant(np.array([[-2., 2.]]), rank=1, modes=7)
    problem = DirectTrainingProblem(initial, strength=.0001, regularizer='hessian')
    train = np.linspace(-.3, .5, 80)[:, None]
    valid = np.linspace(-.25, .45, 40)[:, None]
    result = fit_scored(problem, train, valid, tmp_path, draw_frequencies(train, 128, 1),
                        draw_frequencies(train, 64, 2), noise_weights=np.ones(1),
                        noise_covariances=np.zeros((1, 1, 1)), objective='mmd',
                        steps=80, batch_size=32, learning_rate=.03, checkpoint_every=20)
    assert result['kernel_step'] > 0
    assert result['best_kernel'] < result['history'][0]['validation_mmd']/2
    assert (tmp_path/'fit.npz').exists() and (tmp_path/'kernel_fit.npz').exists()
