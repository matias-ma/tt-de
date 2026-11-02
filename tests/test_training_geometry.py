import jax
import jax.numpy as jnp
import numpy as np
import pytest

from ftde.direct_training import (
    DirectTrainingProblem,
    NoisyDirectTrainingProblem,
    load_tt,
)
from ftde.gaussian_convolution import GaussianConvolution
from ftde.mode_schedule import mask_modes, mode_masks, prepare_mode_schedule
from ftde.simple_initialization import init_tt_constant


@pytest.mark.parametrize('noisy', [False, True])
def test_independent_preconditioner_preserves_density_objective_and_physical_gradient(noisy):
    model = init_tt_constant(np.tile([-1., 1.], (2, 1)), rank=3, modes=7, perturbation=.5)
    forward = GaussianConvolution.from_tt(model, [.2, .3], quadrature_order=40)
    points = np.random.default_rng(4).uniform(-.8, .8, (13, 2))
    directions = tuple(jnp.asarray(np.random.default_rng(k).normal(size=c.coeffs.shape)) for k, c in enumerate(model.cores))
    logs, losses, derivatives = [], [], []
    for metric in [0., 1e-5, .01]:
        kwargs = {'strength': 1e-5, 'precondition_strength': metric, 'regularizer': 'hessian', 'marginal_strength': .001}
        problem = NoisyDirectTrainingProblem(model, forward, **kwargs) if noisy else DirectTrainingProblem(model, **kwargs)
        features = problem.features(points)
        logs.append(np.asarray(problem.logs(problem.params, features)))
        losses.append(float(problem.loss(problem.params, features)))
        def line(t, problem=problem, features=features):
            params = tuple(p+t*(v@f) for p, v, f in zip(problem.params, directions, problem.factors, strict=True))
            return problem.loss(params, features)
        derivatives.append(float(jax.grad(line)(0.)))
        for expected, actual in zip(model.cores, problem.model(problem.params).cores, strict=True):
            np.testing.assert_allclose(actual.coeffs, expected.coeffs, atol=1e-12)
    np.testing.assert_allclose(logs, np.broadcast_to(logs[0], np.shape(logs)), atol=1e-10)
    np.testing.assert_allclose(losses, losses[0], atol=1e-10)
    np.testing.assert_allclose(derivatives, derivatives[0], rtol=1e-9, atol=1e-10)


def test_mask_in_whitened_coordinates_is_a_physical_polynomial_cutoff():
    initial = init_tt_constant(np.tile([-2., 2.], (3, 1)), rank=4, modes=9, perturbation=.5)
    for beta in [0., .001, 1.]:
        problem = DirectTrainingProblem(initial, strength=1e-5, precondition_strength=beta)
        counts = (3, 5, 7)
        model = problem.model(mask_modes(problem.params, mode_masks(problem.sizes, counts)))
        for core, active in zip(model.cores, counts, strict=True):
            np.testing.assert_allclose(core.coeffs[..., active:], 0., atol=1e-13)
        assert problem.energies(mask_modes(problem.params, mode_masks(problem.sizes, counts)))[0] > 0


@pytest.mark.parametrize('noisy', [False, True])
def test_curriculum_blocks_then_learns_higher_modes(tmp_path, noisy):
    initial = init_tt_constant([[-2., 2.]], rank=1, modes=9)
    kwargs = {'strength': 1e-5, 'precondition_strength': .001, 'regularizer': 'hessian'}
    forward = GaussianConvolution.from_tt(initial, [.15], quadrature_order=40)
    problem = NoisyDirectTrainingProblem(initial, forward, **kwargs) if noisy else DirectTrainingProblem(initial, **kwargs)
    train = np.linspace(.1, .5, 60)[:, None]
    valid = np.linspace(.12, .48, 30)[:, None]
    _, record = problem.fit_adam(train, valid, tmp_path, steps=np.int64(40), batch_size=32,
        learning_rate=.02, mode_schedule=[(0, 3), (10, 9)], selection_start=np.int64(10), checkpoint_every=5)
    assert record['best_step'] >= 10
    assert record['best_nll'] < record['history'][0]['validation_nll']-.1
    for row in record['history']:
        assert row['inactive_coefficient_max'] < 1e-13
        assert row['active_modes'] == ([3] if row['step'] < 10 else [9])
        assert row['selection_eligible'] == (row['step'] >= 10)
    final = load_tt(tmp_path/'last.npz')
    assert np.linalg.norm(np.asarray(final.cores[0].coeffs)[..., 3:]) > 1e-4


def test_full_mode_schedule_matches_unrestricted_training(tmp_path):
    initial = init_tt_constant([[-2., 2.]], rank=1, modes=7)
    p = DirectTrainingProblem(initial, precondition_strength=.01)
    train = np.linspace(-.5, .5, 32)[:, None]
    options = {'steps': 10, 'batch_size': 16, 'checkpoint_every': 5}
    a, _ = p.fit_adam(train, train, tmp_path/'a', **options)
    b, _ = p.fit_adam(train, train, tmp_path/'b', mode_schedule=[(0, 7)], **options)
    np.testing.assert_allclose(a.cores[0].coeffs, b.cores[0].coeffs, rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize('schedule', [[], [(1, 7)], [(0, 3)], [(0, 3), (4, 2), (8, 7)],
                                     [(0, 3), (0, 7)], [(0, 0), (4, 7)], [(0, 3), (10, 7)],
                                     [(0, 3.5), (4, 7)], [(0, (3, 3)), (4, 7)]])
def test_invalid_schedules(schedule):
    with pytest.raises((ValueError, TypeError)):
        prepare_mode_schedule((7,), 10, schedule)


@pytest.mark.parametrize('beta', [-1., np.nan, np.inf])
def test_invalid_preconditioning_strength(beta):
    initial = init_tt_constant([[-1., 1.]], rank=1, modes=3)
    with pytest.raises(ValueError):
        DirectTrainingProblem(initial, precondition_strength=beta)
