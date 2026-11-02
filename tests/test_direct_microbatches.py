import numpy as np
import pytest

from ftde.direct_training import (
    DirectTrainingProblem,
    NoisyDirectTrainingProblem,
    load_tt,
)
from ftde.gaussian_convolution import GaussianConvolution
from ftde.simple_initialization import init_tt_constant


@pytest.mark.parametrize('noisy', [False, True])
def test_accumulation_preserves_updates_with_curriculum_and_penalties(tmp_path, noisy):
    initial = init_tt_constant(np.tile([-2., 2.], (2, 1)), rank=3, modes=7, perturbation=.2)
    kwargs = {'regularizer': 'hessian', 'strength': .003, 'marginal_strength': .01}
    if noisy:
        problem = NoisyDirectTrainingProblem(initial, GaussianConvolution.from_tt(initial, [.2, .3]), **kwargs)
    else:
        problem = DirectTrainingProblem(initial, **kwargs)
    points = np.random.default_rng(41).normal(size=(48, 2))*.3
    options = {'steps': 6, 'batch_size': 16, 'checkpoint_every': 2,
               'mode_schedule': [(0, 3), (2, 7)], 'selection_start': 2}
    _, full = problem.fit_adam(points, points, tmp_path/'full', **options)
    _, chunks = problem.fit_adam(points, points, tmp_path/'chunks', microbatches=4, **options)
    assert full['best_step'] == chunks['best_step']
    np.testing.assert_allclose([h['validation_nll'] for h in full['history']],
                               [h['validation_nll'] for h in chunks['history']], atol=1e-11, rtol=1e-11)
    for a, b in zip(load_tt(tmp_path/'full'/'last.npz').cores,
                    load_tt(tmp_path/'chunks'/'last.npz').cores, strict=True):
        np.testing.assert_allclose(a.coeffs, b.coeffs, atol=1e-10, rtol=1e-10)


@pytest.mark.parametrize('value', [0, -1, 3, 1.5])
def test_invalid_chunk_count(tmp_path, value):
    problem = DirectTrainingProblem(init_tt_constant([[-1., 1.]], rank=1, modes=3))
    with pytest.raises(ValueError, match='microbatches'):
        problem.fit_adam(np.zeros((4, 1)), np.zeros((4, 1)), tmp_path, batch_size=8, microbatches=value)
