import numpy as np
import pytest

from ftde.direct_training import DirectTrainingProblem
from ftde.simple_initialization import init_tt_constant
from ftde.tt_density import squared_integral_tt


def test_smoother_random_starts_reduce_function_roughness_without_changing_capacity():
    box = np.tile([-2., 2.], (4, 1))
    options = ({}, {'spectral_decay': 2.}, {'spectral_scale': 4.})
    models = [init_tt_constant(box, rank=8, modes=25, seed=0, **kw) for kw in options]
    shapes = [c.coeffs.shape for c in models[0].cores]
    energies = []
    for model in models:
        assert [c.coeffs.shape for c in model.cores] == shapes
        np.testing.assert_allclose(squared_integral_tt(model), 1., atol=1e-12)
        problem = DirectTrainingProblem(model, strength=0.)
        e = np.asarray(problem.energies(problem.params))
        energies.append(e[3]/e[0])
        assert e[1] > 0  # Still random, not an exactly constant rank-one start.
    assert energies[1] < energies[0]/10
    assert energies[2] < energies[0]/10
    repeated = init_tt_constant(box, rank=8, modes=25, spectral_scale=4., seed=0)
    for a, b in zip(repeated.cores, models[2].cores, strict=True):
        np.testing.assert_array_equal(a.coeffs, b.coeffs)


@pytest.mark.parametrize('options', [{'spectral_decay': -1}, {'spectral_decay': np.nan},
                                     {'spectral_scale': 0}, {'spectral_scale': np.inf}])
def test_invalid_spectral_initialization(options):
    with pytest.raises(ValueError):
        init_tt_constant([[-1., 1.]], **options)
