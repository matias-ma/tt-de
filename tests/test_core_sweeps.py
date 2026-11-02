import jax
import jax.numpy as jnp
import numpy as np

from ftde.core_sweeps import (
    fit_core_sweeps,
    local_quadratic_terms,
    quadratic_action,
    shift_center,
)
from ftde.direct_training import integrated_energies
from ftde.simple_initialization import init_tt_constant


def test_local_quadratic_matches_global_energy_and_gradient():
    rng = np.random.default_rng(4)
    cores = [jnp.asarray(rng.normal(size=s)) for s in [(1, 2, 4), (2, 3, 4), (3, 1, 4)]]
    first, second = [], []
    for _ in cores:
        a, b = rng.normal(size=(4, 4)), rng.normal(size=(4, 4))
        first.append(jnp.asarray(a@a.T)); second.append(jnp.asarray(b@b.T))
    for reg, p in [('h1', 1), ('axis_h2', 2), ('hessian', 3)]:
        for k in range(3):
            terms = local_quadratic_terms(cores, first, second, k, .03, reg)
            def global_energy(c, k=k, p=p):
                trial = list(cores); trial[k] = c
                e = integrated_energies(trial, [jnp.eye(4)]*3, first, second)
                return e[0]+.03*e[p]
            np.testing.assert_allclose(jnp.sum(cores[k]*quadratic_action(cores[k], terms)), global_energy(cores[k]), rtol=1e-12)
            np.testing.assert_allclose(2*quadratic_action(cores[k], terms), jax.grad(global_energy)(cores[k]), rtol=1e-11, atol=1e-9)


def test_gauge_changes_preserve_coefficient_tensor():
    rng = np.random.default_rng(5)
    cores = [jnp.asarray(rng.normal(size=s)) for s in [(1, 2, 4), (2, 3, 4), (3, 1, 4)]]
    def tensor():
        return np.einsum('ian,abm,bjl->nml', *map(np.asarray, cores))
    original = tensor()
    shift_center(cores, 2, -1); shift_center(cores, 1, -1)
    shift_center(cores, 0, 1); shift_center(cores, 1, 1)
    np.testing.assert_allclose(tensor(), original, atol=1e-12)


def test_sweeps_learn_and_monotonically_reduce_objective(tmp_path):
    initial = init_tt_constant(np.tile([-2., 2.], (2, 1)), rank=3, modes=6)
    rng = np.random.default_rng(8)
    train = rng.normal(size=(160, 2))*.3
    valid = rng.normal(size=(80, 2))*.3
    _, result = fit_core_sweeps(initial, train, valid, tmp_path, sweeps=3, newton_steps=4)
    assert result['best_step'] > 0
    assert result['best_nll'] < result['history'][0]['validation_nll']-1.
    assert np.all(np.diff([r['training_objective'] for r in result['history']]) <= 1e-8)
