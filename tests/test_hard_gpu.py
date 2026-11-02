"""Harder correctness tests for the noise-free squared-TT/Tucker estimator.

These use higher dimension, rank, and Chebyshev degree than the quick
regression suite in test_density.py, and are meant to be run on a GPU
(e.g. on a cluster), not on a CPU-only laptop.

--- How to run ---

See the README's "Install" section for the pip install recipe (order
matters: chebdCUR, then this repo, with any CUDA jax build installed last).
Then, with the GPU visible to jax:

    pytest tests/test_hard_gpu.py -v -s    # full-difficulty settings, the default

Each test prints its own diagnostics (log-likelihood error, ranks used,
timing-relevant sizes) via ``-s`` before asserting, so a failure is easy to
diagnose even without re-running.

--- Local smoke check (CPU, tiny sizes, just to catch bugs) ---

    FTDE_TEST_SCALE=smoke pytest tests/test_hard_gpu.py -v -s

This runs the exact same test bodies at drastically reduced dimension/rank/
degree/steps so they finish in seconds on a CPU-only machine. It is NOT a
substitute for the real (full-scale) run -- it only checks the code path is
exception-free and qualitatively sane; accuracy assertions are loosened
accordingly at this scale.
"""

from __future__ import annotations

import os

import jax
import jax.numpy as jnp
from conftest import report, run_all_tests

from ftde.data import sample_correlated_gaussian
from ftde.init import init_tt_from_kde, init_tucker_from_kde
from ftde.tt_density import fit_tt, log_density_tt, squared_integral_tt
from ftde.tucker_density import fit_tucker, log_density_tucker, squared_integral_tucker

SCALE = os.environ.get("FTDE_TEST_SCALE", "full")
IS_SMOKE = SCALE == "smoke"


def _cfg(smoke_value, full_value):
    return smoke_value if IS_SMOKE else full_value


def _true_log_density_gaussian(x: jnp.ndarray, cov: jnp.ndarray) -> jnp.ndarray:
    d = cov.shape[0]
    inv_cov = jnp.linalg.inv(cov)
    log_det = jnp.linalg.slogdet(cov)[1]
    quad = jnp.einsum("...i,ij,...j->...", x, inv_cov, x)
    return -0.5 * (d * jnp.log(2 * jnp.pi) + log_det + quad)


def _true_log_density_iso_mixture(
    x: jnp.ndarray, means: jnp.ndarray, weights: jnp.ndarray
) -> jnp.ndarray:
    # Unit-variance isotropic Gaussian components (matches sample_gaussian_mixture).
    d = means.shape[1]
    diff = x[:, None, :] - means[None, :, :]
    log_comp = -0.5 * (d * jnp.log(2 * jnp.pi) + jnp.sum(diff**2, axis=-1))
    log_comp = log_comp + jnp.log(weights)[None, :]
    m = jnp.max(log_comp, axis=-1, keepdims=True)
    return (m + jnp.log(jnp.sum(jnp.exp(log_comp - m), axis=-1, keepdims=True)))[:, 0]


def test_hard_correlated_gaussian_high_dim():
    """A genuinely high-dimensional, only-approximately-low-rank Gaussian."""
    d = _cfg(3, 10)
    rho = 0.6
    n_train = _cfg(300, 20_000)
    n_test = _cfg(200, 4_000)
    max_rank = _cfg(3, 14)
    n_coeffs = _cfg((9, 17), (17, 33, 65, 129))
    n_starts = _cfg(16, 256)
    gd_steps = _cfg(5, 100)
    n_steps = _cfg(50, 4000)
    batch_size = _cfg(None, 1024)

    key = jax.random.PRNGKey(0)
    k1, k2 = jax.random.split(key)
    data = sample_correlated_gaussian(k1, n=n_train, d=d, rho=rho)

    init = init_tt_from_kde(
        data, max_rank=max_rank, n_coeffs=n_coeffs, n_starts=n_starts, gd_steps=gd_steps
    )

    fitted, history = fit_tt(
        init, data, n_steps=n_steps, lr=3e-3, batch_size=batch_size, log_every=n_steps // 5 or 1
    )

    cov = jnp.full((d, d), rho) + (1.0 - rho) * jnp.eye(d)
    test = sample_correlated_gaussian(k2, n=n_test, d=d, rho=rho)
    est_logp = log_density_tt(fitted, test)
    true_logp = _true_log_density_gaussian(test, cov)
    mean_abs_err = float(jnp.mean(jnp.abs(est_logp - true_logp)))
    z = float(squared_integral_tt(fitted))

    report(
        "hard_correlated_gaussian_high_dim",
        d=d, function=f"correlated Gaussian rho={rho}", method="TT", scale=SCALE,
        rank=fitted.ranks, initial_nll=round(history[0], 4), final_nll=round(history[-1], 4),
        mean_true_logp=round(float(jnp.mean(true_logp)), 4),
        mean_est_logp=round(float(jnp.mean(est_logp)), 4),
        mean_abs_logp_error=round(mean_abs_err, 4), z=round(z, 4),
    )

    assert jnp.isfinite(jnp.array(history[-1]))
    assert history[-1] < history[0] + 1e-6
    assert jnp.isfinite(z) and z > 0.0
    assert mean_abs_err < _cfg(2.0, 0.25)


def test_hard_gaussian_mixture_moderate_dim():
    """A multi-modal target -- much harder for a single TT than a unimodal one."""
    d = _cfg(2, 6)
    n_components = _cfg(2, 5)
    spread = 3.0
    n_train = _cfg(300, 20_000)
    n_test = _cfg(200, 4_000)
    max_rank = _cfg(3, 16)
    n_coeffs = _cfg((9, 17), (17, 33, 65, 129))
    n_starts = _cfg(16, 256)
    gd_steps = _cfg(5, 100)
    n_steps = _cfg(50, 5000)
    batch_size = _cfg(None, 1024)

    key = jax.random.PRNGKey(1)
    k1, k2, k3, k4, k5 = jax.random.split(key, 5)
    means = jax.random.uniform(k1, (n_components, d), minval=-spread, maxval=spread, dtype=jnp.float64)
    weights = jax.random.dirichlet(k2, jnp.ones(n_components), dtype=jnp.float64)
    comps = jax.random.categorical(k3, jnp.log(weights), shape=(n_train,))
    data = means[comps] + jax.random.normal(k4, (n_train, d), dtype=jnp.float64)

    init = init_tt_from_kde(
        data, max_rank=max_rank, n_coeffs=n_coeffs, n_starts=n_starts, gd_steps=gd_steps
    )

    fitted, history = fit_tt(
        init, data, n_steps=n_steps, lr=3e-3, batch_size=batch_size, log_every=n_steps // 5 or 1
    )

    comps_test = jax.random.categorical(k5, jnp.log(weights), shape=(n_test,))
    test = means[comps_test] + jax.random.normal(jax.random.PRNGKey(123), (n_test, d), dtype=jnp.float64)
    est_logp = log_density_tt(fitted, test)
    true_logp = _true_log_density_iso_mixture(test, means, weights)
    mean_abs_err = float(jnp.mean(jnp.abs(est_logp - true_logp)))
    z = float(squared_integral_tt(fitted))

    report(
        "hard_gaussian_mixture_moderate_dim",
        d=d, function=f"{n_components}-component isotropic Gaussian mixture", method="TT", scale=SCALE,
        rank=fitted.ranks, initial_nll=round(history[0], 4), final_nll=round(history[-1], 4),
        mean_abs_logp_error=round(mean_abs_err, 4), z=round(z, 4),
    )

    assert jnp.isfinite(jnp.array(history[-1]))
    assert history[-1] < history[0] + 1e-6
    assert jnp.isfinite(z) and z > 0.0
    assert mean_abs_err < _cfg(2.0, 0.5)


def test_hard_tucker_moderate_dim():
    """Same idea, Tucker representation. Rank kept modest since Tucker core
    size scales as rank**d (see chebtuckerge's max_core_entries guard)."""
    d = _cfg(2, 8)
    rho = 0.5
    n_train = _cfg(300, 20_000)
    n_test = _cfg(200, 4_000)
    max_rank = _cfg(3, 4)
    n_coeffs = _cfg((9, 17), (17, 33, 65))
    n_starts = _cfg(16, 256)
    gd_steps = _cfg(5, 100)
    n_steps = _cfg(50, 4000)
    batch_size = _cfg(None, 1024)

    key = jax.random.PRNGKey(2)
    k1, k2 = jax.random.split(key)
    data = sample_correlated_gaussian(k1, n=n_train, d=d, rho=rho)

    init = init_tucker_from_kde(
        data, max_rank=max_rank, n_coeffs=n_coeffs, n_starts=n_starts, gd_steps=gd_steps
    )

    fitted, history = fit_tucker(
        init, data, n_steps=n_steps, lr=3e-3, batch_size=batch_size, log_every=n_steps // 5 or 1
    )

    cov = jnp.full((d, d), rho) + (1.0 - rho) * jnp.eye(d)
    test = sample_correlated_gaussian(k2, n=n_test, d=d, rho=rho)
    est_logp = log_density_tucker(fitted, test)
    true_logp = _true_log_density_gaussian(test, cov)
    mean_abs_err = float(jnp.mean(jnp.abs(est_logp - true_logp)))
    z = float(squared_integral_tucker(fitted))

    report(
        "hard_tucker_moderate_dim",
        d=d, function=f"correlated Gaussian rho={rho}", method="Tucker", scale=SCALE,
        rank=fitted.ranks, core_size=int(jnp.prod(jnp.array(fitted.ranks))),
        initial_nll=round(history[0], 4), final_nll=round(history[-1], 4),
        mean_abs_logp_error=round(mean_abs_err, 4), z=round(z, 4),
    )

    assert jnp.isfinite(jnp.array(history[-1]))
    assert history[-1] < history[0] + 1e-6
    assert jnp.isfinite(z) and z > 0.0
    assert mean_abs_err < _cfg(2.0, 0.3)


def test_hard_independent_product_control():
    """Positive control: an exactly-separable target in higher dimension.

    Each axis independently follows a mixture of two 1D Gaussians. Since the
    joint density factorizes exactly, a rank-1 TT should already fit this
    almost exactly after training -- this isolates whether errors seen in the
    other (non-separable) hard tests come from genuine rank limitations
    rather than from a bug in the Gram/normalization machinery at higher d.
    """
    d = _cfg(3, 10)
    n_train = _cfg(300, 20_000)
    n_test = _cfg(200, 4_000)
    max_rank = _cfg(2, 2)
    n_coeffs = _cfg((9, 17), (17, 33, 65))
    n_starts = _cfg(16, 128)
    gd_steps = _cfg(5, 50)
    n_steps = _cfg(50, 3000)
    batch_size = _cfg(None, 1024)

    def sample_axis(key, n):
        k1, k2 = jax.random.split(key, 2)
        comp = jax.random.bernoulli(k1, 0.5, shape=(n,))
        base = jax.random.normal(k2, (n,), dtype=jnp.float64)
        return jnp.where(comp, base - 2.0, base + 2.0)

    def true_log_density_1d(x):
        log_a = -0.5 * jnp.log(2 * jnp.pi) - 0.5 * (x + 2.0) ** 2 - jnp.log(2.0)
        log_b = -0.5 * jnp.log(2 * jnp.pi) - 0.5 * (x - 2.0) ** 2 - jnp.log(2.0)
        m = jnp.maximum(log_a, log_b)
        return m + jnp.log(jnp.exp(log_a - m) + jnp.exp(log_b - m))

    key = jax.random.PRNGKey(3)
    keys_train = jax.random.split(key, d)
    data = jnp.stack([sample_axis(k, n_train) for k in keys_train], axis=1)

    init = init_tt_from_kde(
        data, max_rank=max_rank, n_coeffs=n_coeffs, n_starts=n_starts, gd_steps=gd_steps
    )
    fitted, history = fit_tt(
        init, data, n_steps=n_steps, lr=3e-3, batch_size=batch_size, log_every=n_steps // 5 or 1
    )

    keys_test = jax.random.split(jax.random.PRNGKey(30), d)
    test = jnp.stack([sample_axis(k, n_test) for k in keys_test], axis=1)
    est_logp = log_density_tt(fitted, test)
    true_logp = jnp.sum(jax.vmap(true_log_density_1d, in_axes=1)(test), axis=0)
    mean_abs_err = float(jnp.mean(jnp.abs(est_logp - true_logp)))

    report(
        "hard_independent_product_control",
        d=d, function="separable 2-component Gaussian mixture per axis", method="TT", scale=SCALE,
        rank=fitted.ranks, initial_nll=round(history[0], 4), final_nll=round(history[-1], 4),
        mean_abs_logp_error=round(mean_abs_err, 4),
    )

    assert mean_abs_err < _cfg(2.0, 0.15)


if __name__ == "__main__":
    run_all_tests(globals())
