"""Regression tests for the noise-free squared-TT/Tucker density estimator.

Deliberately at d >= 3: a tensor *train* (or Tucker decomposition) only has a
nontrivial chain of bond dimensions / multi-way core once there are at least
two internal bonds (three cores), so d=1 (no bonds at all) or d=2 (exactly
one bond) don't really exercise the structure -- they're closer to testing a
single matrix factorization. Most tests are d=3 and cheap; two are
deliberately pushed to d=5 with a harder (multi-modal) target to check the
method still behaves at moderately higher dimension without a memory
blow-up -- if an easy d>=3/5 example like this ever does blow up memory on a
normal machine, that is a signal of a real bug, not just "needs a bigger
machine" (see todo.txt).
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
from conftest import report, run_all_tests

from ftde.data import sample_correlated_gaussian
from ftde.init import init_tt_from_kde, init_tucker_from_kde
from ftde.tt_density import density_tt, fit_tt, log_density_tt, squared_integral_tt
from ftde.tucker_density import (
    density_tucker,
    fit_tucker,
    log_density_tucker,
    squared_integral_tucker,
)


def _true_log_density_gaussian(x, cov):
    d = cov.shape[0]
    inv_cov = jnp.linalg.inv(cov)
    log_det = jnp.linalg.slogdet(cov)[1]
    quad = jnp.einsum("...i,ij,...j->...", x, inv_cov, x)
    return -0.5 * (d * jnp.log(2 * jnp.pi) + log_det + quad)


def _true_log_density_iso_mixture(x, means, weights):
    d = means.shape[1]
    diff = x[:, None, :] - means[None, :, :]
    log_comp = -0.5 * (d * jnp.log(2 * jnp.pi) + jnp.sum(diff**2, axis=-1))
    log_comp = log_comp + jnp.log(weights)[None, :]
    m = jnp.max(log_comp, axis=-1, keepdims=True)
    return (m + jnp.log(jnp.sum(jnp.exp(log_comp - m), axis=-1, keepdims=True)))[:, 0]


def _importance_sampled_mass(log_density_fn, sample_fn, key, n_points, true_log_density_fn):
    """Estimate integral p_tilde(x) dx via self-normalized importance
    sampling against the known true density used to draw the samples --
    scales to any dimension, unlike a dense evaluation grid."""
    x = sample_fn(key, n_points)
    log_ratio = log_density_fn(x) - true_log_density_fn(x)
    return float(jnp.mean(jnp.exp(log_ratio)))


def test_squared_integral_is_exact_on_a_known_three_mode_core():
    # A 3-mode (2 internal bonds), rank-1 ChebTT of three constants c1, c2, c3
    # on boxes [a_i, b_i] represents g(x) = c1*c2*c3 (constant), so
    # integral g(x)^2 dx = (c1*c2*c3)^2 * volume([a,b]^3) -- hand-verifiable.
    from chebttge.chebtt import ChebTT
    from chebttge.cores import ChebyshevCoeffCore

    boxes = [(-2.0, 3.0), (-1.0, 1.5), (0.0, 4.0)]
    consts = [1.7, -0.9, 1.2]
    cores = tuple(ChebyshevCoeffCore(jnp.array([[[c]]]), domain=box) for c, box in zip(consts, boxes))
    domain = jnp.array(boxes)
    tt = ChebTT(cores=cores, domain=domain, ranks=(1, 1))

    z = squared_integral_tt(tt)
    volume = 1.0
    for a, b in boxes:
        volume *= (b - a)
    expected = (consts[0] * consts[1] * consts[2]) ** 2 * volume

    report(
        "squared_integral_exact_3_mode_tt",
        d=3, function="product of 3 constants", method="TT", ranks=tt.ranks,
        z=float(z), expected=float(expected), abs_error=float(abs(z - expected)),
    )
    assert jnp.allclose(z, expected)


def test_squared_integral_is_exact_on_a_known_three_mode_core_tucker():
    # Tucker analogue: core=1, three rank-1 factors each holding a single
    # constant c_i, so g(x) = c1*c2*c3 -- same hand-verifiable formula.
    from chebtuckerge.chebtucker import ChebTucker
    from chebtuckerge.factors import ChebyshevCoeffFactor

    boxes = [(-2.0, 3.0), (-1.0, 1.5), (0.0, 4.0)]
    consts = [1.7, -0.9, 1.2]
    factors = tuple(ChebyshevCoeffFactor(jnp.array([[c]]), domain=box) for c, box in zip(consts, boxes))
    core = jnp.ones((1, 1, 1))
    domain = jnp.array(boxes)
    t = ChebTucker(core=core, factors=factors, domain=domain)

    z = squared_integral_tucker(t)
    volume = 1.0
    for a, b in boxes:
        volume *= (b - a)
    expected = (consts[0] * consts[1] * consts[2]) ** 2 * volume

    report(
        "squared_integral_exact_3_mode_tucker",
        d=3, function="product of 3 constants", method="Tucker", ranks=t.ranks,
        z=float(z), expected=float(expected), abs_error=float(abs(z - expected)),
    )
    assert jnp.allclose(z, expected)


def test_fit_recovers_correlated_gaussian_log_likelihood():
    key = jax.random.PRNGKey(0)
    k1, k2, k3 = jax.random.split(key, 3)
    d, rho = 3, 0.7
    n_train, n_test, n_mass = 2000, 800, 4000

    data = sample_correlated_gaussian(k1, n=n_train, d=d, rho=rho)
    cov = jnp.full((d, d), rho) + (1.0 - rho) * jnp.eye(d)

    init = init_tt_from_kde(data, max_rank=8, n_coeffs=(17, 33), gd_steps=10, n_starts=32)
    # NOTE: lr=5e-3 (fine at d=2) overshoots into a worse basin at this rank/d
    # -- NLL briefly jumps up before settling, and ends up worse than the
    # init. lr=1e-3 fixes this (monotonic decrease, ~2x better recovery
    # error). Diagnosed by comparing loss trajectories, not just accepted.
    fitted, history = fit_tt(init, data, n_steps=1000, lr=1e-3, log_every=0)

    test = sample_correlated_gaussian(k2, n=n_test, d=d, rho=rho)
    est_logp = log_density_tt(fitted, test)
    true_logp = _true_log_density_gaussian(test, cov)
    mean_abs_error = float(jnp.mean(jnp.abs(est_logp - true_logp)))

    min_density = float(jnp.min(density_tt(fitted, test)))
    mass = _importance_sampled_mass(
        lambda x: log_density_tt(fitted, x),
        lambda k, n: sample_correlated_gaussian(k, n=n, d=d, rho=rho),
        k3, n_mass, lambda x: _true_log_density_gaussian(x, cov),
    )

    report(
        "fit_correlated_gaussian_3d",
        d=d, function=f"correlated Gaussian (rho={rho})", method="TT",
        n_train=n_train, n_test=n_test, n_mass_samples=n_mass,
        ranks=fitted.ranks, degree=fitted.cores[0].coeffs.shape[-1],
        initial_nll=round(history[0], 4), final_nll=round(history[-1], 4),
        mean_abs_logp_error=round(mean_abs_error, 4),
        min_density_on_test_set=f"{min_density:.3e}",
        importance_sampled_total_mass=round(mass, 4),
    )

    assert history[-1] < history[0]  # NLL decreased
    assert mean_abs_error < 0.2
    assert min_density >= 0.0
    assert abs(mass - 1.0) < 0.15


def test_fit_recovers_correlated_gaussian_log_likelihood_tucker():
    key = jax.random.PRNGKey(0)
    k1, k2, k3 = jax.random.split(key, 3)
    d, rho = 3, 0.7
    n_train, n_test, n_mass = 2000, 800, 4000

    data = sample_correlated_gaussian(k1, n=n_train, d=d, rho=rho)
    cov = jnp.full((d, d), rho) + (1.0 - rho) * jnp.eye(d)

    init = init_tucker_from_kde(data, max_rank=6, n_coeffs=(17, 33), gd_steps=10, n_starts=32)
    fitted, history = fit_tucker(init, data, n_steps=1000, lr=1e-3, log_every=0)

    test = sample_correlated_gaussian(k2, n=n_test, d=d, rho=rho)
    est_logp = log_density_tucker(fitted, test)
    true_logp = _true_log_density_gaussian(test, cov)
    mean_abs_error = float(jnp.mean(jnp.abs(est_logp - true_logp)))

    min_density = float(jnp.min(density_tucker(fitted, test)))
    mass = _importance_sampled_mass(
        lambda x: log_density_tucker(fitted, x),
        lambda k, n: sample_correlated_gaussian(k, n=n, d=d, rho=rho),
        k3, n_mass, lambda x: _true_log_density_gaussian(x, cov),
    )

    report(
        "fit_correlated_gaussian_3d_tucker",
        d=d, function=f"correlated Gaussian (rho={rho})", method="Tucker",
        n_train=n_train, n_test=n_test, n_mass_samples=n_mass,
        ranks=fitted.ranks, core_size=int(jnp.prod(jnp.array(fitted.ranks))),
        degree=fitted.factors[0].core.coeffs.shape[-1],
        initial_nll=round(history[0], 4), final_nll=round(history[-1], 4),
        mean_abs_logp_error=round(mean_abs_error, 4),
        min_density_on_test_set=f"{min_density:.3e}",
        importance_sampled_total_mass=round(mass, 4),
    )

    assert history[-1] < history[0]
    assert mean_abs_error < 0.2
    assert min_density >= 0.0
    assert abs(mass - 1.0) < 0.15


def _sample_chained_banana(key, n, curvature=0.2):
    ka, kb, kc = jax.random.split(key, 3)
    x1 = jax.random.normal(ka, (n,), dtype=jnp.float64)
    x2 = jax.random.normal(kb, (n,), dtype=jnp.float64) + curvature * (x1**2 - 1.0)
    x3 = jax.random.normal(kc, (n,), dtype=jnp.float64) + curvature * (x2**2 - 1.0)
    return jnp.stack([x1, x2, x3], axis=1)


def test_fit_runs_on_nonseparable_chained_banana_distribution():
    # A 3D "chained banana": x2 depends nonlinearly on x1, and x3 depends
    # nonlinearly on x2 -- genuinely non-separable across all 3 axes, not
    # just a 2D banana plus an independent extra dimension.
    key = jax.random.PRNGKey(1)
    n_train, n_test = 2000, 800

    data = _sample_chained_banana(key, n_train)
    init = init_tt_from_kde(data, max_rank=8, n_coeffs=(17, 33), gd_steps=10, n_starts=32)
    fitted, history = fit_tt(init, data, n_steps=1000, lr=1e-3, log_every=0)

    test = _sample_chained_banana(jax.random.PRNGKey(4), n_test)
    min_density = float(jnp.min(density_tt(fitted, test)))

    report(
        "fit_chained_banana_tt",
        d=3, function="chained banana (non-separable across all 3 axes)", method="TT",
        n_train=n_train, n_test=n_test, ranks=fitted.ranks,
        degree=fitted.cores[0].coeffs.shape[-1],
        initial_nll=round(history[0], 4), final_nll=round(history[-1], 4),
        min_density_on_test_set=f"{min_density:.3e}",
    )

    assert history[-1] < history[0]
    assert jnp.isfinite(jnp.array(history[-1]))
    assert min_density >= 0.0


def test_fit_runs_on_nonseparable_chained_banana_distribution_tucker():
    key = jax.random.PRNGKey(1)
    n_train, n_test = 2000, 800

    data = _sample_chained_banana(key, n_train)
    init = init_tucker_from_kde(data, max_rank=6, n_coeffs=(17, 33), gd_steps=10, n_starts=32)
    fitted, history = fit_tucker(init, data, n_steps=1000, lr=1e-3, log_every=0)

    test = _sample_chained_banana(jax.random.PRNGKey(4), n_test)
    min_density = float(jnp.min(density_tucker(fitted, test)))

    report(
        "fit_chained_banana_tucker",
        d=3, function="chained banana (non-separable across all 3 axes)", method="Tucker",
        n_train=n_train, n_test=n_test, ranks=fitted.ranks,
        core_size=int(jnp.prod(jnp.array(fitted.ranks))),
        degree=fitted.factors[0].core.coeffs.shape[-1],
        initial_nll=round(history[0], 4), final_nll=round(history[-1], 4),
        min_density_on_test_set=f"{min_density:.3e}",
    )

    assert history[-1] < history[0]
    assert jnp.isfinite(jnp.array(history[-1]))
    assert min_density >= 0.0


def test_fit_recovers_5d_correlated_gaussian_log_likelihood():
    # Pushed to d=5 (vs. d=3 above) as the "dimension 5" example: more bond
    # dimensions, more parameters, genuinely harder cross-approximation and
    # optimization landscape, still cheap enough to run locally (~1 minute,
    # a couple GB of memory).
    key = jax.random.PRNGKey(0)
    k1, k2 = jax.random.split(key)
    d, rho = 5, 0.6
    n_train, n_test = 4000, 1000

    data = sample_correlated_gaussian(k1, n=n_train, d=d, rho=rho)
    cov = jnp.full((d, d), rho) + (1.0 - rho) * jnp.eye(d)

    init = init_tt_from_kde(data, max_rank=8, n_coeffs=(17, 33), gd_steps=10, n_starts=32)
    fitted, history = fit_tt(init, data, n_steps=1500, lr=1e-3, log_every=0)

    test = sample_correlated_gaussian(k2, n=n_test, d=d, rho=rho)
    est_logp = log_density_tt(fitted, test)
    true_logp = _true_log_density_gaussian(test, cov)
    mean_abs_error = float(jnp.mean(jnp.abs(est_logp - true_logp)))
    min_density = float(jnp.min(density_tt(fitted, test)))

    report(
        "fit_correlated_gaussian_5d",
        d=d, function=f"correlated Gaussian (rho={rho})", method="TT",
        n_train=n_train, n_test=n_test, ranks=fitted.ranks,
        degree=fitted.cores[0].coeffs.shape[-1],
        initial_nll=round(history[0], 4), final_nll=round(history[-1], 4),
        mean_abs_logp_error=round(mean_abs_error, 4),
        min_density_on_test_set=f"{min_density:.3e}",
    )

    assert history[-1] < history[0]
    assert mean_abs_error < 0.4
    assert min_density >= 0.0


def test_fit_recovers_5d_gaussian_mixture_harder_problem():
    # A genuinely harder problem at the same d=5: a 3-component, multi-modal
    # mixture instead of a single unimodal Gaussian -- much harder for a
    # single TT (no n_components mixture-of-TTs escape hatch here, by
    # design; see todo.txt).
    key = jax.random.PRNGKey(2)
    k1, k2, k3, k4, k5 = jax.random.split(key, 5)
    d, n_components, spread = 5, 3, 2.5
    n_train, n_test = 4000, 1000

    means = jax.random.uniform(k1, (n_components, d), minval=-spread, maxval=spread, dtype=jnp.float64)
    weights = jax.random.dirichlet(k2, jnp.ones(n_components), dtype=jnp.float64)
    comps = jax.random.categorical(k3, jnp.log(weights), shape=(n_train,))
    data = means[comps] + jax.random.normal(k4, (n_train, d), dtype=jnp.float64)

    init = init_tt_from_kde(data, max_rank=10, n_coeffs=(17, 33), gd_steps=10, n_starts=32)
    fitted, history = fit_tt(init, data, n_steps=1500, lr=1e-3, log_every=0)

    comps_test = jax.random.categorical(k5, jnp.log(weights), shape=(n_test,))
    test = means[comps_test] + jax.random.normal(jax.random.PRNGKey(9), (n_test, d), dtype=jnp.float64)
    est_logp = log_density_tt(fitted, test)
    true_logp = _true_log_density_iso_mixture(test, means, weights)
    mean_abs_error = float(jnp.mean(jnp.abs(est_logp - true_logp)))
    min_density = float(jnp.min(density_tt(fitted, test)))

    report(
        "fit_5d_gaussian_mixture_harder",
        d=d, function=f"{n_components}-component isotropic Gaussian mixture", method="TT",
        n_train=n_train, n_test=n_test, ranks=fitted.ranks,
        degree=fitted.cores[0].coeffs.shape[-1],
        initial_nll=round(history[0], 4), final_nll=round(history[-1], 4),
        mean_abs_logp_error=round(mean_abs_error, 4),
        min_density_on_test_set=f"{min_density:.3e}",
    )

    assert history[-1] < history[0]
    assert mean_abs_error < 0.7
    assert min_density >= 0.0


if __name__ == "__main__":
    run_all_tests(globals())
