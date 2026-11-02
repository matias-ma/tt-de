"""Regression tests for the noisy (deconvolution) density estimator.

Deliberately at d >= 3 (see test_density.py's module docstring for why
d=1,2 don't meaningfully exercise TT structure) and deliberately cheap
(low degree/rank/steps): these check mechanics (training runs, losses
finite and decrease, forward/objective consistency, degree-selection runs
end to end) -- not deconvolution accuracy at scale, which is demonstrated in
scripts/demo_noisy_1d.py, demo_noisy_2d.py (run individually) and validated
further at full scale on GPU (see scripts/gpu_noisy_hard.py).
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
from conftest import report, run_all_tests

from ftde.cheb_product import square_tt, square_tucker
from ftde.ginzburg_landau import sample_ginzburg_landau
from ftde.init import init_tt_from_kde, init_tucker_from_kde
from ftde.noise_kernel import (
    build_gaussian_mixture_noise_kernel_tt,
    build_gaussian_noise_kernel_tt,
    build_gaussian_noise_kernel_tucker,
)
from ftde.noisy_model_selection import (
    select_noisy_tt_degree,
    select_noisy_tt_reg_lambda,
    select_noisy_tucker_degree,
    select_noisy_tucker_reg_lambda,
)
from ftde.noisy_regularization import calibrated_reg_strength, tikhonov_penalty_tt
from ftde.noisy_tt_density import fit_noisy_tt, log_density_noisy_tt
from ftde.noisy_tucker_density import fit_noisy_tucker, log_density_noisy_tucker
from ftde.tt_density import squared_integral_tt
from ftde.tt_rounding import round_tt
from ftde.tucker_density import squared_integral_tucker


def _sample_correlated_gaussian(key, n, d, rho):
    cov = jnp.full((d, d), rho) + (1.0 - rho) * jnp.eye(d)
    chol = jnp.linalg.cholesky(cov)
    return jax.random.normal(key, (n, d), dtype=jnp.float64) @ chol.T


def test_square_tt_matches_direct_cross_approximation():
    # 3 modes (2 internal bonds): g(x) = prod_i f_i(x_i), a rank-1 TT --
    # squaring it and comparing to an independent cross-approximation of
    # g(x)^2 checks the pointwise-product primitive, not just a single mode.
    from chebttge import ChebTTOptions, tt_qtensge

    d = 3

    def f(x):
        return (
            jnp.exp(-0.3 * x[..., 0] ** 2)
            * (1.0 + 0.2 * jnp.sin(2 * x[..., 1]))
            * jnp.cos(0.3 * x[..., 2])
        )

    dom = jnp.array([[-3.0, 3.0]] * d)
    opts = ChebTTOptions(
        domain=dom, max_rank=1, core_representation="coeff_tensor",
        coeff_core_n_values=(17, 33), n_starts=8, gd_steps=0,
    )
    g = tt_qtensge(f, opts, d=d)
    g_sq = square_tt(g)

    def f_sq(x):
        return f(x) ** 2

    g_sq_direct = tt_qtensge(f_sq, opts, d=d)
    key = jax.random.PRNGKey(5)
    n_points = 300
    xs = jax.random.uniform(key, (n_points, d), minval=-2.9, maxval=2.9, dtype=jnp.float64)
    est = jax.vmap(g_sq)(xs)
    true = jax.vmap(g_sq_direct)(xs)
    max_abs_error = float(jnp.max(jnp.abs(est - true)))

    report(
        "square_tt_matches_direct",
        d=d, function="separable product of 3 univariate factors, squared", method="TT",
        n_eval_points=n_points, ranks=g_sq.ranks, degree=g_sq.cores[0].coeffs.shape[-1],
        max_abs_error=f"{max_abs_error:.3e}",
    )
    assert max_abs_error < 5e-8


def test_square_tucker_matches_direct_cross_approximation():
    from chebtuckerge import ChebTuckerOptions, tucker_qtensge

    d = 3

    def f(x):
        return (
            jnp.exp(-0.3 * x[..., 0] ** 2)
            * (1.0 + 0.2 * jnp.sin(2 * x[..., 1]))
            * jnp.cos(0.3 * x[..., 2])
        )

    dom = jnp.array([[-3.0, 3.0]] * d)
    opts = ChebTuckerOptions(
        domain=dom, max_rank=1, core_representation="coeff_tensor",
        coeff_core_n_values=(17, 33), n_starts=8, gd_steps=0,
    )
    g = tucker_qtensge(f, opts, d=d)
    g_sq = square_tucker(g)

    def f_sq(x):
        return f(x) ** 2

    g_sq_direct = tucker_qtensge(f_sq, opts, d=d)
    key = jax.random.PRNGKey(5)
    n_points = 300
    xs = jax.random.uniform(key, (n_points, d), minval=-2.9, maxval=2.9, dtype=jnp.float64)
    est = jax.vmap(g_sq)(xs)
    true = jax.vmap(g_sq_direct)(xs)
    max_abs_error = float(jnp.max(jnp.abs(est - true)))

    report(
        "square_tucker_matches_direct",
        d=d, function="separable product of 3 univariate factors, squared", method="Tucker",
        n_eval_points=n_points, ranks=g_sq.ranks, degree=g_sq.factors[0].core.coeffs.shape[-1],
        max_abs_error=f"{max_abs_error:.3e}",
    )
    assert max_abs_error < 5e-8


def test_noisy_training_runs_and_decreases_loss():
    d = 3
    rho = 0.5
    n_train = 1500
    noise_std = 0.3

    key = jax.random.PRNGKey(0)
    k1, k2 = jax.random.split(key)
    clean = _sample_correlated_gaussian(k1, n_train, d, rho)
    noisy = clean + noise_std * jax.random.normal(k2, (n_train, d), dtype=jnp.float64)

    signal_domain = jnp.array([[-4.0, 4.0]] * d)
    g0 = init_tt_from_kde(noisy, max_rank=4, domain=signal_domain, n_coeffs=(13,), gd_steps=10, n_starts=32)
    u = build_gaussian_noise_kernel_tt(
        jnp.eye(d) * noise_std**2, g0.domain, max_rank=1, n_coeffs=(33, 65), n_starts=64, gd_steps=10
    )

    g_fit, history = fit_noisy_tt(g0, u, noisy, n_steps=200, lr=1e-3, log_every=0, max_grad_norm=50.0)
    z = float(squared_integral_tt(g_fit))

    report(
        "noisy_training_runs",
        d=d, function=f"correlated Gaussian (rho={rho})", kernel=f"isotropic Gaussian noise std={noise_std}",
        method="TT", n_train=n_train, signal_ranks=g_fit.ranks, kernel_ranks=u.ranks,
        kernel_degree=u.cores[0].coeffs.shape[-1],
        initial_nll=round(history[0], 4), final_nll=round(history[-1], 4), z=round(z, 4),
    )

    assert jnp.all(jnp.isfinite(jnp.array(history)))
    assert history[-1] < history[0]
    assert jnp.isfinite(z) and z > 0.0


def test_noisy_training_runs_and_decreases_loss_tucker():
    d = 3
    rho = 0.5
    n_train = 1500
    noise_std = 0.3

    key = jax.random.PRNGKey(0)
    k1, k2 = jax.random.split(key)
    clean = _sample_correlated_gaussian(k1, n_train, d, rho)
    noisy = clean + noise_std * jax.random.normal(k2, (n_train, d), dtype=jnp.float64)

    signal_domain = jnp.array([[-4.0, 4.0]] * d)
    g0 = init_tucker_from_kde(noisy, max_rank=4, domain=signal_domain, n_coeffs=(13,), gd_steps=10, n_starts=32)
    u = build_gaussian_noise_kernel_tucker(
        jnp.eye(d) * noise_std**2, g0.domain, max_rank=1, n_coeffs=(33, 65), n_starts=64, gd_steps=10
    )

    g_fit, history = fit_noisy_tucker(g0, u, noisy, n_steps=200, lr=1e-3, log_every=0, max_grad_norm=50.0)
    z = float(squared_integral_tucker(g_fit))

    report(
        "noisy_training_runs_tucker",
        d=d, function=f"correlated Gaussian (rho={rho})", kernel=f"isotropic Gaussian noise std={noise_std}",
        method="Tucker", n_train=n_train, signal_ranks=g_fit.ranks, kernel_ranks=u.ranks,
        kernel_degree=u.factors[0].core.coeffs.shape[-1],
        initial_nll=round(history[0], 4), final_nll=round(history[-1], 4), z=round(z, 4),
    )

    assert jnp.all(jnp.isfinite(jnp.array(history)))
    assert history[-1] < history[0]
    assert jnp.isfinite(z) and z > 0.0


def test_noisy_nll_is_consistent_with_forward_convolution():
    # The noisy NLL objective should agree with directly convolving g^2 with U
    # and normalizing by the same Z as the noise-free model.
    from chebttge import ChebTTOptions, convolve_tt, tt_qtensge

    d = 3

    def g_func(x):
        return (2 * jnp.pi) ** (-0.25 * d) * jnp.exp(-0.25 * jnp.sum(x**2, axis=-1))

    dom = jnp.array([[-3.0, 3.0]] * d)
    opts = ChebTTOptions(
        domain=dom, max_rank=1, core_representation="coeff_tensor",
        coeff_core_n_values=(17, 33), n_starts=8, gd_steps=0,
    )
    g = tt_qtensge(g_func, opts, d=d)

    u = build_gaussian_noise_kernel_tt(
        jnp.eye(d) * 0.3**2, dom, max_rank=1, n_coeffs=(17, 33), n_starts=32, gd_steps=5
    )

    y = jnp.array([[0.0, 0.0, 0.0], [0.5, -0.2, 0.3], [-1.0, 0.4, -0.6]])
    logq_direct = jnp.log(convolve_tt(square_tt(g), u)(y)) - jnp.log(squared_integral_tt(g))
    logq_via_helper = log_density_noisy_tt(g, u, y)
    max_abs_diff = float(jnp.max(jnp.abs(logq_direct - logq_via_helper)))

    report(
        "noisy_nll_consistency",
        d=d, function="isotropic Gaussian", kernel="isotropic Gaussian noise std=0.3",
        method="TT", n_eval_points=y.shape[0], max_abs_diff=f"{max_abs_diff:.3e}",
    )
    assert max_abs_diff < 1e-6


def test_noisy_nll_is_consistent_with_forward_convolution_tucker():
    from chebtuckerge import ChebTuckerOptions, convolve_tucker, tucker_qtensge

    d = 3

    def g_func(x):
        return (2 * jnp.pi) ** (-0.25 * d) * jnp.exp(-0.25 * jnp.sum(x**2, axis=-1))

    dom = jnp.array([[-3.0, 3.0]] * d)
    # NOTE: n_starts=8 (fine for the 1D TT convolution unit tests) is too few
    # for a reliable 3D cross-approximation pivot search here -- it can
    # silently converge to a degenerate, near-constant (wrong) per-mode
    # factor. n_starts=64/gd_steps=10 fixes it; found by actually running
    # this rather than assuming the low-d settings would transfer.
    opts = ChebTuckerOptions(
        domain=dom, max_rank=1, core_representation="coeff_tensor",
        coeff_core_n_values=(17, 33), n_starts=64, gd_steps=10,
    )
    g = tucker_qtensge(g_func, opts, d=d)

    u = build_gaussian_noise_kernel_tucker(
        jnp.eye(d) * 0.3**2, dom, max_rank=1, n_coeffs=(17, 33), n_starts=64, gd_steps=10
    )

    y = jnp.array([[0.0, 0.0, 0.0], [0.5, -0.2, 0.3], [-1.0, 0.4, -0.6]])
    logq_direct = jnp.log(convolve_tucker(square_tucker(g), u)(y)) - jnp.log(squared_integral_tucker(g))
    logq_via_helper = log_density_noisy_tucker(g, u, y)
    max_abs_diff = float(jnp.max(jnp.abs(logq_direct - logq_via_helper)))

    report(
        "noisy_nll_consistency_tucker",
        d=d, function="isotropic Gaussian", kernel="isotropic Gaussian noise std=0.3",
        method="Tucker", n_eval_points=y.shape[0], max_abs_diff=f"{max_abs_diff:.3e}",
    )
    assert max_abs_diff < 1e-6


def test_gaussian_mixture_kernel_resolves_narrow_component():
    """A narrow mixture component must survive the expanded convolution domain."""
    from chebttge import ChebTTOptions, convolve_tt, tt_qtensge

    d = 3
    signal_domain = jnp.array([[-3.0, 3.0]] * d)
    signal_opts = ChebTTOptions(
        domain=signal_domain, max_rank=1, core_representation="coeff_tensor",
        coeff_core_n_values=(33,), n_starts=8, gd_steps=2,
    )
    signal = tt_qtensge(
        lambda x: jnp.sqrt(jnp.prod(jnp.exp(-x**2 / 2.0) / jnp.sqrt(2.0 * jnp.pi), axis=-1)),
        signal_opts, d=d,
    )
    stds = jnp.array([0.15, 0.45])
    weights = jnp.array([0.7, 0.3])
    kernel = build_gaussian_mixture_noise_kernel_tt(
        stds, weights, signal_domain, d, n_starts=4, gd_steps=1,
    )
    points = jnp.array([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]])
    estimate = convolve_tt(square_tt(signal), kernel)(points)
    expected = []
    for point in points:
        value = 0.0
        for std, weight in zip(stds, weights, strict=True):
            value = value + weight * jnp.prod(
                jnp.exp(-point**2 / (2.0 * (1.0 + std**2)))
                / jnp.sqrt(2.0 * jnp.pi * (1.0 + std**2))
            )
        expected.append(value)
    expected = jnp.asarray(expected)
    rel_error = jnp.max(jnp.abs(estimate - expected) / expected)
    report(
        "gaussian_mixture_kernel_resolution", d=d, method="TT",
        component_stds=tuple(float(s) for s in stds), degree=kernel.cores[0].coeffs.shape[-1],
        max_relative_error=f"{float(rel_error):.3e}",
    )
    assert rel_error < 2e-3


def test_gaussian_mixture_kernel_works_in_one_dimension():
    signal_domain = jnp.array([[-3.0, 3.0]])
    kernel = build_gaussian_mixture_noise_kernel_tt(
        jnp.array([0.2, 0.5]), jnp.array([2.0, 1.0]), signal_domain, 1,
        n_coeffs=129, n_starts=4, gd_steps=1,
    )
    assert kernel.ranks == ()
    assert jnp.all(jnp.isfinite(kernel(jnp.array([[0.0], [1.0]]))))


def test_noisy_training_runs_on_ginzburg_landau_samples():
    """Exercise the noisy path on a chained, multimodal synthetic target."""
    d = 3
    clean = sample_ginzburg_landau(
        seed=12, n=360, d=d,
        proposal_std=0.45, n_burnin=120, thinning=3, n_chains=60,
    )
    key = jax.random.PRNGKey(13)
    component = jax.random.bernoulli(key, 0.3, (clean.shape[0],))
    std = jnp.where(component, 0.7, 0.25)
    noisy = clean + std[:, None] * jax.random.normal(
        jax.random.PRNGKey(14), clean.shape, dtype=jnp.float64
    )
    domain = jnp.array([[-3.0, 3.0]] * d)
    g0 = init_tt_from_kde(
        noisy, max_rank=2, domain=domain, n_coeffs=(5,), n_starts=8, gd_steps=2,
    )
    kernel = build_gaussian_mixture_noise_kernel_tt(
        jnp.array([0.25, 0.7]), jnp.array([0.7, 0.3]), domain, d,
        n_coeffs=65, n_starts=4, gd_steps=1,
    )
    _, history = fit_noisy_tt(
        g0, kernel, noisy, n_steps=30, lr=2e-4, log_every=0, max_grad_norm=10.0,
    )
    report(
        "noisy_ginzburg_landau_training", d=d, method="TT",
        n_samples=clean.shape[0], signal_degree=5,
        kernel_degree=kernel.cores[0].coeffs.shape[-1],
        initial_nll=round(history[0], 4), final_nll=round(history[-1], 4),
    )
    assert jnp.all(jnp.isfinite(jnp.asarray(history)))
    assert history[-1] < history[0]


def test_tt_rounding_reduces_convolution_rank():
    d = 3
    domain = jnp.array([[-3.0, 3.0]] * d)
    g = init_tt_from_kde(
        _sample_correlated_gaussian(jax.random.PRNGKey(21), 300, d, 0.4),
        max_rank=2, domain=domain, n_coeffs=(5,), n_starts=8, gd_steps=2,
    )
    kernel = build_gaussian_mixture_noise_kernel_tt(
        jnp.array([0.25, 0.7]), jnp.array([0.7, 0.3]), domain, d,
        n_coeffs=65, n_starts=4, gd_steps=1,
    )
    from chebttge import convolve_tt

    unrounded = convolve_tt(square_tt(g), kernel)
    rounded = round_tt(unrounded, max_rank=3)
    report(
        "tt_rounding", unrounded_ranks=unrounded.ranks, rounded_ranks=rounded.ranks,
    )
    assert all(rank <= 3 for rank in rounded.ranks)
    assert jnp.all(jnp.isfinite(rounded(jnp.zeros((4, d)))))


def test_select_noisy_tt_degree_runs_and_picks_a_valid_candidate():
    # Cheap mechanics check for the 1-SE-rule degree selector (see
    # noisy_model_selection.py for why plain held-out noisy NLL alone can't
    # reliably pick model complexity) -- not an accuracy test.
    d = 3
    rho = 0.4
    n = 900
    noise_std = 0.3

    key = jax.random.PRNGKey(3)
    k1, k2 = jax.random.split(key)
    clean = _sample_correlated_gaussian(k1, n, d, rho)
    noisy = clean + noise_std * jax.random.normal(k2, (n, d), dtype=jnp.float64)

    signal_domain = jnp.array([[-4.0, 4.0]] * d)
    u = build_gaussian_noise_kernel_tt(
        jnp.eye(d) * noise_std**2, signal_domain, max_rank=1, n_coeffs=(33, 65), n_starts=64, gd_steps=10
    )

    candidate_degrees = (5, 9, 13)
    result = select_noisy_tt_degree(
        noisy, u, candidate_degrees,
        signal_domain=signal_domain, max_rank=3, val_fraction=0.3,
        n_steps=80, lr=1e-3, refit_on_full_data=True,
    )

    report(
        "select_noisy_tt_degree",
        d=d, function=f"correlated Gaussian (rho={rho})", kernel=f"isotropic Gaussian noise std={noise_std}",
        method="TT + 1-SE degree selection", n=n, candidates=candidate_degrees,
        val_nll_by_degree={c.n_coeffs: round(c.val_nll_mean, 4) for c in result.candidates},
        selected_degree=result.selected.n_coeffs,
    )

    assert result.selected.n_coeffs in candidate_degrees
    assert len(result.candidates) == len(candidate_degrees)
    assert all(jnp.isfinite(c.val_nll_mean) for c in result.candidates)
    assert result.final_model is not None


def test_calibrated_reg_strength_hits_target_budget_and_scales_with_reg_lambda():
    # calibrated_reg_strength solves for reg_strength such that
    # reg_strength * max_per_coefficient_weight == reg_lambda * target_budget
    # -- check that identity directly, and that reg_lambda=0 disables it.
    domain = jnp.array([[-3.0, 3.0], [-1.0, 1.0]])
    noise_std = jnp.array([0.4, 0.1])
    degree = 9
    target_budget = 0.5

    from ftde.noisy_regularization import chebyshev_tikhonov_weights

    g_sq_degree = 2 * degree - 1
    max_weight = max(
        float(jnp.max(chebyshev_tikhonov_weights(g_sq_degree, (float(a), float(b)), float(s))))
        for (a, b), s in zip(domain, noise_std, strict=True)
    )

    reg_strength_1 = calibrated_reg_strength(domain, noise_std, degree, 1.0, target_budget)
    reg_strength_3 = calibrated_reg_strength(domain, noise_std, degree, 3.0, target_budget)
    reg_strength_0 = calibrated_reg_strength(domain, noise_std, degree, 0.0, target_budget)

    report(
        "calibrated_reg_strength",
        domain=[tuple(float(x) for x in row) for row in domain], noise_std=tuple(float(s) for s in noise_std),
        degree=degree, target_budget=target_budget, max_weight=f"{max_weight:.3e}",
        reg_strength_at_lambda_1=f"{reg_strength_1:.3e}", reg_strength_at_lambda_3=f"{reg_strength_3:.3e}",
    )
    assert reg_strength_0 == 0.0
    assert abs(reg_strength_1 * max_weight - target_budget) < 1e-9 * target_budget
    assert abs(reg_strength_3 - 3.0 * reg_strength_1) < 1e-9 * reg_strength_1


def test_per_axis_target_budget_fixes_heterogeneous_axis_underflow():
    # Regression test for a real bug found on cryo-EM dim=10 data (see
    # todo.txt): with genuinely heterogeneous per-axis noise_std, a single
    # externally-calibrated reg_strength (via calibrated_reg_strength) is
    # dominated by whichever one axis has the largest raw Tikhonov weight,
    # which can force it to underflow to an effective no-op on every OTHER
    # axis even though reg_lambda is nonzero. per_axis_target_budget
    # instead normalizes each axis independently and should not underflow.
    domain = jnp.array([[-3.0, 3.0], [-3.0, 3.0], [-3.0, 3.0]])
    # Axis 0 has a much larger noise_std than axes 1-2 -- its raw Tikhonov
    # weight at the same degree is astronomically larger, exactly the
    # pathology observed on real (10-axis) cryo-EM data.
    noise_std = jnp.array([2.0, 0.05, 0.05])
    degree = 13
    reg_lambda = 1.0

    global_reg_strength = calibrated_reg_strength(domain, noise_std, degree, reg_lambda)

    # A dummy g^2-like ChebTT: build via square_tt of a small random TT, just
    # to exercise tikhonov_penalty_tt itself, not full training.
    g0 = init_tt_from_kde(
        _sample_correlated_gaussian(jax.random.PRNGKey(7), 300, 3, 0.0),
        max_rank=2, domain=domain, n_coeffs=(degree,), n_starts=8, gd_steps=2,
    )
    g_sq = square_tt(g0)

    penalty_global = float(tikhonov_penalty_tt(g_sq, noise_std)) * global_reg_strength
    penalty_per_axis = float(
        tikhonov_penalty_tt(g_sq, noise_std, per_axis_target_budget=0.5)
    ) * reg_lambda

    report(
        "per_axis_target_budget_fixes_underflow",
        domain_widths=(6.0, 6.0, 6.0), noise_std=tuple(float(s) for s in noise_std), degree=degree,
        global_reg_strength=f"{global_reg_strength:.3e}",
        penalty_with_global_calibration=f"{penalty_global:.3e}",
        penalty_with_per_axis_budget=f"{penalty_per_axis:.3e}",
    )
    # The global-calibration path underflows to ~0 for this heterogeneous
    # case; the per-axis path should not.
    assert global_reg_strength < 1e-30
    assert penalty_per_axis > 1e-6


def test_tikhonov_regularized_training_runs_and_shrinks_high_degree_coefficients():
    # A cheap end-to-end check that reg_strength > 0 actually changes the fit
    # in the expected direction: at a deliberately high degree relative to
    # this small/clean training set, the regularized fit's high-frequency
    # g^2 coefficients should have smaller total energy than the
    # unregularized fit's (the whole point of the penalty).
    d = 3
    rho = 0.4
    n_train = 900
    noise_std_val = 0.3

    key = jax.random.PRNGKey(1)
    k1, k2 = jax.random.split(key)
    clean = _sample_correlated_gaussian(k1, n_train, d, rho)
    noisy = clean + noise_std_val * jax.random.normal(k2, (n_train, d), dtype=jnp.float64)

    signal_domain = jnp.array([[-4.0, 4.0]] * d)
    degree = 17
    noise_std = jnp.full((d,), noise_std_val)
    g0 = init_tt_from_kde(noisy, max_rank=3, domain=signal_domain, n_coeffs=(degree,), gd_steps=10, n_starts=32)
    u = build_gaussian_noise_kernel_tt(
        jnp.eye(d) * noise_std_val**2, g0.domain, max_rank=1, n_coeffs=(33, 65), n_starts=64, gd_steps=10
    )

    reg_strength = calibrated_reg_strength(g0.domain, noise_std, degree, reg_lambda=1.0)
    g_unreg, hist_unreg = fit_noisy_tt(
        g0, u, noisy, n_steps=150, lr=1e-4, log_every=0, max_grad_norm=50.0,
    )
    g_reg, hist_reg = fit_noisy_tt(
        g0, u, noisy, n_steps=150, lr=1e-4, log_every=0, max_grad_norm=50.0,
        noise_std=noise_std, reg_strength=reg_strength,
    )
    penalty_unreg = float(tikhonov_penalty_tt(square_tt(g_unreg), noise_std))
    penalty_reg = float(tikhonov_penalty_tt(square_tt(g_reg), noise_std))

    report(
        "tikhonov_regularized_training",
        d=d, degree=degree, reg_strength=f"{reg_strength:.3e}",
        unreg_final_nll=round(hist_unreg[-1], 4), reg_final_nll=round(hist_reg[-1], 4),
        unreg_penalty=f"{penalty_unreg:.4f}", reg_penalty=f"{penalty_reg:.4f}",
    )
    assert jnp.all(jnp.isfinite(jnp.array(hist_reg)))
    assert penalty_reg < penalty_unreg


def test_select_noisy_tucker_degree_runs_and_picks_a_valid_candidate():
    # Tucker mirror of test_select_noisy_tt_degree_runs_and_picks_a_valid_candidate.
    d = 3
    rho = 0.4
    n = 900
    noise_std = 0.3

    key = jax.random.PRNGKey(3)
    k1, k2 = jax.random.split(key)
    clean = _sample_correlated_gaussian(k1, n, d, rho)
    noisy = clean + noise_std * jax.random.normal(k2, (n, d), dtype=jnp.float64)

    signal_domain = jnp.array([[-4.0, 4.0]] * d)
    u = build_gaussian_noise_kernel_tucker(
        jnp.eye(d) * noise_std**2, signal_domain, max_rank=1, n_coeffs=(33, 65), n_starts=64, gd_steps=10
    )

    candidate_degrees = (5, 9, 13)
    result = select_noisy_tucker_degree(
        noisy, u, candidate_degrees,
        signal_domain=signal_domain, max_rank=3, val_fraction=0.3,
        n_steps=80, lr=1e-3, refit_on_full_data=True,
    )

    report(
        "select_noisy_tucker_degree",
        d=d, function=f"correlated Gaussian (rho={rho})", kernel=f"isotropic Gaussian noise std={noise_std}",
        method="Tucker + 1-SE degree selection", n=n, candidates=candidate_degrees,
        val_nll_by_degree={c.n_coeffs: round(c.val_nll_mean, 4) for c in result.candidates},
        selected_degree=result.selected.n_coeffs,
    )

    assert result.selected.n_coeffs in candidate_degrees
    assert len(result.candidates) == len(candidate_degrees)
    assert all(jnp.isfinite(c.val_nll_mean) for c in result.candidates)
    assert result.final_model is not None


def test_select_noisy_tt_reg_lambda_runs_and_picks_a_valid_candidate():
    # Cheap mechanics check for the reg_lambda 1-SE-rule selector (mirrors
    # test_select_noisy_tt_degree_runs_and_picks_a_valid_candidate) -- not an
    # accuracy test.
    d = 3
    rho = 0.4
    n = 900
    noise_std_val = 0.3

    key = jax.random.PRNGKey(3)
    k1, k2 = jax.random.split(key)
    clean = _sample_correlated_gaussian(k1, n, d, rho)
    noisy = clean + noise_std_val * jax.random.normal(k2, (n, d), dtype=jnp.float64)

    signal_domain = jnp.array([[-4.0, 4.0]] * d)
    noise_std = jnp.full((d,), noise_std_val)
    u = build_gaussian_noise_kernel_tt(
        jnp.eye(d) * noise_std_val**2, signal_domain, max_rank=1, n_coeffs=(33, 65), n_starts=64, gd_steps=10
    )

    candidate_reg_lambdas = (0.0, 1.0, 3.0)
    result = select_noisy_tt_reg_lambda(
        noisy, u, candidate_reg_lambdas,
        degree=9, noise_std=noise_std, signal_domain=signal_domain, max_rank=3, val_fraction=0.3,
        n_steps=80, lr=1e-4, refit_on_full_data=True,
    )

    report(
        "select_noisy_tt_reg_lambda",
        d=d, function=f"correlated Gaussian (rho={rho})", kernel=f"isotropic Gaussian noise std={noise_std_val}",
        method="TT + 1-SE reg_lambda selection", n=n, candidates=candidate_reg_lambdas,
        val_nll_by_reg_lambda={c.reg_lambda: round(c.val_nll_mean, 4) for c in result.candidates},
        selected_reg_lambda=result.selected.reg_lambda,
    )

    assert result.selected.reg_lambda in candidate_reg_lambdas
    assert len(result.candidates) == len(candidate_reg_lambdas)
    assert all(jnp.isfinite(c.val_nll_mean) for c in result.candidates)
    assert result.final_model is not None


def test_select_noisy_tucker_reg_lambda_runs_and_picks_a_valid_candidate():
    # Tucker mirror of test_select_noisy_tt_reg_lambda_runs_and_picks_a_valid_candidate.
    d = 3
    rho = 0.4
    n = 900
    noise_std_val = 0.3

    key = jax.random.PRNGKey(3)
    k1, k2 = jax.random.split(key)
    clean = _sample_correlated_gaussian(k1, n, d, rho)
    noisy = clean + noise_std_val * jax.random.normal(k2, (n, d), dtype=jnp.float64)

    signal_domain = jnp.array([[-4.0, 4.0]] * d)
    noise_std = jnp.full((d,), noise_std_val)
    u = build_gaussian_noise_kernel_tucker(
        jnp.eye(d) * noise_std_val**2, signal_domain, max_rank=1, n_coeffs=(33, 65), n_starts=64, gd_steps=10
    )

    candidate_reg_lambdas = (0.0, 1.0, 3.0)
    result = select_noisy_tucker_reg_lambda(
        noisy, u, candidate_reg_lambdas,
        degree=9, noise_std=noise_std, signal_domain=signal_domain, max_rank=3, val_fraction=0.3,
        n_steps=80, lr=1e-4, refit_on_full_data=True,
    )

    report(
        "select_noisy_tucker_reg_lambda",
        d=d, function=f"correlated Gaussian (rho={rho})", kernel=f"isotropic Gaussian noise std={noise_std_val}",
        method="Tucker + 1-SE reg_lambda selection", n=n, candidates=candidate_reg_lambdas,
        val_nll_by_reg_lambda={c.reg_lambda: round(c.val_nll_mean, 4) for c in result.candidates},
        selected_reg_lambda=result.selected.reg_lambda,
    )

    assert result.selected.reg_lambda in candidate_reg_lambdas
    assert len(result.candidates) == len(candidate_reg_lambdas)
    assert all(jnp.isfinite(c.val_nll_mean) for c in result.candidates)
    assert result.final_model is not None


if __name__ == "__main__":
    run_all_tests(globals())
