"""Shared plumbing for the cryo-EM scripts (cryoem_density.py,
cryoem_path_density.py): cache loading, domain sizing, and the actual
naive-vs-deconvolution model fitting (TT and Tucker), factored out because
both scripts fit the same kind of model on the same kind of cached data.

Run scripts/cryoem_cache.py first (once) to build cryoem_data/cache/*.npz
from the large cryoem_data/cryoem_test.joblib + simulation_info.joblib files.
These scripts only ever read the small cached files.
"""

from __future__ import annotations

from pathlib import Path

import jax.numpy as jnp
import numpy as np
from chebttge import marginal as marginal_tt_fn

from ftde.cheb_product import square_tt, square_tucker
from ftde.gaussian_convolution import GaussianConvolution
from ftde.init import init_tt_from_kde, init_tucker_from_kde
from ftde.kde import make_gaussian_kde, scott_bandwidth
from ftde.noise_kernel import (
    build_diagonal_gaussian_noise_kernel_tt,
    build_diagonal_gaussian_noise_kernel_tucker,
)
from ftde.noisy_regularization import DEFAULT_REG_TARGET_BUDGET
from ftde.noisy_tt_density import fit_noisy_tt
from ftde.noisy_tucker_density import fit_noisy_tucker
from ftde.tt_density import density_tt, fit_tt, squared_integral_tt
from ftde.tucker_density import (
    density_tucker,
    fit_tucker,
    marginal_tucker,
    squared_integral_tucker,
)

CACHE_DIR = Path("cryoem_data/cache")
FIGS_DIR = Path("figs/cryoem")

# Chebyshev degree capped at a fixed, moderate value: the standard implicit
# smoothness regularizer for (otherwise ill-posed) MLE deconvolution -- see
# todo.txt. Rank/degree/n_starts below are tuned empirically for this dataset
# on a CPU-only machine (squaring a rank-r signal gives rank r^2, and
# convolve_tt's per-step cost scales roughly like (r^2)^2 * kernel_degree^2 --
# rank=8 measured at ~9.4s/step, rank=4 at ~0.6s/step at d=10).
SIGNAL_DEGREE = (13,)
SIGNAL_RANK = 4

# Per-dim (rank, degree) override for _rank_degree_for_dim: at dim=4, the
# (SIGNAL_RANK, SIGNAL_DEGREE) defaults above are already the validated best
# found (naive l2=0.271 vs KDE's 0.323 -- see todo.txt). At dim=10 they are
# badly wrong -- a sweep found the real problem was under-ranking, not
# overfitting: rank=4/degree=13 gives naive l2~0.81-1.01 (10x worse than
# KDE's 0.076), while rank=16/degree=3 cuts that to l2=0.111 (~1.5x KDE).
# rank=16 is naive-ONLY, though: square_tt(g) there has an exact rank of
# 256, which makes deconvolution's exact convolution either computationally
# prohibitive (~8.5h/1500 steps unrounded) or numerically unsafe (rounding
# g^2 down via pre_conv_max_rank hit transient NaN losses during training,
# consistent with gl_gpu_noisy.py's own warning that SVD-based round_tt's
# gradient is ill-conditioned near-degenerate spectra at this scale). rank=8
# is the safer middle ground used here instead: still a ~2.7x improvement
# over the old default (naive l2=0.30), small enough that EXACT (unrounded)
# deconvolution is both stable and affordable (~35 min/1500 steps), so
# naive/deconv/KDE stay directly comparable at one operating point. Even at
# this improved, stable rank, deconvolution (with or without the frequency-
# aware Tikhonov regularizer, at any reg_lambda tried) did NOT beat naive on
# the along-path metric (l2 0.338-0.362 vs naive's 0.302) -- consistent with
# the dim=4 finding that this dataset's noise is too mild for deconvolution
# to have real headroom over just ignoring it.
_RANK_DEGREE_FOR_DIM_10_PLUS = (8, 5)


def _rank_degree_for_dim(dim: int) -> tuple[int, int]:
    if dim >= 10:
        return _RANK_DEGREE_FOR_DIM_10_PLUS
    return SIGNAL_RANK, SIGNAL_DEGREE[0]
# The noise kernel is built diagonal (axis-aligned): the true averaged noise
# covariance here has weak off-diagonal correlation (<0.25), and a diagonal
# kernel can be built per-axis without a multi-dimensional cross-approximation
# search -- which, for this data, actually FAILS (silently degenerates to a
# constant) on the full covariance, because convolve_tt's exact convolution
# requires U's domain to be ~3x the signal's domain width regardless of the
# noise's own (here, much smaller) scale, so the true Gaussian underflows to
# exact 0.0 over all but a tiny fraction of that required domain -- see
# build_diagonal_gaussian_noise_kernel_tt's docstring for the full story.
KERNEL_N_COEFFS = (33, 65, 129)
N_STARTS, GD_STEPS = 64, 10
N_STEPS, LR, BATCH_SIZE = 1500, 1e-3, 2048

# The cryo-EM latent dims have genuinely heterogeneous per-mode domain
# widths/noise scales (later PCs carry much less signal variance than
# earlier ones), so deconv fits here pass reg_strength=reg_lambda directly
# with per_axis_target_budget=DEFAULT_REG_TARGET_BUDGET (normalizes each
# axis's own Tikhonov weight independently) rather than a single
# externally pre-computed calibrated_reg_strength -- see
# ftde.noisy_regularization's module comment for why: a single global
# reg_strength calibrated against the single worst-case axis silently
# underflows to a no-op on every other axis (confirmed empirically on this
# exact dataset at dim=10 -- see todo.txt).


def load_marginal_cache(dim: int):
    data = np.load(CACHE_DIR / f"dim{dim}.npz")
    return jnp.asarray(data["points"]), jnp.asarray(data["cov_mean"])


def load_path_cache(noise: str, dim: int):
    data = np.load(CACHE_DIR / f"path_{noise}_dim{dim}.npz")
    return jnp.asarray(data["points"]), jnp.asarray(data["cov_mean"]), np.asarray(data["labels"])


def data_domain(points: jnp.ndarray, margin_frac: float = 0.05) -> jnp.ndarray:
    lo = jnp.min(points, axis=0)
    hi = jnp.max(points, axis=0)
    margin = margin_frac * (hi - lo)
    return jnp.stack([lo - margin, hi + margin], axis=1)


def robust_domain(points: jnp.ndarray, trim_pct: float = 1.0, margin_frac: float = 0.05) -> jnp.ndarray:
    """Domain from the [trim_pct, 100-trim_pct] percentile range per
    dimension instead of raw min/max (see gl_common.robust_domain, same
    idea). At dim=20 here, the raw min/max domain (data_domain) is wide
    enough relative to where the KDE-sqrt target actually has any
    appreciable mass that init_tt_from_kde's cross-approximation pivot
    search silently collapses to a trivial rank-1 fit (confirmed directly:
    even n_starts=256/gd_steps=50, well above this module's own smaller
    defaults, still collapses on the raw domain) -- exactly the "needle in
    a haystack" failure already documented in noise_kernel.py for the
    noise-kernel cross-approximation at d>~3. Trimming to a percentile
    range shrinks the domain enough (by roughly half its width at dim=20)
    for the search to reliably find real structure instead.
    """
    lo = jnp.percentile(points, trim_pct, axis=0)
    hi = jnp.percentile(points, 100.0 - trim_pct, axis=0)
    margin = margin_frac * (hi - lo)
    return jnp.stack([lo - margin, hi + margin], axis=1)


def filter_to_domain(points: jnp.ndarray, domain: jnp.ndarray) -> jnp.ndarray:
    inside = jnp.all((points >= domain[:, 0]) & (points <= domain[:, 1]), axis=1)
    kept = points[inside]
    n_dropped = int(points.shape[0] - kept.shape[0])
    if n_dropped:
        print(f"  (dropped {n_dropped}/{points.shape[0]} points outside the trimmed domain)")
    return kept


def fit_signal(
    points: jnp.ndarray, cov_mean: jnp.ndarray, method: str, *,
    log_prefix: str = "", naive_steps: int | None = None, deconv_steps: int | None = None,
    rank: int | None = None, degree: int | None = None, reg_lambda: float = 0.0,
    deconv_lr: float | None = None,
    convolution_backend: str = "spectral", quadrature_order: int | None = None,
    rsi_max_rank: int = 64, rsi_sketch_size: int | None = None,
):
    """Fit the naive (no-deconvolution) and/or deconvolved squared-TT/Tucker
    density on `points`, given the averaged local noise covariance `cov_mean`.

    ``naive_steps=0`` skips the naive (no-deconvolution) fit entirely --
    useful since, as observed on the Ginzburg-Landau benchmark, the naive fit
    can converge to a badly-biased local minimum on genuinely noisy data and
    is only kept around here as a reference baseline. ``reg_lambda`` is the
    dimensionless Tikhonov regularization budget multiplier, applied via
    per-axis-normalized weights (see ``ftde.noisy_regularization``'s module
    comment for why NOT a single pre-computed ``calibrated_reg_strength``);
    0 (the old, implicit default) disables it. ``deconv_lr`` overrides the
    deconv fit's learning rate (default: same ``LR`` as the naive fit); the
    dim=4 investigation (see todo.txt) found a higher degree + nonzero
    reg_lambda needs a lower deconv learning rate (e.g. 1e-4) than the
    unregularized default to converge cleanly instead of oscillating.
    ``rank``/``degree`` default to ``_rank_degree_for_dim(points.shape[1])``
    when not given (see that function for why dim=4 and dim>=10 need very
    different settings on this dataset).

    ``convolution_backend="gaussian"`` selects direct analytic-Gaussian
    quadrature for TT, avoiding the explicit square and SVD rounding.
    ``quadrature_order`` overrides its per-axis resolution heuristic.
    ``convolution_backend="rsi"`` uses differentiable frozen-row RSI for TT;
    ``rsi_max_rank`` caps its convolution rank. Tucker retains its existing
    spectral convolution.

    Returns (points, naive_fit, naive_hist, deconv_fit, deconv_hist, domain)
    -- `points` is returned because it may be filtered down from the input
    (see below), and callers must plot/evaluate using this filtered set,
    not the original. When naive_steps=0, naive_fit is None and naive_hist
    is [nan].
    """
    if convolution_backend not in {"spectral", "gaussian", "rsi"}:
        raise ValueError("convolution_backend must be spectral, gaussian or rsi")
    if method != "tt" and convolution_backend in {"gaussian", "rsi"}:
        raise ValueError("Gaussian and RSI convolution currently support TT only")
    naive_steps = N_STEPS if naive_steps is None else naive_steps
    deconv_steps = N_STEPS if deconv_steps is None else deconv_steps
    if rank is None or degree is None:
        default_rank, default_degree = _rank_degree_for_dim(points.shape[1])
        rank = default_rank if rank is None else rank
        degree = default_degree if degree is None else degree
    degree = (degree,)
    deconv_lr = LR if deconv_lr is None else deconv_lr
    # A percentile-trimmed domain, not raw min/max: at dim=20, the raw
    # domain is wide enough relative to where the signal actually has mass
    # that init_tt_from_kde's cross-approximation pivot search silently
    # collapses to a trivial rank-1 fit (confirmed directly) -- see
    # robust_domain's docstring. Every point used for KDE-init/training
    # must lie inside the model's own declared domain, so points falling
    # outside the trimmed domain are dropped.
    domain = robust_domain(points)
    points = filter_to_domain(points, domain)
    variances = jnp.diag(cov_mean)
    noise_std = jnp.sqrt(variances)

    if method == "tt":
        init = init_tt_from_kde(
            points, max_rank=rank, domain=domain, n_coeffs=degree,
            n_starts=N_STARTS, gd_steps=GD_STEPS,
        )
        print(f"{log_prefix}signal init ranks:", init.ranks)
        gaussian = None
        u = None
        if convolution_backend == "gaussian":
            gaussian = GaussianConvolution.from_tt(init, noise_std, quadrature_order=quadrature_order)
            print(f"{log_prefix}direct Gaussian quadrature orders:", gaussian.orders)
        else:
            u = build_diagonal_gaussian_noise_kernel_tt(variances, init.domain, n_coeffs=KERNEL_N_COEFFS)
            print(f"{log_prefix}noise kernel ranks:", u.ranks, "degree:", u.cores[0].coeffs.shape[-1])

        if naive_steps <= 0:
            print(f"{log_prefix}naive_steps<=0: skipping the naive (no-deconvolution) fit...")
            naive_fit, naive_hist = None, [float("nan")]
        else:
            print(f"{log_prefix}fitting naive (no deconvolution)...")
            naive_fit, naive_hist = fit_tt(
                init, points, n_steps=naive_steps, lr=LR, batch_size=BATCH_SIZE, log_every=max(naive_steps // 5, 1),
            )
        print(f"{log_prefix}Tikhonov reg_lambda={reg_lambda} (per-axis-normalized budget={DEFAULT_REG_TARGET_BUDGET}) deconv_lr={deconv_lr:.1e}")
        print(f"{log_prefix}fitting with deconvolution ({convolution_backend} convolution)...")
        deconv_fit, deconv_hist = fit_noisy_tt(
            init, u, points, n_steps=deconv_steps, lr=deconv_lr, batch_size=BATCH_SIZE, log_every=max(deconv_steps // 5, 1),
            noise_std=noise_std, reg_strength=reg_lambda,
            per_axis_target_budget=DEFAULT_REG_TARGET_BUDGET if reg_lambda else None,
            gaussian_convolution=gaussian,
            convolution_method="rsi" if convolution_backend == "rsi" else "naive",
            rsi_max_rank=rsi_max_rank, rsi_sketch_size=rsi_sketch_size,
        )
    elif method == "tucker":
        init = init_tucker_from_kde(
            points, max_rank=rank, domain=domain, n_coeffs=degree,
            n_starts=N_STARTS, gd_steps=GD_STEPS,
        )
        print(f"{log_prefix}signal init ranks:", init.ranks)
        u = build_diagonal_gaussian_noise_kernel_tucker(variances, init.domain, n_coeffs=KERNEL_N_COEFFS)
        print(f"{log_prefix}noise kernel ranks:", u.ranks, "degree:", u.factors[0].core.coeffs.shape[-1])

        if naive_steps <= 0:
            print(f"{log_prefix}naive_steps<=0: skipping the naive (no-deconvolution) fit...")
            naive_fit, naive_hist = None, [float("nan")]
        else:
            print(f"{log_prefix}fitting naive (no deconvolution)...")
            naive_fit, naive_hist = fit_tucker(
                init, points, n_steps=naive_steps, lr=LR, batch_size=BATCH_SIZE, log_every=max(naive_steps // 5, 1),
            )
        print(f"{log_prefix}Tikhonov reg_lambda={reg_lambda} (per-axis-normalized budget={DEFAULT_REG_TARGET_BUDGET}) deconv_lr={deconv_lr:.1e}")
        print(f"{log_prefix}fitting with deconvolution ({convolution_backend} convolution)...")
        deconv_fit, deconv_hist = fit_noisy_tucker(
            init, u, points, n_steps=deconv_steps, lr=deconv_lr, batch_size=BATCH_SIZE, log_every=max(deconv_steps // 5, 1),
            noise_std=noise_std, reg_strength=reg_lambda,
            per_axis_target_budget=DEFAULT_REG_TARGET_BUDGET if reg_lambda else None,
        )
    else:
        raise ValueError(f"unknown method {method!r}")

    print(f"{log_prefix}naive nll: {naive_hist[0]:.4f} -> {naive_hist[-1]:.4f}")
    print(f"{log_prefix}deconv nll: {deconv_hist[0]:.4f} -> {deconv_hist[-1]:.4f}")
    return points, naive_fit, naive_hist, deconv_fit, deconv_hist, domain


def density_fn(fitted, method: str):
    """The fitted model's own full joint density p(x) = g(x)^2 / Z, as a
    plain callable -- works the same for a naive_fit or a deconv_fit, since
    both are just squared-TT/Tucker representations of the (estimated) clean
    density (the deconvolution only affects what `fitted` was trained against,
    not how it's evaluated).
    """
    if method == "tt":
        return lambda x: density_tt(fitted, x)
    elif method == "tucker":
        return lambda x: density_tucker(fitted, x)
    raise ValueError(f"unknown method {method!r}")


def kde_marginal_fn(points: jnp.ndarray, dims: tuple[int, ...]):
    """Exact marginal of the diagonal-bandwidth Gaussian KDE fit on the full
    ``points`` (all dims), restricted to ``dims``. A diagonal Gaussian KDE is
    a mixture of axis-aligned Gaussians (one component per data point), and
    marginalizing a Gaussian mixture over a subset of axes just drops the
    other axes' kernel factors -- so re-using the SAME points and the SAME
    per-axis Scott's-rule bandwidth (computed on the full ``d``-dimensional
    data, since Scott's rule itself depends on ``d``), restricted to
    ``dims``, is the exact marginal, not an approximation.
    """
    bandwidth = scott_bandwidth(points)
    dims_arr = jnp.asarray(dims)
    return make_gaussian_kde(points[:, dims_arr], bandwidth=bandwidth[dims_arr])


def marginal_density_fn(fitted, method: str, dims: tuple[int, ...]):
    """Exact closed-form marginal p(x_dims) for a fitted squared-TT/Tucker
    model, via chebttge/ftde's own marginalization (no grid/MC averaging).
    """
    if method == "tt":
        h = square_tt(fitted)
        z = squared_integral_tt(fitted)
        m = marginal_tt_fn(h, keep_modes=dims)
        return lambda xy: m(xy) / z
    elif method == "tucker":
        h = square_tucker(fitted)
        z = squared_integral_tucker(fitted)
        m = marginal_tucker(h, keep_modes=dims)
        return lambda xy: m(xy) / z
    raise ValueError(f"unknown method {method!r}")
