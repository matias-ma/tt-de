"""Noisy (deconvolution) Ginzburg-Landau density estimation at moderate and
high dimension: meant for a GPU, not a laptop.

Corrupts samples from the discretized Ginzburg-Landau field (see
src/ftde/ginzburg_landau.py) with a 2-component Gaussian-mixture noise
kernel, then compares three estimators against the known true clean
density (squared relative L2 error, same convention as
github.com/darenwang/variance-reduced-tensor-sketching):

  - KDE            : plain Gaussian KDE on the noisy data (ignores noise;
                      bandwidth chosen via least-squares cross-validation,
                      not the default Scott's rule -- see ftde.kde)
  - naive TT       : squared-TT NLL-fit directly on the noisy data (ignores noise)
  - deconv TT      : squared-TT fit by minimizing the NLL of U * g^2 (exact
                      closed-form convolution against the known noise kernel
                      U), with the frequency-aware Tikhonov penalty from
                      src/ftde/noisy_regularization.py to control
                      deconvolution's classical ill-posedness.

Also saves 2D marginal figures (true GL density, true clean data, observed noisy data, noise
kernel U, KDE baseline, naive fit, deconv fit) to figs/ginzburg_landau/.

--- How to run on the GPU cluster ---

See the README's "Install" section for the pip install recipe (order
matters: chebdCUR, then this repo, with any CUDA jax build installed last).
Then, with the GPU visible to jax:

    python scripts/gl_gpu_noisy.py                          # moderate (d=8) and high (d=20)
    python scripts/gl_gpu_noisy.py 8                         # only d=8
    python scripts/gl_gpu_noisy.py 8 12 20                   # custom dims
    python scripts/gl_gpu_noisy.py 8 --deconv-steps 1000     # fewer (expensive) deconv steps
    python scripts/gl_gpu_noisy.py 8 --naive-steps 500 --deconv-steps 1000
    python scripts/gl_gpu_noisy.py 8 --n-pts 5000              # override number of observed points
    python scripts/gl_gpu_noisy.py 8 --sigma 2.0                # 2x the default noise scale
    python scripts/gl_gpu_noisy.py 8 --noise isotropic           # old axis-aligned mixture noise
    python scripts/gl_gpu_noisy.py 8 --noise colored              # heteroscedastic, 100 sampled covariances
    python scripts/gl_gpu_noisy.py 8 --n-pts 1e5                  # scientific notation is fine
    python scripts/gl_gpu_noisy.py 8 --naive-steps 0             # skip the naive baseline entirely
    python scripts/gl_gpu_noisy.py 8 --pre-conv-max-rank 0       # disable pre-conv rounding (exact, rank^4-slow)
    python scripts/gl_gpu_noisy.py 8 --pre-conv-max-rank 20      # round g^2 to a specific rank instead
    python scripts/gl_gpu_noisy.py 8 --reg-lambda 3.0             # regularize 3x harder than the baseline
    python scripts/gl_gpu_noisy.py 8 --reg-lambda 0                # disable regularization entirely

--- Local smoke check (CPU, tiny/fast, just to catch bugs) ---

    FTDE_TEST_SCALE=smoke python scripts/gl_gpu_noisy.py

Runs the same code path at trivially small dimension/rank/degree/steps so it
finishes in well under a minute on a CPU-only machine. This does NOT
validate accuracy at scale -- it only checks the code path is
exception-free and qualitatively sane.
"""

from __future__ import annotations

import argparse
import os
import time

import jax.numpy as jnp
import numpy as np
from gl_common import (
    dim_pairs_for,
    filter_to_domain,
    int_arg,
    relative_l2_error,
    robust_domain,
    save_noisy_marginal_figure,
)

from ftde.ginzburg_landau import (
    ginzburg_landau_density,
    ginzburg_landau_log_normalizer,
    sample_ginzburg_landau,
)
from ftde.init import init_tt_from_kde
from ftde.kde import make_gaussian_kde, select_gaussian_kde_bandwidth
from ftde.noise_kernel import (
    build_gaussian_mixture_noise_kernel_tt,
    build_pairwise_correlated_noise_kernel_tt,
)
from ftde.noisy_regularization import chebyshev_tikhonov_weights
from ftde.noisy_tt_density import fit_noisy_tt
from ftde.tt_convolution import TTConvolution
from ftde.tt_density import density_tt, fit_tt

SCALE = os.environ.get("FTDE_TEST_SCALE", "full")
IS_SMOKE = SCALE == "smoke"

# Noise: at --sigma 1.0 (the default), two options:
#  - isotropic (--noise isotropic, the old default/behavior): a 2-component
#    axis-aligned Gaussian mixture, deliberately noisier relative to the
#    signal than real cryo-EM data tested elsewhere in this project (there,
#    noise was ~4-11% of total variance; here it's ~30%).
#  - nonisotropic (--noise nonisotropic, now the DEFAULT): a single
#    correlated Gaussian, correlated in independent adjacent PAIRS of
#    coordinates -- (0,1), (2,3), ... -- each pair with off-diagonal
#    covariance std^2*rho, zero correlation across pairs. Genuinely
#    non-diagonal (unlike the isotropic mixture), and physically reasonable
#    for this chain-structured target (nearby coordinates' measurement
#    noise plausibly correlates). `std` is set to match the isotropic
#    mixture's own overall RMS noise scale at sigma=1, so --sigma is
#    comparable across both --noise modes.
#
#    WHY PAIRS, not full correlation across all distances (e.g. AR(1)
#    cov[i,j]=std^2*rho^|i-j|): confirmed empirically that fitting a
#    genuinely d-dimensional correlated noise kernel via cross-approximation
#    silently degenerates to a trivial rank-1/degree-1 fit once d is more
#    than ~2-3 (crashing downstream) -- the "needle in a haystack" pivot-
#    search failure documented in noise_kernel.py. A 2D cross-approximation
#    per pair is confirmed to work reliably where the full d-dim case fails,
#    so build_pairwise_correlated_noise_kernel_tt uses exactly that -- see
#    its docstring. The noise actually SAMPLED below uses the same
#    block-diagonal (pairs-only) covariance as the kernel U built for
#    deconvolution, not a fuller/fancier correlation structure -- U must be
#    the TRUE noise distribution for deconvolution to be meaningful at all,
#    so this is a real modeling constraint, not just cosmetic consistency.
NOISE_STDS = jnp.array([0.3, 0.9])
NOISE_WEIGHTS = jnp.array([0.7, 0.3])
NONISO_BASE_STD = float(jnp.sqrt(jnp.sum(NOISE_WEIGHTS * NOISE_STDS**2)))
NONISO_RHO = 0.5
NONISO_KERNEL_RANK = 3


def _noniso_std(sigma: float) -> float:
    return sigma * NONISO_BASE_STD


def _pairwise_block_covariance(dim: int, std: float, rho: float) -> np.ndarray:
    """Block-diagonal: independent adjacent pairs (0,1), (2,3), ... each
    correlated by `rho`, zero correlation across pairs -- see the
    module-level comment above for why this (not full AR(1)) structure.
    """
    cov = np.eye(dim) * std**2
    for k in range(0, dim - 1, 2):
        cov[k, k + 1] = cov[k + 1, k] = rho * std**2
    return cov


def _noniso_covariance(dim: int, sigma: float) -> np.ndarray:
    return _pairwise_block_covariance(dim, _noniso_std(sigma), NONISO_RHO)


# "Colored" noise: unlike isotropic/nonisotropic (one fixed noise
# distribution shared by every point), each observation's noise is drawn
# from its OWN covariance matrix, itself randomly sampled from a pool of
# COLORED_N_COVARIANCES=100 distinct (std_k, rho_k) pairwise-block
# covariances -- genuinely heteroscedastic, correlated ("colored", not
# white) noise, the kind seen in real per-particle measurement noise (e.g.
# the real cryo-EM data elsewhere in this project has its own per-point
# noise covariance, handled the same way this does -- see the next
# paragraph). std_k is log-uniform around the nonisotropic base scale
# (varying the noise's physical strength point to point) and rho_k is
# uniform in a range that keeps every covariance safely positive-definite
# (block eigenvalues are 1 +/- rho, so |rho| must stay under 1 with margin).
#
# No single fixed U can be the TRUE kernel for every point when the noise
# genuinely varies per point -- exact deconvolution would need a different
# kernel per point. Instead, following this project's own existing
# precedent for real per-point-varying cryo-EM noise (cryoem_common.py's
# `fit_signal`, which builds one kernel from the AVERAGED per-point noise
# covariance), we build one approximate kernel from the covariances'
# average. Because every one of the 100 covariances shares the same
# pairwise-block structure, their average is EXACTLY another pairwise-block
# covariance (averaging is linear), so this reduces to one more call to the
# same build_pairwise_correlated_noise_kernel_tt used for --noise
# nonisotropic -- just with the averaged (std, rho) instead of a single
# fixed pair. This is a real, acknowledged approximation (not exact
# deconvolution of heteroscedastic noise), same as the cryo-EM case.
COLORED_N_COVARIANCES = 100
COLORED_STD_LOG_RANGE = (0.4, 2.2)  # multiplicative range around NONISO_BASE_STD*sigma
# Deliberately one-sided (not e.g. (-0.6, 0.6)): a symmetric-around-zero
# range averages away to ~0 correlation over 100 draws (confirmed
# empirically: averaged rho was ~0.1 with a symmetric range, giving a
# visually "blobby" near-isotropic kernel/scatter despite each individual
# covariance being genuinely correlated) -- the elongation direction is the
# SAME fixed pairwise axis for every draw here (always (0,1), (2,3), ...,
# never rotated), so if rho's sign flips draw to draw, the elongations
# along that axis and its mirror image wash each other out on average
# instead of reinforcing. Keeping rho consistently positive (and pushed
# higher, up to 0.85) keeps both the per-point noise draws AND their
# average genuinely elongated/"colored" rather than round.
COLORED_RHO_RANGE = (0.35, 0.85)


def _colored_covariance_params(sigma: float, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """Sample COLORED_N_COVARIANCES (std_k, rho_k) pairs. Returns arrays of
    shape (COLORED_N_COVARIANCES,).
    """
    rng = np.random.default_rng(seed)
    base_std = _noniso_std(sigma)
    log_lo, log_hi = np.log(COLORED_STD_LOG_RANGE[0]), np.log(COLORED_STD_LOG_RANGE[1])
    stds = base_std * np.exp(rng.uniform(log_lo, log_hi, size=COLORED_N_COVARIANCES))
    rhos = rng.uniform(*COLORED_RHO_RANGE, size=COLORED_N_COVARIANCES)
    return stds, rhos


def _colored_effective_std_rho(stds: np.ndarray, rhos: np.ndarray) -> tuple[float, float]:
    """The single (std, rho) pairwise-block covariance that equals the
    AVERAGE of the sampled per-covariance matrices (see the module-level
    comment above) -- what the approximate deconvolution kernel is built
    from.
    """
    mean_var = float(np.mean(stds**2))
    mean_offdiag = float(np.mean(rhos * stds**2))
    return float(np.sqrt(mean_var)), mean_offdiag / mean_var


def _sample_colored_noise(dim: int, sigma: float, n_pts: int, seed: int) -> np.ndarray:
    stds, rhos = _colored_covariance_params(sigma, seed=seed)
    rng = np.random.default_rng(seed + 1)
    labels = rng.integers(0, COLORED_N_COVARIANCES, size=n_pts)
    noise = np.empty((n_pts, dim))
    for k in range(COLORED_N_COVARIANCES):
        idx = np.flatnonzero(labels == k)
        if idx.size == 0:
            continue
        cov_k = _pairwise_block_covariance(dim, stds[k], rhos[k])
        noise[idx] = rng.multivariate_normal(mean=np.zeros(dim), cov=cov_k, size=idx.size)
    return noise


MODERATE_DIM, HIGH_DIM = (3, 4) if IS_SMOKE else (8, 20)


def _rank_degree_for_dim(dim: int) -> tuple[int, int]:
    """Model capacity per dimension -- smoothed into 3 tiers rather than a
    single dim<=10 cutoff; see the identical function's docstring in
    gl_gpu_noise_free.py for why.
    """
    if IS_SMOKE:
        return 3, 9
    if dim <= 8:
        return 8, 17
    if dim <= 14:
        return 10, 19
    return 14, 21


# The deconvolution step is far more expensive than the naive fit per
# training step: squaring a rank-R signal gives a rank-R^2 tensor
# (cheb_product.square_tt), and convolve_tt's output rank is the *product*
# of that squared rank with the kernel's own rank (chebdCUR's convolve_tt),
# with per-mode cost scaling roughly like
# O(kernel_degree^2 * (signal_rank^2)^2 * kernel_rank^2). At rank=8 that's
# a combined bond rank of 8^2 * kernel_rank=2 = 128 -- 16x more expensive
# than rank=4 (bond rank 4^2*2=32) for the SAME kernel degree, since cost
# scales like rank^4. Give the deconv stage its own, deliberately smaller
# rank so this doesn't quietly become infeasible; the naive/KDE-init rank
# above is unaffected.
def _deconv_rank_for_dim(dim: int) -> int:
    if IS_SMOKE:
        return 3
    return 4 if dim <= 10 else 6


# square_tt(g) gives g^2 an EXACT rank of rank(g)^2 (a worst-case
# combinatorial bound from the Chebyshev linearization identity, not
# necessarily the rank g^2 actually needs), and convolve_tt's cost scales
# with THAT rank squared again -- so the true driver of the "rank^4" cost
# is g^2's rank, not g's own. round_tt (SVD-based, JAX-jit-safe with a
# fixed max_rank) can compress g^2 back down before convolving, which
# directly cuts this cost (unlike --deconv-steps or the deconv rank
# itself). Empirically tested here (small scale, d=6): rounding rank(g)^2
# all the way back down to rank(g) itself is NOT safe -- it produced a
# much worse fit from the very first step and NaN losses partway through
# training (SVD's gradient is ill-conditioned near-degenerate spectra,
# which training can drift into). Rounding to ~2x rank(g) instead was
# stable and gave comparable or better NLL in that same test, while still
# cutting the dominant cost term substantially (rank(g)^2=36 -> 12 is
# already a ~9x reduction in the combined bond rank at rank(g)=6). This is
# ONE data point, not a guarantee -- if you see NaN losses with rounding
# enabled, raise this value or disable it with --pre-conv-max-rank 0.
def _pre_conv_max_rank_for(deconv_rank: int) -> int:
    return 2 * deconv_rank


N_TRAIN = 300 if IS_SMOKE else 20_000
N_TEST = 200 if IS_SMOKE else 4_000
N_STARTS = 8 if IS_SMOKE else 96
GD_STEPS = 2 if IS_SMOKE else 15
DEFAULT_NAIVE_STEPS = 0
DEFAULT_DECONV_STEPS = 60 if IS_SMOKE else 4000
NAIVE_LR = 1e-3
DECONV_LR = 1e-4
# The Tikhonov penalty's own per-coefficient weight, expm1((noise_std * k *
# pi / L)^2), grows EXPONENTIALLY in noise_std -- confirmed empirically:
# at the degree square_tt roughly doubles the signal to, this project's old
# fixed REG_STRENGTH=1e-5 (implicitly tuned for the old isotropic default's
# conservative std=0.3) only worked by luck at that specific scale; at
# std=0.55 the max per-coefficient weight is already ~1e11x larger
# (4.6e4 -> 4.9e15), which silently blows the training NLL up to the
# millions from the very first step. Since --sigma now lets the noise scale
# vary freely (and nonisotropic's std differs from isotropic's), a FIXED
# reg_strength constant is not safe -- REG_TARGET_BUDGET instead fixes the
# overall regularization *budget* (reg_strength * max_weight), matching
# the old default's own effective budget at std=0.3, and reg_strength is
# solved for per-run in `run()` to hit that budget at whatever noise_std/
# degree/domain this run actually has.
REG_TARGET_BUDGET = 0.5
BATCH_SIZE = None if IS_SMOKE else 2048
# Let the kernel builder resolve the narrowest mixture component on its
# expanded convolution domain. A fixed small value can be insufficient for
# some signal domains and silently corrupt the forward likelihood.
KERNEL_N_COEFFS = None
LOG_EVERY = 100


def _probe_seconds_per_step(train_fn, probe_steps: int = 3) -> float:
    """Run a few throwaway training steps (fresh optimizer state -- doesn't
    affect the real training run that follows) purely to time them, so we
    can print an ETA before committing to the full run.
    """
    t0 = time.time()
    train_fn(n_steps=probe_steps, log_every=0)
    return (time.time() - t0) / probe_steps


def run(
    dim: int, naive_steps: int, deconv_steps: int, n_pts: int, noise_type: str, sigma: float,
    pre_conv_max_rank: int | None, reg_lambda: float,
    rank: int | None = None, deconv_rank: int | None = None, degree: int | None = None,
    convolution_backend: str = "rsi", save_figures: bool = True,
    rsi_max_rank: int = 128, rsi_sketch_size: int | None = None,
) -> dict[str, float]:
    default_rank, default_degree = _rank_degree_for_dim(dim)
    rank = rank if rank is not None else default_rank
    degree = degree if degree is not None else default_degree
    deconv_rank = deconv_rank if deconv_rank is not None else _deconv_rank_for_dim(dim)
    if convolution_backend == "direct":
        if pre_conv_max_rank is not None and pre_conv_max_rank > 0:
            raise ValueError("direct convolution does not use pre-convolution rank rounding")
        pre_conv_max_rank = None
    elif convolution_backend not in {"spectral", "rsi"}:
        raise ValueError("convolution_backend must be spectral, rsi or direct")
    elif pre_conv_max_rank is None and convolution_backend == "spectral":
        pre_conv_max_rank = _pre_conv_max_rank_for(deconv_rank)
    elif pre_conv_max_rank is not None and pre_conv_max_rank <= 0:
        pre_conv_max_rank = None
    t0 = time.time()
    print(f"\n=== [gl_gpu_noisy:{SCALE}] dim={dim}  naive_rank={rank}  deconv_rank={deconv_rank}  "
          f"degree={degree}  n_train={n_pts}  noise={noise_type}  sigma={sigma}  "
          f"naive_steps={naive_steps}  deconv_steps={deconv_steps}  pre_conv_max_rank={pre_conv_max_rank} ===")

    print(f"sampling clean data and adding {noise_type} noise (sigma={sigma})...")
    clean_train = sample_ginzburg_landau(seed=0, n=n_pts, d=dim)
    clean_test = sample_ginzburg_landau(seed=1, n=N_TEST, d=dim)
    rng = np.random.default_rng(0)
    noise_cov = None
    colored_stds = colored_rhos = None
    if noise_type == "isotropic":
        stds = NOISE_STDS * sigma
        comp = rng.choice(len(NOISE_WEIGHTS), size=n_pts, p=np.asarray(NOISE_WEIGHTS))
        noise = rng.normal(size=(n_pts, dim)) * np.asarray(stds)[comp][:, None]
    elif noise_type == "colored":
        colored_stds, colored_rhos = _colored_covariance_params(sigma, seed=0)
        noise = _sample_colored_noise(dim, sigma, n_pts, seed=0)
    else:
        noise_cov = _noniso_covariance(dim, sigma)
        noise = rng.multivariate_normal(mean=np.zeros(dim), cov=noise_cov, size=n_pts)
    noisy_train_full = jnp.asarray(np.asarray(clean_train) + noise)
    print(f"  done ({time.time() - t0:.1f}s elapsed)")

    print("computing exact log Z and true clean density on held-out points...")
    log_z = ginzburg_landau_log_normalizer(dim)
    y_true = np.asarray(ginzburg_landau_density(clean_test, log_z=log_z))
    print(f"  log_Z_true={log_z:.4f}")

    print("selecting KDE bandwidth via least-squares cross-validation...")
    bandwidth = select_gaussian_kde_bandwidth(noisy_train_full)
    print(f"  bandwidth[0]={float(bandwidth[0]):.4f}  (vs. Scott's rule, which badly oversmooths this "
          f"multimodal target -- see ftde.kde.select_gaussian_kde_bandwidth's docstring)")

    print("fitting Gaussian KDE baseline on the (full, untrimmed) noisy data...")
    kde = make_gaussian_kde(noisy_train_full, bandwidth=bandwidth)
    y_kde = np.asarray(kde(clean_test))
    err_kde = relative_l2_error(y_kde, y_true)
    print(f"  KDE baseline err={err_kde:.4f}")

    # A handful of points from the wide-tailed noise component (std=0.9) can
    # push the *raw* min/max of the noisy data far past where the true
    # signal actually has any mass -- fitting the squared-TT's domain to
    # that raw range wastes most of its fixed Chebyshev resolution on
    # regions that are essentially zero. Trimming to a percentile range
    # instead (dropping the most extreme ~1% of points) shrinks the domain
    # substantially with negligible data loss; see gl_common.robust_domain.
    print("computing a trimmed (percentile-based, not raw min/max) domain...")
    domain = robust_domain(noisy_train_full)
    noisy_train = filter_to_domain(noisy_train_full, domain)
    print(f"  domain[0]={domain[0]}  (vs. raw min/max would be wider)")

    if naive_steps <= 0:
        print("naive-steps<=0: skipping the naive (no-deconvolution) baseline entirely...")
        naive_fit, naive_hist, err_naive = None, [float("nan")], float("nan")
    else:
        print("cross-approximating sqrt(KDE) to initialize the naive squared-TT...")
        init = init_tt_from_kde(
            noisy_train, max_rank=rank, domain=domain, n_coeffs=(degree,), bandwidth=bandwidth,
            n_starts=N_STARTS, gd_steps=GD_STEPS,
        )
        print(f"  init TT ranks={init.ranks}  domain[0]={init.domain[0]}")

        print("probing naive-fit step cost...")
        naive_probe_fn = lambda **kw: fit_tt(init, noisy_train, lr=NAIVE_LR, batch_size=BATCH_SIZE, **kw)
        naive_sec_per_step = _probe_seconds_per_step(naive_probe_fn)
        print(f"  ~{naive_sec_per_step:.3f}s/step -> ETA {naive_sec_per_step * naive_steps / 60:.1f} min "
              f"for {naive_steps} steps")

        print("training naive (no-deconvolution) TT baseline on the noisy data...")
        naive_fit, naive_hist = fit_tt(
            init, noisy_train, n_steps=naive_steps, lr=NAIVE_LR, batch_size=BATCH_SIZE, log_every=LOG_EVERY,
        )
        y_naive = np.asarray(density_tt(naive_fit, clean_test))
        err_naive = relative_l2_error(y_naive, y_true)
        print(f"  naive TT: final_nll={naive_hist[-1]:.4f}  err={err_naive:.4f}")

    fit_start = time.time()
    print(f"cross-approximating sqrt(KDE) to initialize the (smaller-rank) deconv squared-TT "
          f"(rank={deconv_rank})...")
    deconv_init = init_tt_from_kde(
        noisy_train, max_rank=deconv_rank, domain=domain, n_coeffs=(degree,), bandwidth=bandwidth,
        n_starts=N_STARTS, gd_steps=GD_STEPS,
    )
    print(f"  deconv init TT ranks={deconv_init.ranks}  domain[0]={deconv_init.domain[0]}")

    if noise_type == "isotropic":
        print("building known Gaussian-mixture noise kernel U (isotropic)...")
        stds = NOISE_STDS * sigma
        u = build_gaussian_mixture_noise_kernel_tt(
            stds, NOISE_WEIGHTS, deconv_init.domain, dim, n_coeffs=KERNEL_N_COEFFS,
        )
        # The Tikhonov penalty wants a single per-mode noise std, but our
        # noise is a genuine mixture. A mixture's characteristic function is
        # dominated at high frequency by its narrowest component (the term
        # that decays slowest), so the narrowest component's std is the
        # conservative (safer, more-regularizing) choice.
        noise_std_per_mode = jnp.full((dim,), float(jnp.min(stds)))
    elif noise_type == "colored":
        print(f"building known pairwise-correlated Gaussian noise kernel U (colored, from the "
              f"average of {COLORED_N_COVARIANCES} sampled covariances)...")
        eff_std, eff_rho = _colored_effective_std_rho(colored_stds, colored_rhos)
        print(f"  sampled std range=[{colored_stds.min():.3f}, {colored_stds.max():.3f}]  "
              f"rho range=[{colored_rhos.min():.3f}, {colored_rhos.max():.3f}]  "
              f"-> averaged (std={eff_std:.3f}, rho={eff_rho:.3f})")
        u = build_pairwise_correlated_noise_kernel_tt(
            eff_std, eff_rho, deconv_init.domain, dim,
            max_rank=NONISO_KERNEL_RANK, n_starts=max(N_STARTS, 32), gd_steps=max(GD_STEPS, 8),
        )
        # Conservative choice, same logic as the isotropic mixture above:
        # the SMALLEST std among the 100 sampled covariances decays slowest
        # in frequency, so it dominates how much high-frequency information
        # actually survives across the whole heterogeneous population.
        noise_std_per_mode = jnp.full((dim,), float(colored_stds.min()))
    else:
        print("building known pairwise-correlated Gaussian noise kernel U (nonisotropic)...")
        # Unlike the search-free per-axis mixture/diagonal builders, this
        # does a genuine (if only 2D) cross-approximation search per pair,
        # which needs more starts/steps to avoid degenerating to a trivial
        # constant fit than this script's other (search-free) KDE-init
        # calls do -- confirmed empirically that the smoke-scale N_STARTS/
        # GD_STEPS (8/2) is NOT enough here and silently produces a
        # degree-1 kernel that then crashes downstream in convolve_tt.
        u = build_pairwise_correlated_noise_kernel_tt(
            _noniso_std(sigma), NONISO_RHO, deconv_init.domain, dim,
            max_rank=NONISO_KERNEL_RANK, n_starts=max(N_STARTS, 32), gd_steps=max(GD_STEPS, 8),
        )
        # Here the noise is a single (not mixture) Gaussian, so there's no
        # "narrowest component" conservatism needed -- just its own true
        # per-mode marginal std (the diagonal of its covariance).
        noise_std_per_mode = jnp.sqrt(jnp.diag(jnp.asarray(noise_cov)))
    print(f"  U ranks={u.ranks}  U domain[0]={u.domain[0]}  U degree={u.cores[0].coeffs.shape[-1]}")
    operator = TTConvolution.from_tt(deconv_init, u) if convolution_backend == "direct" else None
    if operator is not None:
        print("  direct TT-kernel contraction (no Gaussian assumption), quadrature orders:", operator.orders)
    kernel_rank = u.ranks[0] if u.ranks else 1
    unrounded_bond = deconv_rank**2 * kernel_rank
    effective_sq_rank = deconv_rank**2 if pre_conv_max_rank is None else min(deconv_rank**2, pre_conv_max_rank)
    rounded_bond = effective_sq_rank * kernel_rank
    print(f"  -> convolution combined bond rank per mode WITHOUT pre-conv rounding "
          f"~= deconv_rank^2 * kernel_rank = {deconv_rank}^2 * {kernel_rank} = {unrounded_bond}")
    if pre_conv_max_rank is not None:
        print(f"  -> WITH pre_conv_max_rank={pre_conv_max_rank}: combined bond rank "
              f"~= {effective_sq_rank} * {kernel_rank} = {rounded_bond}  "
              f"({unrounded_bond / max(rounded_bond, 1):.1f}x smaller)")

    # See REG_TARGET_BUDGET's comment above: a fixed reg_strength constant
    # is not safe once noise_std can vary (via --sigma/--noise), since the
    # Tikhonov penalty's own weight grows exponentially in noise_std. Solve
    # for the reg_strength that reproduces this project's previously-
    # validated effective budget at whatever this run's actual noise_std/
    # degree/domain are. square_tt roughly doubles the signal's own degree
    # (2*degree - 1), which is what the penalty actually applies to.
    # --reg-lambda then scales that self-calibrated budget up/down (default
    # 1.0 = the validated baseline) -- a direct multiplier on reg_strength
    # is not exposed on its own, since a raw reg_strength that was safe at
    # one noise_std/degree/domain is not safe at another (see
    # REG_TARGET_BUDGET's comment); scaling the BUDGET keeps that safety
    # while still being tunable.
    g_sq_degree = 2 * degree - 1
    max_tikhonov_weight = float(jnp.max(chebyshev_tikhonov_weights(
        g_sq_degree, (float(deconv_init.domain[0, 0]), float(deconv_init.domain[0, 1])),
        float(jnp.max(noise_std_per_mode)),
    )))
    reg_strength = reg_lambda * REG_TARGET_BUDGET / max(max_tikhonov_weight, 1e-300)
    print(f"  Tikhonov reg_strength={reg_strength:.3e}  (max per-coefficient weight={max_tikhonov_weight:.3e}, "
          f"targeting an effective budget of {reg_lambda} * {REG_TARGET_BUDGET} = {reg_lambda * REG_TARGET_BUDGET})")

    if operator is not None:
        print("probing direct deconv-fit step cost (contracts two signal cores and the noise TT "
              "without constructing or rounding the squared signal TT)...")
    else:
        print("probing spectral deconv-fit step cost (squaring the signal then convolving with U "
              "dominates the cost; pre_conv_max_rank rounds g^2 before convolving)...")
    deconv_probe_fn = lambda **kw: fit_noisy_tt(
        deconv_init, u, noisy_train, lr=DECONV_LR, batch_size=BATCH_SIZE,
        noise_std=noise_std_per_mode, reg_strength=reg_strength, pre_conv_max_rank=pre_conv_max_rank,
        forward_operator=operator, convolution_method="rsi" if convolution_backend == "rsi" else "naive",
        rsi_max_rank=rsi_max_rank, rsi_sketch_size=rsi_sketch_size, **kw,
    )
    deconv_sec_per_step = _probe_seconds_per_step(deconv_probe_fn)
    print(f"  ~{deconv_sec_per_step:.3f}s/step -> ETA {deconv_sec_per_step * deconv_steps / 60:.1f} min "
          f"for {deconv_steps} steps  (use --deconv-steps to change this)")

    print("training deconvolved (regularized) TT via noisy NLL minimization...")
    deconv_fit, deconv_hist = fit_noisy_tt(
        deconv_init, u, noisy_train, n_steps=deconv_steps, lr=DECONV_LR, batch_size=BATCH_SIZE,
        log_every=LOG_EVERY, noise_std=noise_std_per_mode, reg_strength=reg_strength,
        pre_conv_max_rank=pre_conv_max_rank, forward_operator=operator,
        convolution_method="rsi" if convolution_backend == "rsi" else "naive",
        rsi_max_rank=rsi_max_rank, rsi_sketch_size=rsi_sketch_size,
    )
    fit_seconds = time.time() - fit_start
    if any(np.isnan(deconv_hist)):
        print("  WARNING: deconv training hit NaN -- pre_conv_max_rank may be rounding too "
              "aggressively for this run (see _pre_conv_max_rank_for's comment); try raising "
              "--pre-conv-max-rank or disabling it with --pre-conv-max-rank 0.")
    y_deconv = np.asarray(density_tt(deconv_fit, clean_test))
    err_deconv = relative_l2_error(y_deconv, y_true)

    dt = time.time() - t0
    print(f"--- dim={dim} RESULT: naive_nll={naive_hist[-1]:.4f}  deconv_nll={deconv_hist[-1]:.4f}  "
          f"kde_err={err_kde:.4f}  naive_err={err_naive:.4f}  deconv_err={err_deconv:.4f}  "
          f"fit_seconds={fit_seconds:.1f}  ({dt:.1f}s total) ---")

    if save_figures:
        print("saving marginal figures...")
        for (i, j) in dim_pairs_for(dim):
            save_noisy_marginal_figure(
                dim, naive_fit, deconv_fit, u, clean_test, noisy_train, bandwidth, domain, i, j,
                log_z=log_z, kde_error=err_kde,
                naive_error=err_naive, deconv_error=err_deconv,
                naive_steps=naive_steps, deconv_steps=deconv_steps,
                n_density_points=n_pts, noise_type=noise_type,
                naive_rank=rank, deconv_rank=deconv_rank, degree=degree-1, sigma=sigma,
                fit_seconds=fit_seconds,
            )

    return {
        "naive_nll": naive_hist[-1], "deconv_nll": deconv_hist[-1],
        "kde_err": err_kde, "naive_err": err_naive, "deconv_err": err_deconv,
        "fit_seconds": fit_seconds,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("dims", nargs="*", type=int, default=None)
    parser.add_argument("--naive-steps", type=int_arg, default=None, help="override naive-fit steps (default 0)")
    parser.add_argument("--deconv-steps", type=int_arg, default=None, help="override deconv-fit steps (default 4000, 60 smoke)")
    parser.add_argument("--n-pts", type=int_arg, default=None, help="override number of observed points (default 20000, 300 smoke); accepts scientific notation, e.g. 1e5")
    parser.add_argument("--sigma", type=float, default=1.0, help="overall noise scale multiplier (default 1.0)")
    parser.add_argument(
        "--noise", choices=["isotropic", "nonisotropic", "colored"], default="nonisotropic",
        help="noise covariance structure: 'nonisotropic' (default, one correlated pairwise-block "
             "Gaussian), 'isotropic' (the old default: axis-aligned 2-component mixture), or "
             f"'colored' (heteroscedastic: each point's noise is drawn from its own covariance, "
             f"itself randomly sampled from {COLORED_N_COVARIANCES} distinct correlated "
             "covariances -- the deconvolution kernel approximates this with their average, "
             "same approach this project already uses for real per-point-varying cryo-EM noise)",
    )
    parser.add_argument(
        "--pre-conv-max-rank", type=int, default=None,
        help="round g^2 to this rank before convolving with U, cutting the deconv step's dominant "
             "cost (default: 2x the deconv signal rank, empirically stable -- see "
             "_pre_conv_max_rank_for's comment). Pass 0 to disable (exact, but rank^4-expensive) "
             "or a specific value to override; going all the way down to the signal's own rank was "
             "observed to produce much worse fits and NaN losses in testing, so be cautious below "
             "the default.",
    )
    parser.add_argument(
        "--reg-lambda", type=float, default=1.0,
        help="multiplier on the self-calibrated Tikhonov regularization budget (default 1.0 = "
             "this project's validated baseline). >1 regularizes harder (smoother, less overfit "
             "to noise, but risks underfitting real structure); <1 (or 0 to disable entirely) "
             "regularizes less (more capacity to fit real structure, but more exposed to "
             "deconvolution's classical ill-posedness -- see noisy_regularization.py).",
    )
    parser.add_argument("--rank", type=int, default=None, help="override the naive-fit TT rank (default depends on dim)")
    parser.add_argument("--deconv-rank", type=int, default=None, help="override the deconv-fit TT rank (default depends on dim, see _deconv_rank_for_dim)")
    parser.add_argument("--degree", type=int, default=None, help="override Chebyshev degree (default depends on dim)")
    parser.add_argument("--convolution-backend", choices=["rsi", "spectral", "direct"], default="rsi",
                        help="direct contracts the signal twice with the prescribed TT noise kernel, without rounding")
    parser.add_argument("--no-plots", action="store_true", help="evaluate a trial without replacing saved figures")
    parser.add_argument("--rsi-max-rank", type=int, default=128)
    parser.add_argument("--rsi-sketch-size", type=int, default=None)
    args = parser.parse_args()

    dims = args.dims or [MODERATE_DIM, HIGH_DIM]
    naive_steps = args.naive_steps if args.naive_steps is not None else DEFAULT_NAIVE_STEPS
    deconv_steps = args.deconv_steps if args.deconv_steps is not None else DEFAULT_DECONV_STEPS
    n_pts = args.n_pts if args.n_pts is not None else N_TRAIN
    results = {
        dim: run(
            dim, naive_steps, deconv_steps, n_pts, args.noise, args.sigma, args.pre_conv_max_rank,
            args.reg_lambda, rank=args.rank, deconv_rank=args.deconv_rank, degree=args.degree,
            convolution_backend=args.convolution_backend, save_figures=not args.no_plots,
            rsi_max_rank=args.rsi_max_rank, rsi_sketch_size=args.rsi_sketch_size,
        )
        for dim in dims
    }

    print("\n=== summary (noisy/deconvolution, squared relative L2 error vs. true Ginzburg-Landau density) ===")
    for dim, r in results.items():
        print(f"dim={dim:3d}  naive_nll={r['naive_nll']:.4f}  deconv_nll={r['deconv_nll']:.4f}  "
              f"kde_err={r['kde_err']:.4f}  naive_err={r['naive_err']:.4f}  deconv_err={r['deconv_err']:.4f}")
