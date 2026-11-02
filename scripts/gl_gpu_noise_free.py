"""Noise-free Ginzburg-Landau density estimation at moderate and high
dimension: meant for a GPU, not a laptop.

Fits a squared-TT density to samples from the discretized Ginzburg-Landau
field (see src/ftde/ginzburg_landau.py -- a bistable, chained, genuinely
non-trivial target with a known exact density) via NLL minimization, and
reports squared relative L2 error against the true density on held-out points
(same convention as github.com/darenwang/variance-reduced-tensor-sketching:
``err = ||y_est - y_true||^2 / ||y_true||^2``), alongside a plain Gaussian
KDE baseline for reference (bandwidth chosen via least-squares cross-
validation, not the default Scott's rule -- see ftde.kde for why). Also
saves 2D marginal figures (true density, training histogram, KDE, fitted TT) to
figs/ginzburg_landau/.

--- How to run on the GPU cluster ---

See the README's "Install" section for the pip install recipe (order
matters: chebdCUR, then this repo, with any CUDA jax build installed last).
Then, with the GPU visible to jax:

    python scripts/gl_gpu_noise_free.py               # moderate (d=8) and high (d=20)
    python scripts/gl_gpu_noise_free.py 8              # only d=8
    python scripts/gl_gpu_noise_free.py 8 12 20        # custom dims
    python scripts/gl_gpu_noise_free.py 20 --steps 1000 # override training steps
    python scripts/gl_gpu_noise_free.py 20 --n-pts 5000 # override number of observed points

--- Local smoke check (CPU, tiny/fast, just to catch bugs) ---

    FTDE_TEST_SCALE=smoke python scripts/gl_gpu_noise_free.py

Runs the same code path at trivially small dimension/rank/degree/steps so it
finishes in well under a minute on a CPU-only machine. This does NOT
validate accuracy at scale -- it only checks the code path is
exception-free and qualitatively sane.
"""

from __future__ import annotations

import argparse
import os
import time

import numpy as np
from gl_common import (
    data_domain,
    dim_pairs_for,
    int_arg,
    relative_l2_error,
    save_noise_free_marginal_figure,
)

from ftde.ginzburg_landau import (
    ginzburg_landau_density,
    ginzburg_landau_log_normalizer,
    sample_ginzburg_landau,
)
from ftde.init import init_tt_from_kde
from ftde.kde import make_gaussian_kde, select_gaussian_kde_bandwidth
from ftde.tt_density import density_tt, fit_tt

SCALE = os.environ.get("FTDE_TEST_SCALE", "full")
IS_SMOKE = SCALE == "smoke"

MODERATE_DIM, HIGH_DIM = (3, 4) if IS_SMOKE else (8, 20)


def _rank_degree_for_dim(dim: int) -> tuple[int, int]:
    """Model capacity per dimension.

    Deliberately smoothed into 3 tiers rather than a single dim<=10 cutoff:
    a hard jump straight from (8, 17) to (14, 21) right at the d=10/d=12
    boundary means d=10 and d=12 aren't just "2 dimensions harder" -- they'd
    be using genuinely different model sizes, with the same fixed step
    budget potentially insufficient for the bigger model to converge as
    well. That confound is worth removing before concluding a big d=10->d=12
    accuracy jump is really about dimension and not an artifact of this
    schedule.
    """
    if IS_SMOKE:
        return 3, 9
    if dim <= 8:
        return 8, 17
    if dim <= 14:
        return 10, 19
    return 14, 21


N_TRAIN = 300 if IS_SMOKE else 20_000
N_TEST = 200 if IS_SMOKE else 4_000
N_STARTS = 8 if IS_SMOKE else 96
GD_STEPS = 2 if IS_SMOKE else 15
DEFAULT_N_STEPS = 60 if IS_SMOKE else 4000
LR = 1e-3
BATCH_SIZE = None if IS_SMOKE else 2048
LOG_EVERY = 100


def run(dim: int, n_steps: int, n_pts: int, rank: int | None = None, degree: int | None = None,
        save_figures: bool = True, extra_test_seed: int | None = None) -> dict[str, float]:
    default_rank, default_degree = _rank_degree_for_dim(dim)
    rank = rank if rank is not None else default_rank
    degree = degree if degree is not None else default_degree
    t0 = time.time()
    print(f"\n=== [gl_gpu_noise_free:{SCALE}] dim={dim}  rank={rank}  degree={degree} "
          f"n_train={n_pts}  n_steps={n_steps} ===")

    print("sampling clean train/test sets (single-coordinate MH sweeps)...")
    train = sample_ginzburg_landau(seed=0, n=n_pts, d=dim)
    test = sample_ginzburg_landau(seed=1, n=N_TEST, d=dim)
    print(f"  done ({time.time() - t0:.1f}s elapsed)")

    print("computing exact log Z via the transfer-operator recursion...")
    log_z = ginzburg_landau_log_normalizer(dim)
    y_true = np.asarray(ginzburg_landau_density(test, log_z=log_z))
    print(f"  log_Z_true={log_z:.4f}")

    domain = data_domain(train)

    print("selecting KDE bandwidth via least-squares cross-validation...")
    bandwidth = select_gaussian_kde_bandwidth(train)
    print(f"  bandwidth[0]={float(bandwidth[0]):.4f}  (vs. Scott's rule, which badly oversmooths this "
          f"multimodal target -- see ftde.kde.select_gaussian_kde_bandwidth's docstring)")

    print("fitting Gaussian KDE baseline...")
    kde = make_gaussian_kde(train, bandwidth=bandwidth)
    y_kde = np.asarray(kde(test))
    err_kde = relative_l2_error(y_kde, y_true)
    print(f"  KDE baseline err={err_kde:.4f}")

    print("cross-approximating sqrt(KDE) to initialize the squared-TT...")
    fit_start = time.time()
    init = init_tt_from_kde(
        train, max_rank=rank, domain=domain, n_coeffs=(degree,), bandwidth=bandwidth,
        n_starts=N_STARTS, gd_steps=GD_STEPS,
    )
    print(f"  init TT ranks={init.ranks}  domain[0]={init.domain[0]}")

    print("training via NLL minimization...")
    fit, history = fit_tt(
        init, train, n_steps=n_steps, lr=LR, batch_size=BATCH_SIZE, log_every=LOG_EVERY,
    )
    fit_seconds = time.time() - fit_start

    y_fit = np.asarray(density_tt(fit, test))
    err_fit = relative_l2_error(y_fit, y_true)
    extra_error = float("nan")
    if extra_test_seed is not None:
        extra_test = sample_ginzburg_landau(seed=extra_test_seed, n=N_TEST, d=dim)
        extra_truth = np.asarray(ginzburg_landau_density(extra_test, log_z=log_z))
        extra_error = relative_l2_error(np.asarray(density_tt(fit, extra_test)), extra_truth)
        print(f"  independent test seed={extra_test_seed}: tt_err={extra_error:.4f}")
    dt = time.time() - t0
    print(f"--- dim={dim} RESULT: final_nll={history[-1]:.4f}  "
          f"kde_err={err_kde:.4f}  tt_err={err_fit:.4f}  "
          f"fit_seconds={fit_seconds:.1f}  ({dt:.1f}s total) ---")

    if save_figures:
        print("saving marginal figures...")
        for (i, j) in dim_pairs_for(dim):
            save_noise_free_marginal_figure(
                dim, fit, test, train, bandwidth, domain, i, j, log_z=log_z,
                n_steps=n_steps, kde_error=err_kde, fit_error=err_fit,
                rank=rank, degree=degree-1, fit_seconds=fit_seconds,
            )

    return {"nll": history[-1], "kde_err": err_kde, "tt_err": err_fit,
            "fit_seconds": fit_seconds, "extra_tt_err": extra_error}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("dims", nargs="*", type=int, default=None)
    parser.add_argument("--steps", type=int_arg, default=None, help="override training steps (default 4000, 60 smoke)")
    parser.add_argument("--n-pts", type=int_arg, default=None, help="override number of observed points (default 20000, 300 smoke); accepts scientific notation, e.g. 1e5")
    parser.add_argument("--rank", type=int, default=None, help="override TT rank (default depends on dim, see _rank_degree_for_dim)")
    parser.add_argument("--degree", type=int, default=None, help="override Chebyshev degree (default depends on dim)")
    parser.add_argument("--no-plots", action="store_true", help="evaluate a trial without replacing saved figures")
    parser.add_argument("--extra-test-seed", type=int, default=None,
                        help="also report error on an independent held-out sample")
    args = parser.parse_args()

    dims = args.dims or [MODERATE_DIM, HIGH_DIM]
    n_steps = args.steps if args.steps is not None else DEFAULT_N_STEPS
    n_pts = args.n_pts if args.n_pts is not None else N_TRAIN
    results = {dim: run(dim, n_steps, n_pts, rank=args.rank, degree=args.degree,
                        save_figures=not args.no_plots, extra_test_seed=args.extra_test_seed)
               for dim in dims}

    print("\n=== summary (noise-free, squared relative L2 error vs. true Ginzburg-Landau density) ===")
    for dim, r in results.items():
        print(f"dim={dim:3d}  final_nll={r['nll']:.4f}  kde_err={r['kde_err']:.4f}  tt_err={r['tt_err']:.4f}")
