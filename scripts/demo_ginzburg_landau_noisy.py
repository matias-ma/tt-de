"""Naive KDE vs. naive squared-TT vs. deconvolved (regularized) squared-TT
on a synthetic but genuinely non-trivial target: the discretized
Ginzburg-Landau field (bistable per coordinate, chained across dimensions
-- see src/ftde/ginzburg_landau.py), corrupted by a substantially noisier
kernel than the real cryo-EM data used elsewhere in this project (there,
noise was only ~4-11% of the observed variance; here it's ~30%).

Unlike the cryo-EM experiments, here the TRUE clean density is known
exactly (up to 1D grid-quadrature error in its normalizing constant, via
the transfer-operator method -- not Monte Carlo), so we can evaluate
estimators directly against ground truth, using the same convention as
github.com/darenwang/variance-reduced-tensor-sketching's own benchmark:
``err = ||y_est - y_true||^2 / ||y_true||^2`` on held-out points drawn
fresh from the true distribution.

Usage: python scripts/demo_ginzburg_landau_noisy.py [dim ...]
  dim: one or more dimensions to test (default: 2 4 6 8)
"""

from __future__ import annotations

import sys

import jax.numpy as jnp
import numpy as np

from ftde.ginzburg_landau import (
    ginzburg_landau_density,
    ginzburg_landau_log_normalizer,
    sample_ginzburg_landau,
)
from ftde.init import init_tt_from_kde
from ftde.kde import make_gaussian_kde
from ftde.noise_kernel import build_gaussian_mixture_noise_kernel_tt
from ftde.noisy_tt_density import fit_noisy_tt
from ftde.tt_density import density_tt, fit_tt

# Ginzburg-Landau target parameters live as defaults on the model functions
# themselves (ftde.ginzburg_landau.GL_BETA/GL_LAM, h=1/(d+1)), matching
# Peng, Khoo & Wang (arXiv:2401.11646) Simulation IV exactly.

# Noise: a 2-component Gaussian mixture, deliberately much noisier relative
# to the signal than the real cryo-EM data (there, noise was ~4-11% of
# total variance; here, sum(w_k * std_k^2) / GL's own variance (~0.6-0.85)
# is ~30%).
NOISE_STDS = jnp.array([0.3, 0.9])
NOISE_WEIGHTS = jnp.array([0.7, 0.3])

N_TRAIN, N_TEST = 15000, 3000
SIGNAL_RANK, SIGNAL_DEGREE = 4, 13
N_STARTS, GD_STEPS = 64, 10
BATCH_SIZE = 2048
# Let the kernel builder resolve the narrowest mixture component on its
# expanded convolution domain. A fixed 129 is insufficient for some signal
# domains and silently corrupts the forward likelihood.
KERNEL_N_COEFFS = None
NAIVE_STEPS, NAIVE_LR = 1500, 1e-3
DECONV_STEPS, DECONV_LR, REG_STRENGTH = 3000, 1e-4, 1e-5


def relative_l2_error(y_est: np.ndarray, y_true: np.ndarray) -> float:
    """Same convention as github.com/darenwang/variance-reduced-tensor-sketching:
    ||y_est - y_true||_2^2 / ||y_true||_2^2 on held-out points.
    """
    return float(np.sum((y_est - y_true) ** 2) / np.sum(y_true**2))


def data_domain(points: jnp.ndarray, margin_frac: float = 0.1) -> jnp.ndarray:
    lo = jnp.min(points, axis=0)
    hi = jnp.max(points, axis=0)
    margin = margin_frac * (hi - lo)
    return jnp.stack([lo - margin, hi + margin], axis=1)


def run(dim: int) -> dict[str, float]:
    print(f"=== Ginzburg-Landau noisy density estimation: dim={dim} ===")

    clean_train = sample_ginzburg_landau(seed=0, n=N_TRAIN, d=dim)
    clean_test = sample_ginzburg_landau(seed=1, n=N_TEST, d=dim)

    rng = np.random.default_rng(0)
    comp = rng.choice(len(NOISE_WEIGHTS), size=N_TRAIN, p=np.asarray(NOISE_WEIGHTS))
    noise = rng.normal(size=(N_TRAIN, dim)) * np.asarray(NOISE_STDS)[comp][:, None]
    noisy_train = jnp.asarray(np.asarray(clean_train) + noise)

    log_z = ginzburg_landau_log_normalizer(dim)
    y_true = np.asarray(ginzburg_landau_density(clean_test, log_z=log_z))
    print(f"n_train={N_TRAIN}  n_test={N_TEST}  log_Z_true={log_z:.4f}")

    domain = data_domain(noisy_train)

    kde = make_gaussian_kde(noisy_train)
    y_kde = np.asarray(kde(clean_test))
    err_kde = relative_l2_error(y_kde, y_true)
    print(f"KDE            err={err_kde:.4f}")

    init = init_tt_from_kde(
        noisy_train, max_rank=SIGNAL_RANK, domain=domain, n_coeffs=(SIGNAL_DEGREE,),
        n_starts=N_STARTS, gd_steps=GD_STEPS,
    )
    naive_fit, naive_hist = fit_tt(
        init, noisy_train, n_steps=NAIVE_STEPS, lr=NAIVE_LR, batch_size=BATCH_SIZE, log_every=NAIVE_STEPS,
    )
    y_naive = np.asarray(density_tt(naive_fit, clean_test))
    err_naive = relative_l2_error(y_naive, y_true)
    print(f"naive TT (no deconv)  nll={naive_hist[-1]:.4f}  err={err_naive:.4f}")

    u = build_gaussian_mixture_noise_kernel_tt(
        NOISE_STDS, NOISE_WEIGHTS, init.domain, dim, n_coeffs=KERNEL_N_COEFFS,
    )
    # The frequency-aware Tikhonov penalty (noisy_regularization.py) wants a
    # single per-mode noise std, but our noise is a genuine mixture. A
    # mixture's characteristic function sum_k w_k*exp(-std_k^2*omega^2/2) is
    # dominated at high frequency by its NARROWEST component (the term that
    # decays slowest), so the narrowest component's std is the conservative
    # (safer, more-regularizing) choice -- using the mixture's overall RMS
    # std here would understate how much information survives at high
    # degree and under-regularize.
    noise_std_per_mode = jnp.full((dim,), float(jnp.min(NOISE_STDS)))
    deconv_fit, deconv_hist = fit_noisy_tt(
        init, u, noisy_train, n_steps=DECONV_STEPS, lr=DECONV_LR, batch_size=BATCH_SIZE,
        log_every=DECONV_STEPS, noise_std=noise_std_per_mode, reg_strength=REG_STRENGTH,
    )
    y_deconv = np.asarray(density_tt(deconv_fit, clean_test))
    err_deconv = relative_l2_error(y_deconv, y_true)
    print(f"deconv TT (regularized)  nll={deconv_hist[-1]:.4f}  err={err_deconv:.4f}")

    return {"kde": err_kde, "naive": err_naive, "deconv": err_deconv}


if __name__ == "__main__":
    dims = [int(a) for a in sys.argv[1:]] or [2, 4, 6, 8]
    results = {dim: run(dim) for dim in dims}

    print("\n=== summary (relative L2 error vs. true Ginzburg-Landau density) ===")
    for dim, errs in results.items():
        print(f"dim={dim:3d}  " + "  ".join(f"{k}={v:.4f}" for k, v in errs.items()))
