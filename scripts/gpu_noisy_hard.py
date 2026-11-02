"""Harder noisy density estimation (deconvolution): a moderate-dimension
correlated-Gaussian signal corrupted by correlated Gaussian measurement
noise, recovered via exact TT convolution + NLL minimization.

This is meant for a GPU, not your laptop -- the noise kernel's domain must be
~3x the signal domain's width per mode (an inherent requirement of the exact,
closed-form, fixed-integration-window convolution; see chebdCUR's
`_convolution_safe_output_domain`), and resolving a comparatively narrow
noise kernel over that wide a domain needs high Chebyshev degree (a few
hundred coefficients per mode) -- exactly the kind of thing that blows up a
CPU-only machine (see the smoke-scale note below) but is comfortable on GPU.

--- How to run on the GPU cluster ---

See the README's "Install" section for the pip install recipe (order
matters: chebdCUR, then this repo, with any CUDA jax build installed last).
Then, with the GPU visible to jax:

    python scripts/gpu_noisy_hard.py    # full difficulty is the default

--- Local smoke check (CPU, tiny/fast, just to catch bugs) ---

    FTDE_TEST_SCALE=smoke python scripts/gpu_noisy_hard.py

Runs the same code path at trivially small dimension/rank/degree/steps so it
finishes in seconds on a CPU-only machine. This does NOT validate accuracy at
scale -- it only checks the code path is exception-free and qualitatively
sane.
"""

import os

import jax
import jax.numpy as jnp

from ftde.init import init_tt_from_kde
from ftde.noise_kernel import build_gaussian_noise_kernel_tt
from ftde.noisy_tt_density import fit_noisy_tt
from ftde.tt_density import fit_tt, log_density_tt

SCALE = os.environ.get("FTDE_TEST_SCALE", "full")
IS_SMOKE = SCALE == "smoke"


def _cfg(smoke_value, full_value):
    return smoke_value if IS_SMOKE else full_value


d = _cfg(2, 4)
n_train = _cfg(300, 20_000)
n_test = _cfg(200, 4_000)
signal_half_width = _cfg(3.0, 6.0)
signal_rank = _cfg(2, 10)
signal_n_coeffs = _cfg((9, 17), (17, 33, 65, 129))
noise_kernel_rank = _cfg(2, 6)
noise_kernel_n_coeffs = _cfg((9, 17), (33, 65, 129, 257))
n_starts = _cfg(16, 128)
gd_steps = _cfg(5, 40)
n_steps = _cfg(60, 4000)
batch_size = _cfg(None, 1024)

key = jax.random.PRNGKey(0)
k1, k2, k3, k4 = jax.random.split(key, 4)

# A correlated (non-diagonal) clean covariance and a correlated (non-diagonal,
# distinct) noise covariance -- so neither the signal nor the noise kernel is
# a trivially separable/rank-1 TT.
base_clean = jax.random.uniform(k1, (d, d), minval=-0.3, maxval=0.3, dtype=jnp.float64)
cov_clean = jnp.eye(d) + base_clean @ base_clean.T
base_noise = jax.random.uniform(k2, (d, d), minval=-0.2, maxval=0.2, dtype=jnp.float64)
cov_noise = 0.3 * (jnp.eye(d) + base_noise @ base_noise.T)

chol_clean = jnp.linalg.cholesky(cov_clean)
chol_noise = jnp.linalg.cholesky(cov_noise)

clean = jax.random.normal(k3, (n_train, d), dtype=jnp.float64) @ chol_clean.T
noisy = clean + jax.random.normal(k4, (n_train, d), dtype=jnp.float64) @ chol_noise.T

signal_domain = jnp.array([[-signal_half_width, signal_half_width]] * d)

print(f"[gpu_noisy_hard:{SCALE}] d={d}, n_train={n_train}")
print("fitting initial cross-approximation to sqrt(KDE(noisy data))...")
g0 = init_tt_from_kde(
    noisy,
    max_rank=signal_rank,
    domain=signal_domain,
    n_coeffs=signal_n_coeffs,
    n_starts=n_starts,
    gd_steps=gd_steps,
)
print("g0 ranks:", g0.ranks, " g0 domain:", g0.domain[0])

print("building known correlated-Gaussian noise kernel U...")
u = build_gaussian_noise_kernel_tt(
    cov_noise,
    g0.domain,
    max_rank=noise_kernel_rank,
    n_coeffs=noise_kernel_n_coeffs,
    n_starts=n_starts,
    gd_steps=gd_steps,
)
print("U ranks:", u.ranks, " U domain:", u.domain[0], " U degree:", u.cores[0].coeffs.shape[-1])

print("training via noisy NLL minimization (deconvolution)...")
g_fit, history = fit_noisy_tt(
    g0, u, noisy, n_steps=n_steps, lr=3e-3, batch_size=batch_size, log_every=max(n_steps // 5, 1)
)
assert jnp.isfinite(jnp.array(history[-1]))
assert history[-1] < history[0] + 1e-6

print("training naive (no-deconvolution) baseline on the same noisy data...")
naive_fit, _ = fit_tt(
    g0, noisy, n_steps=n_steps, lr=3e-3, batch_size=batch_size, log_every=0
)


def true_clean_log_density(x):
    inv = jnp.linalg.inv(cov_clean)
    log_det = jnp.linalg.slogdet(cov_clean)[1]
    quad = jnp.einsum("...i,ij,...j->...", x, inv, x)
    return -0.5 * (d * jnp.log(2.0 * jnp.pi) + log_det + quad)


test_clean = jax.random.normal(jax.random.PRNGKey(100), (n_test, d), dtype=jnp.float64) @ chol_clean.T
true_logp = true_clean_log_density(test_clean)
est_logp = log_density_tt(g_fit, test_clean)
naive_logp = log_density_tt(naive_fit, test_clean)

est_err = float(jnp.mean(jnp.abs(est_logp - true_logp)))
naive_err = float(jnp.mean(jnp.abs(naive_logp - true_logp)))
print(f"[gpu_noisy_hard:{SCALE}] mean|deconvolved - true clean logp| = {est_err:.4f}")
print(f"[gpu_noisy_hard:{SCALE}] mean|naive noisy-fit - true clean logp| = {naive_err:.4f}")
print(f"[gpu_noisy_hard:{SCALE}] deconvolution improvement factor: {naive_err / max(est_err, 1e-12):.2f}x")

assert est_err < _cfg(5.0, naive_err)
