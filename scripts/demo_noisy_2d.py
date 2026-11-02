"""Noisy density estimation in 2D: recover a correlated-Gaussian clean
density from noisy observations under known (isotropic) Gaussian noise.
"""

import jax
import jax.numpy as jnp

from ftde.init import init_tt_from_kde
from ftde.noise_kernel import build_gaussian_noise_kernel_tt
from ftde.noisy_tt_density import fit_noisy_tt
from ftde.tt_density import fit_tt, log_density_tt

key = jax.random.PRNGKey(1)
k1, k2 = jax.random.split(key)

rho = 0.6
cov_clean = jnp.array([[1.0, rho], [rho, 1.0]])
n = 6000
chol = jnp.linalg.cholesky(cov_clean)
clean = jax.random.normal(k1, (n, 2), dtype=jnp.float64) @ chol.T

noise_std = 0.35
noise_cov = jnp.eye(2) * noise_std**2
noisy = clean + noise_std * jax.random.normal(k2, (n, 2), dtype=jnp.float64)

signal_domain = jnp.array([[-6.0, 6.0], [-6.0, 6.0]])
print("fitting initial cross-approximation to sqrt(KDE(noisy data))...")
g0 = init_tt_from_kde(noisy, max_rank=4, domain=signal_domain, n_coeffs=(13,), gd_steps=10, n_starts=32)
print("g0 ranks:", g0.ranks)

print("building known Gaussian noise kernel U...")
# n_starts/gd_steps must be generous enough for the cross-approximation pivot
# search even in 2D+ -- too few (e.g. 8) can land on a degenerate pivot and
# silently converge to a near-constant (wrong) per-mode function.
U = build_gaussian_noise_kernel_tt(noise_cov, g0.domain, max_rank=1, n_coeffs=(33, 65, 129), n_starts=64, gd_steps=10)
print("U domain:", U.domain, " U core degrees:", [c.coeffs.shape[-1] for c in U.cores])

print("training via noisy NLL minimization (deconvolution)...")
g_fit, history = fit_noisy_tt(g0, U, noisy, n_steps=1000, lr=1e-2, log_every=200, max_grad_norm=50.0)

print("training naive (no-deconvolution) baseline on the same noisy data...")
naive_fit, _ = fit_tt(g0, noisy, n_steps=1000, lr=1e-2, log_every=0, max_grad_norm=50.0)


def true_clean_log_density(x):
    inv = jnp.linalg.inv(cov_clean)
    log_det = jnp.linalg.slogdet(cov_clean)[1]
    quad = jnp.einsum("...i,ij,...j->...", x, inv, x)
    return -0.5 * (2 * jnp.log(2 * jnp.pi) + log_det + quad)


test_clean = jax.random.normal(jax.random.PRNGKey(7), (2000, 2), dtype=jnp.float64) @ chol.T
true_logp = true_clean_log_density(test_clean)
est_logp = log_density_tt(g_fit, test_clean)
naive_logp = log_density_tt(naive_fit, test_clean)

print("mean true clean log-lik:", float(jnp.mean(true_logp)))
print("mean abs error (deconvolved vs true clean):", float(jnp.mean(jnp.abs(est_logp - true_logp))))
print("mean abs error (naive noisy-fit vs true clean):", float(jnp.mean(jnp.abs(naive_logp - true_logp))))
