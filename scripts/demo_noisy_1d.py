"""Noisy density estimation: recover a clean 1D mixture density from noisy
observations, using a known Gaussian noise kernel and the exact TT
convolution. Compares the deconvolved fit against the noisy density itself
(what you'd get from ignoring the noise) to check real deconvolution is
happening, not just curve-fitting the noisy KDE.
"""

import jax
import jax.numpy as jnp

from ftde.init import init_tt_from_kde
from ftde.noise_kernel import build_gaussian_noise_kernel_tt
from ftde.noisy_tt_density import fit_noisy_tt, log_density_noisy_tt
from ftde.tt_density import log_density_tt

key = jax.random.PRNGKey(0)
k1, k2, k3 = jax.random.split(key, 3)

# True clean distribution: a 1D two-component Gaussian mixture.
n = 6000
comp = jax.random.bernoulli(k1, 0.5, shape=(n,))
base = jax.random.normal(k2, (n,), dtype=jnp.float64)
clean = jnp.where(comp, base * 0.6 - 1.5, base * 0.6 + 1.5)[:, None]

# Moderate noise: comparable to, but smaller than, the signal's own spread.
# (At noise this large relative to the signal that MLE deconvolution starts
# fighting genuine statistical ill-posedness rather than easy machinery
# validation, we found noise_std ~0.5 is already past the "easy" regime.)
noise_std = 0.25
noise_cov = jnp.array([[noise_std**2]])
noisy = clean + noise_std * jax.random.normal(k3, (n, 1), dtype=jnp.float64)

print("fitting initial cross-approximation to sqrt(KDE(noisy data))...")
# Explicit, tight signal domain (rather than the automatic 6-sigma-of-noisy-data
# box) keeps the noise kernel's required domain -- and hence its required
# Chebyshev degree -- from ballooning; see the domain-sizing note in
# noise_kernel.gaussian_noise_domain.
#
# n_coeffs is a FIXED single rung (17), not an adaptive ladder: plain
# unregularized NLL deconvolution is a known ill-posed problem (many p_tilde
# shapes explain the noisy data similarly well), and letting the degree grow
# freely lets the optimizer find "wigglier" fits that explain the noisy
# marginal fine but recover the clean density worse. Capping the degree is a
# simple, standard implicit smoothness regularizer for this.
signal_domain = jnp.array([[-6.0, 6.0]])
g0 = init_tt_from_kde(noisy, max_rank=6, domain=signal_domain, n_coeffs=(17,), gd_steps=10, n_starts=32)
print("g0 domain:", g0.domain, " initial ranks:", g0.ranks)

print("building known Gaussian noise kernel U (needs high degree: it's a")
print("narrow bump on a domain forced wide by the safety condition)...")
U = build_gaussian_noise_kernel_tt(
    noise_cov, g0.domain, max_rank=1, n_coeffs=(33, 65, 129, 257), n_starts=8, gd_steps=0
)
print("U domain:", U.domain, " U degree:", U.cores[0].coeffs.shape[-1])

print("training via noisy NLL minimization (deconvolution)...")
g_fit, history = fit_noisy_tt(g0, U, noisy, n_steps=1200, lr=1e-2, log_every=200, max_grad_norm=50.0)

# True clean log-density (mixture of two N(+-1.5, 0.6^2)).
def true_clean_log_density(x):
    x = x[..., 0]
    log_a = -0.5 * jnp.log(2 * jnp.pi * 0.6**2) - 0.5 * ((x + 1.5) / 0.6) ** 2 - jnp.log(2.0)
    log_b = -0.5 * jnp.log(2 * jnp.pi * 0.6**2) - 0.5 * ((x - 1.5) / 0.6) ** 2 - jnp.log(2.0)
    m = jnp.maximum(log_a, log_b)
    return m + jnp.log(jnp.exp(log_a - m) + jnp.exp(log_b - m))


test_clean = jax.random.normal(jax.random.PRNGKey(42), (2000,), dtype=jnp.float64)
comp_t = jax.random.bernoulli(jax.random.PRNGKey(43), 0.5, shape=(2000,))
test_clean = jnp.where(comp_t, test_clean * 0.6 - 1.5, test_clean * 0.6 + 1.5)[:, None]

est_clean_logp = log_density_tt(g_fit, test_clean)  # deconvolved p_tilde = g^2/Z, evaluated on clean test points
true_clean_logp = true_clean_log_density(test_clean)
print("mean true clean log-lik:", float(jnp.mean(true_clean_logp)))
print("mean est  clean log-lik (deconvolved p_tilde):", float(jnp.mean(est_clean_logp)))
print("mean abs error (deconvolved vs true clean):", float(jnp.mean(jnp.abs(est_clean_logp - true_clean_logp))))

# Baseline: what if we'd ignored the noise entirely and just fit p to the
# noisy data directly (no deconvolution)? Should do noticeably worse on
# clean-data log-likelihood, showing the convolution/deconvolution mattered.
from ftde.tt_density import fit_tt

naive_fit, _ = fit_tt(g0, noisy, n_steps=1200, lr=1e-2, log_every=0, max_grad_norm=50.0)
naive_logp = log_density_tt(naive_fit, test_clean)
print("mean abs error (naive noisy-fit vs true clean):", float(jnp.mean(jnp.abs(naive_logp - true_clean_logp))))

# Sanity: fitted model's noisy log-lik on held-out noisy data.
test_noisy = clean[:2000] + noise_std * jax.random.normal(jax.random.PRNGKey(44), (2000, 1), dtype=jnp.float64)
noisy_ll = log_density_noisy_tt(g_fit, U, test_noisy)
print("mean noisy-model log-lik on held-out noisy data:", float(jnp.mean(noisy_ll)))
