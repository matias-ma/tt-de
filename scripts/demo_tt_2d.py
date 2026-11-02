"""Fit a squared-TT density to a 2D correlated Gaussian and check accuracy."""

import jax
import jax.numpy as jnp

from ftde.data import sample_correlated_gaussian
from ftde.init import init_tt_from_kde
from ftde.tt_density import density_tt, fit_tt, squared_integral_tt

key = jax.random.PRNGKey(0)
k1, k2 = jax.random.split(key)
data = sample_correlated_gaussian(k1, n=4000, d=2, rho=0.85)

print("fitting initial cross-approximation to sqrt(KDE)...")
init = init_tt_from_kde(data, max_rank=6, n_coeffs=(17, 33), gd_steps=30, n_starts=64)
print("initial TT ranks:", init.ranks)
print("initial squared integral (Z):", float(squared_integral_tt(init)))

print("training via NLL minimization...")
fitted, history = fit_tt(init, data, n_steps=800, lr=5e-3, log_every=100)

print("final unnormalized squared integral Z of g (log_density divides by this):",
      float(squared_integral_tt(fitted)))

# Compare against the true density on a held-out sample and via the analytic formula.
cov = jnp.array([[1.0, 0.85], [0.85, 1.0]])
inv_cov = jnp.linalg.inv(cov)
log_det = jnp.linalg.slogdet(cov)[1]


def true_log_density(x):
    diff = x
    quad = jnp.einsum("...i,ij,...j->...", diff, inv_cov, diff)
    return -0.5 * (2 * jnp.log(2 * jnp.pi) + log_det + quad)


test = sample_correlated_gaussian(k2, n=2000, d=2, rho=0.85)
from ftde.tt_density import log_density_tt

est_logp = log_density_tt(fitted, test)
true_logp = true_log_density(test)

print("mean true log-lik:", float(jnp.mean(true_logp)))
print("mean est  log-lik:", float(jnp.mean(est_logp)))
print("mean abs log-density error on test set:", float(jnp.mean(jnp.abs(est_logp - true_logp))))

# grid check of normalization via brute-force quadrature (2D midpoint rule)
lo = jnp.min(data, axis=0) - 6.0
hi = jnp.max(data, axis=0) + 6.0
n_grid = 400
xs = jnp.linspace(lo[0], hi[0], n_grid)
ys = jnp.linspace(lo[1], hi[1], n_grid)
X, Y = jnp.meshgrid(xs, ys, indexing="ij")
grid = jnp.stack([X.ravel(), Y.ravel()], axis=1)
dens = density_tt(fitted, grid)
cell_area = (hi[0] - lo[0]) / n_grid * (hi[1] - lo[1]) / n_grid
mass = float(jnp.sum(dens) * cell_area)
print("numerically-integrated total mass of fitted density (should be ~1):", mass)
