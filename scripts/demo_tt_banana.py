"""Fit a squared-TT density to the 2D banana distribution (non-separable)."""

import jax
import jax.numpy as jnp

from ftde.data import sample_banana
from ftde.init import init_tt_from_kde
from ftde.kde import make_gaussian_kde
from ftde.tt_density import density_tt, fit_tt, log_density_tt

key = jax.random.PRNGKey(1)
k1, k2 = jax.random.split(key)
data = sample_banana(k1, n=3000, curvature=0.2)

print("fitting initial cross-approximation to sqrt(KDE)...")
init = init_tt_from_kde(data, max_rank=6, n_coeffs=(17, 33), gd_steps=10, n_starts=32)
print("initial TT ranks:", init.ranks)

print("training via NLL minimization...")
fitted, history = fit_tt(init, data, n_steps=600, lr=5e-3, log_every=100)

test = sample_banana(k2, n=1500, curvature=0.2)
est_logp = log_density_tt(fitted, test)
print("mean estimated log-lik on held-out test data:", float(jnp.mean(est_logp)))

# Reference: a large-N held-out KDE evaluated at the test points (not the same
# KDE bandwidth used for init, an independent large-sample cross-check).
ref_kde = make_gaussian_kde(sample_banana(jax.random.PRNGKey(99), n=20000, curvature=0.2))
ref_logp = jnp.log(ref_kde(test))
print("mean reference (large-N KDE) log-lik on same test data:", float(jnp.mean(ref_logp)))
print("mean abs difference (est - ref KDE):", float(jnp.mean(jnp.abs(est_logp - ref_logp))))

lo = jnp.min(data, axis=0) - 6.0
hi = jnp.max(data, axis=0) + 6.0
n_grid = 200
xs = jnp.linspace(lo[0], hi[0], n_grid)
ys = jnp.linspace(lo[1], hi[1], n_grid)
X, Y = jnp.meshgrid(xs, ys, indexing="ij")
grid = jnp.stack([X.ravel(), Y.ravel()], axis=1)
dens = density_tt(fitted, grid)
cell_area = (hi[0] - lo[0]) / n_grid * (hi[1] - lo[1]) / n_grid
mass = float(jnp.sum(dens) * cell_area)
print("numerically-integrated total mass of fitted density (should be ~1):", mass)
print("min density on grid (should be >= 0):", float(jnp.min(dens)))
