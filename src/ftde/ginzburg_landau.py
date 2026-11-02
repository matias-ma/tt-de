"""The discretized Ginzburg-Landau field: a standard hard benchmark for
tensor-train density estimation (used e.g. in TT-cross sampling papers, and
in this project's own workshop poster's "10-dimensional Ginzburg-Landau"
experiment). Each coordinate has a double-well marginal (bistable around
+-1) coupled to its neighbors, giving a genuinely more complex, multi-modal
target than a plain Gaussian mixture -- unlike the real cryo-EM data tested
elsewhere in this project, here we know the target exactly, so we can
evaluate estimators against the true density directly.

Model and parametrization match Peng, Khoo & Wang, "Multivariate Density
Estimation via Variance-Reduced Sketching" (arXiv:2401.11646), Simulation
IV / eq. (26) exactly (not just "a" Ginzburg-Landau model -- this project's
own KDE baseline numbers were cross-checked directly against that paper's
own reported KDE baseline, see todo.txt), so results are directly
comparable to that paper's reported numbers (including their own
`variance-reduced-tensor-sketching`'s `main.py` convention of reporting
``||y_est - y_true||^2 / ||y_true||^2`` on held-out points, used here too).

Model: for x in R^d, with fixed (zero) boundary x_0 = x_{d+1} = 0,

    U(x) = beta * ( sum_{i=0}^{d} (lam/2) * ((x_{i+1}-x_i)/h)^2
                  + sum_{i=1}^{d} (1/(4*lam)) * (x_i^2-1)^2 )

    p(x) = exp(-U(x)) / Z

with the paper's own default coefficients ``beta=1/8``, ``lam=0.02``, and
``h=1/(d+1)`` (dimension-dependent -- unlike this project's own earlier,
non-paper-matching parametrization, the nearest-neighbor coupling strength
here grows with ``d``, since ``h`` shrinks). ``Z`` has no closed form, but
``U``'s nearest-neighbor chain structure means it factorizes into d+1
pairwise ("edge") terms and d single-site terms, so ``Z`` is exactly
computable (up to 1D grid quadrature error, not Monte Carlo sampling error)
via a standard transfer-operator recursion -- the same trick used to
compute partition functions of any 1D Markov-chain-structured Gibbs
distribution.
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np

GL_BETA = 1.0 / 8.0
GL_LAM = 0.02


def _default_h(d: int) -> float:
    return 1.0 / (d + 1)


def ginzburg_landau_log_prob(
    x: jnp.ndarray, beta: float = GL_BETA, lam: float = GL_LAM, h: float | None = None,
) -> jnp.ndarray:
    """Unnormalized log-density ``-U(x)`` for batched ``x`` of shape (..., d).
    ``h`` defaults to ``1/(d+1)`` (the paper's own convention) if not given.
    """
    x = jnp.asarray(x, dtype=jnp.float64)
    if h is None:
        h = _default_h(x.shape[-1])
    zero = jnp.zeros(x.shape[:-1] + (1,), dtype=x.dtype)
    x_padded = jnp.concatenate([zero, x, zero], axis=-1)  # (..., d+2)
    edge_diffs = (x_padded[..., 1:] - x_padded[..., :-1]) / h  # (..., d+1)
    kinetic = 0.5 * lam * jnp.sum(edge_diffs**2, axis=-1)
    site = jnp.sum((1.0 - x**2) ** 2, axis=-1) / (4.0 * lam)
    return -beta * (kinetic + site)


def ginzburg_landau_log_normalizer(
    d: int, beta: float = GL_BETA, lam: float = GL_LAM, h: float | None = None,
    *, n_grid: int = 2000, x_max: float = 3.0,
) -> float:
    """Exact (up to 1D grid quadrature error) log partition function ``log
    Z`` via the transfer-operator recursion, exploiting that ``U`` only
    couples nearest neighbors: writing the unnormalized density as a chain
    ``K(0,x_1) s(x_1) K(x_1,x_2) s(x_2) ... s(x_d) K(x_d,0)`` with edge
    kernel ``K(a,b) = exp(-beta*lam*(b-a)^2/(2*h^2))`` and site term
    ``s(x) = exp(-beta*(1-x^2)^2/(4*lam))``, ``Z`` is a sequence of d+1
    matrix-vector products on a 1D grid (trapezoidal quadrature) -- no
    sampling, no Monte Carlo, exact up to grid resolution.
    """
    if h is None:
        h = _default_h(d)
    grid = np.linspace(-x_max, x_max, n_grid)
    weights = np.full(n_grid, grid[1] - grid[0])
    weights[0] *= 0.5
    weights[-1] *= 0.5  # trapezoidal rule

    edge_kernel = np.exp(-beta * lam * (grid[:, None] - grid[None, :]) ** 2 / (2.0 * h**2))
    site_term = np.exp(-beta * (1.0 - grid**2) ** 2 / (4.0 * lam))

    # v[j] represents the accumulated (unnormalized) message reaching x_i = grid[j],
    # starting from the fixed left boundary x_0 = 0.
    # The fixed boundary is exactly zero, even when an even-sized grid has
    # no zero node. Selecting the nearest grid row shifts the boundary.
    boundary_kernel = np.exp(-beta * lam * grid**2 / (2.0 * h**2))
    v = boundary_kernel * site_term  # K(0, x_1) * s(x_1)
    for _ in range(d - 1):
        v = site_term * (edge_kernel @ (v * weights))
    z = float(np.sum(v * weights * boundary_kernel))  # close final edge K(x_d, 0)
    return float(np.log(z))


def ginzburg_landau_density(
    x: jnp.ndarray, beta: float = GL_BETA, lam: float = GL_LAM, h: float | None = None,
    *, log_z: float | None = None, d: int | None = None,
) -> jnp.ndarray:
    """Exact normalized density ``p(x) = exp(-U(x)) / Z``. Pass a
    precomputed ``log_z`` (from ``ginzburg_landau_log_normalizer``) to
    avoid recomputing it on every call; otherwise it is computed once here
    from ``d = x.shape[-1]``.
    """
    x = jnp.asarray(x, dtype=jnp.float64)
    if log_z is None:
        log_z = ginzburg_landau_log_normalizer(d if d is not None else x.shape[-1], beta, lam, h)
    return jnp.exp(ginzburg_landau_log_prob(x, beta, lam, h) - log_z)


def sample_ginzburg_landau(
    seed: int,
    n: int,
    d: int,
    *,
    beta: float = GL_BETA,
    lam: float = GL_LAM,
    h: float | None = None,
    proposal_std: float = 0.5,
    n_burnin: int = 200,
    thinning: int = 5,
    n_chains: int = 200,
) -> jnp.ndarray:
    """Single-coordinate (systematic-scan) Metropolis-Hastings sampler for
    the Ginzburg-Landau target. Runs ``n_chains`` parallel chains (plain
    numpy -- sampling doesn't need autodiff), discards ``n_burnin`` sweeps,
    then collects every ``thinning``-th sweep from each chain until at
    least ``n`` samples are gathered.

    A single full-vector random-walk proposal (propose all ``d``
    coordinates at once, accept/reject the whole move) suffers badly from
    the curse of dimensionality here: the chance that a simultaneous,
    equally-sized perturbation of every coordinate is jointly favorable
    collapses as ``d`` grows, so acceptance rate falls sharply with
    dimension (empirically ~25% at d=2 down to ~1% at d=10, and <=0.4% for
    a meaningful fraction of chains at d=8-10) -- some chains then barely
    move at all during the whole run, so the "thinned" samples they
    contribute are near-exact duplicates of each other, which silently
    inflates any downstream accuracy metric (e.g. relative L2 error against
    the true density) by effectively repeating a handful of correlated
    points with heavy weight. Since ``U`` only couples nearest neighbors, a
    systematic-scan sweep that proposes and accepts/rejects one coordinate
    at a time (a standard choice for chain/lattice models like this one)
    gives a per-coordinate acceptance rate of ~45% independent of ``d``
    (confirmed empirically at d=6, 10, 20), so mixing no longer degrades
    with dimension and burn-in/thinning can stay modest (measured in sweeps
    of ``d`` single-coordinate updates, not raw single-coordinate steps).
    """
    rng = np.random.default_rng(seed)
    if h is None:
        h = _default_h(d)

    def log_prob_np(x):
        return np.asarray(ginzburg_landau_log_prob(jnp.asarray(x), beta, lam, h))

    x = rng.normal(scale=0.5, size=(n_chains, d))
    logp = log_prob_np(x)

    def mh_sweep(x, logp):
        for j in range(d):
            proposal = x.copy()
            proposal[:, j] += rng.normal(scale=proposal_std, size=x.shape[0])
            proposal_logp = log_prob_np(proposal)
            accept = np.log(rng.uniform(size=x.shape[0])) < (proposal_logp - logp)
            x = np.where(accept[:, None], proposal, x)
            logp = np.where(accept, proposal_logp, logp)
        return x, logp

    for _ in range(n_burnin):
        x, logp = mh_sweep(x, logp)

    n_rounds = int(np.ceil(n / n_chains))
    samples = []
    for _ in range(n_rounds):
        for _ in range(thinning):
            x, logp = mh_sweep(x, logp)
        samples.append(x.copy())
    samples = np.concatenate(samples, axis=0)[:n]
    return jnp.asarray(samples)
