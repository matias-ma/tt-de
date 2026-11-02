"""Squared functional-TT density estimator (noise-free).

We parametrize an unnormalized square root ``g`` of the density as a
Chebyshev functional tensor train (shared coefficient-tensor cores), and take

    p(x) = g(x)^2 / Z,      Z = integral g(x)^2 dx,

so that ``p`` is guaranteed non-negative for any choice of the (real) TT
coefficients. ``Z`` is computed exactly in closed form by contracting the
TT with itself through per-mode Chebyshev Gram matrices -- no quadrature,
no Monte Carlo.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import optax
from chebttge.chebtt import ChebTT
from chebttge.cores import ChebyshevCoeffCore

from .cheb_gram import chebyshev_gram_matrix

TTParams = list[jnp.ndarray]


def tt_params(tt: ChebTT) -> TTParams:
    """Extract the trainable Chebyshev coefficient arrays of a ChebTT."""
    return [core.coeffs for core in tt.cores]


def tt_domains(tt: ChebTT) -> list[tuple[float, float]]:
    return [core.domain for core in tt.cores]


def rebuild_tt(template: ChebTT, params: TTParams) -> ChebTT:
    """Reconstruct a ChebTT with new coefficients, same domains/structure."""
    cores = tuple(
        ChebyshevCoeffCore(p, d) for p, d in zip(params, tt_domains(template), strict=True)
    )
    return ChebTT(cores=cores, domain=template.domain, ranks=template.ranks)


def squared_integral_tt(tt: ChebTT) -> jnp.ndarray:
    """Exact ``integral g(x)^2 dx`` for a coefficient-tensor ChebTT ``g``."""
    state = jnp.ones((1, 1), dtype=jnp.float64)
    for core in tt.cores:
        coeffs = core.coeffs  # (r_left, r_right, n_coeffs)
        n_coeffs = coeffs.shape[-1]
        gram = chebyshev_gram_matrix(n_coeffs, core.domain)
        core_gram = jnp.einsum("pkm,nm->pkn", coeffs, gram)
        state = jnp.einsum("pq,pkm,qlm->kl", state, core_gram, coeffs)
    return state[0, 0]


def log_density_tt(tt: ChebTT, x: jnp.ndarray, eps: float = 1e-300) -> jnp.ndarray:
    """log p(x) for batched ``x`` of shape (batch, d) or (d,)."""
    g = tt(x)
    z = squared_integral_tt(tt)
    return jnp.log(g**2 + eps) - jnp.log(z)


def density_tt(tt: ChebTT, x: jnp.ndarray) -> jnp.ndarray:
    return jnp.exp(log_density_tt(tt, x))


def nll_tt(params: TTParams, template: ChebTT, x: jnp.ndarray) -> jnp.ndarray:
    tt = rebuild_tt(template, params)
    return -jnp.mean(log_density_tt(tt, x))


def make_train_step(template: ChebTT, optimizer: optax.GradientTransformation):
    def loss_fn(params: TTParams, x: jnp.ndarray) -> jnp.ndarray:
        return nll_tt(params, template, x)

    loss_and_grad = jax.jit(jax.value_and_grad(loss_fn))

    @jax.jit
    def step(params: TTParams, opt_state, x: jnp.ndarray):
        loss, grads = loss_and_grad(params, x)
        updates, opt_state = optimizer.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)
        return params, opt_state, loss

    return step


def fit_tt(
    init_tt: ChebTT,
    x: jnp.ndarray,
    *,
    n_steps: int = 2000,
    lr: float = 3e-3,
    batch_size: int | None = None,
    seed: int = 0,
    log_every: int = 200,
    max_grad_norm: float = 1.0,
) -> tuple[ChebTT, list[float]]:
    """Minimize the NLL of the squared-TT density on samples ``x`` (n, d).

    Gradient-norm clipping guards against optimizer divergence when the
    cross-approximation initialization is coarse (e.g. a low-quality/cheap
    initial fit) and Adam's fixed step size would otherwise overshoot.
    """
    optimizer = optax.chain(optax.clip_by_global_norm(max_grad_norm), optax.adam(lr))
    params = tt_params(init_tt)
    opt_state = optimizer.init(params)
    step = make_train_step(init_tt, optimizer)

    n = x.shape[0]
    key = jax.random.PRNGKey(seed)
    history: list[float] = []
    for i in range(n_steps):
        if batch_size is None or batch_size >= n:
            batch = x
        else:
            key, sub = jax.random.split(key)
            idx = jax.random.randint(sub, (batch_size,), 0, n)
            batch = x[idx]
        params, opt_state, loss = step(params, opt_state, batch)
        history.append(float(loss))
        if log_every and (i % log_every == 0 or i == n_steps - 1):
            print(f"[fit_tt] step {i:5d}  nll {float(loss): .6f}")

    fitted = rebuild_tt(init_tt, params)
    return fitted, history
