"""Noisy density estimation, Tucker version -- see noisy_tt_density.py."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import optax
from chebtuckerge import convolve_tucker
from chebtuckerge.chebtucker import ChebTucker

from .cheb_product import square_tucker
from .noisy_regularization import tikhonov_penalty_tucker
from .tucker_density import (
    TuckerParams,
    rebuild_tucker,
    squared_integral_tucker,
    tucker_params,
)


def log_density_noisy_tucker(g: ChebTucker, u: ChebTucker | None, y: jnp.ndarray, eps: float = 1e-300, *, forward_operator=None) -> jnp.ndarray:
    """Convolved squared density; a direct operator avoids the explicit square."""
    if forward_operator is not None:
        return forward_operator.log_density(g, y)
    if u is None:
        raise ValueError('Supply a kernel or forward_operator')
    g_sq = square_tucker(g)
    q_unnorm = convolve_tucker(g_sq, u)
    z = squared_integral_tucker(g)
    return jnp.log(jnp.maximum(q_unnorm(y), eps)) - jnp.log(z)


def density_noisy_tucker(g: ChebTucker, u: ChebTucker | None, y: jnp.ndarray, *, forward_operator=None) -> jnp.ndarray:
    return jnp.exp(log_density_noisy_tucker(g, u, y, forward_operator=forward_operator))


def nll_noisy_tucker(
    params: TuckerParams,
    template: ChebTucker,
    u: ChebTucker | None,
    y: jnp.ndarray,
    *,
    noise_std: jnp.ndarray | None = None,
    reg_strength: float = 0.0,
    per_axis_target_budget: float | None = None,
    forward_operator=None,
) -> jnp.ndarray:
    """See ``noisy_tt_density.nll_noisy_tt`` -- same frequency-aware Tikhonov
    penalty (and same ``per_axis_target_budget`` option), applied to the
    Tucker factors instead of TT cores.
    """
    g = rebuild_tucker(template, params)
    nll = -jnp.mean(log_density_noisy_tucker(g, u, y, forward_operator=forward_operator))
    if reg_strength:
        nll = nll + reg_strength * tikhonov_penalty_tucker(
            square_tucker(g), noise_std, per_axis_target_budget=per_axis_target_budget,
        )
    return nll


def make_noisy_train_step(
    template: ChebTucker,
    u: ChebTucker | None,
    optimizer: optax.GradientTransformation,
    *,
    noise_std: jnp.ndarray | None = None,
    reg_strength: float = 0.0,
    per_axis_target_budget: float | None = None,
    forward_operator=None,
):
    def loss_fn(params: TuckerParams, y: jnp.ndarray) -> jnp.ndarray:
        return nll_noisy_tucker(
            params, template, u, y, noise_std=noise_std, reg_strength=reg_strength,
            per_axis_target_budget=per_axis_target_budget, forward_operator=forward_operator,
        )

    loss_and_grad = jax.jit(jax.value_and_grad(loss_fn))

    @jax.jit
    def step(params: TuckerParams, opt_state, y: jnp.ndarray):
        loss, grads = loss_and_grad(params, y)
        updates, opt_state = optimizer.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)
        return params, opt_state, loss

    return step


def fit_noisy_tucker(
    init_g: ChebTucker,
    u: ChebTucker | None,
    noisy_y: jnp.ndarray,
    *,
    n_steps: int = 2000,
    lr: float = 3e-3,
    batch_size: int | None = None,
    seed: int = 0,
    log_every: int = 200,
    max_grad_norm: float = 1.0,
    noise_std: jnp.ndarray | None = None,
    reg_strength: float = 0.0,
    per_axis_target_budget: float | None = None,
    forward_operator=None,
) -> tuple[ChebTucker, list[float]]:
    """See ``noisy_tt_density.fit_noisy_tt`` -- same ``reg_strength``/``noise_std``/
    ``per_axis_target_budget`` frequency-aware Tikhonov penalty.
    """
    optimizer = optax.chain(optax.clip_by_global_norm(max_grad_norm), optax.adam(lr))
    params = tucker_params(init_g)
    opt_state = optimizer.init(params)
    step = make_noisy_train_step(
        init_g, u, optimizer, noise_std=noise_std, reg_strength=reg_strength,
        per_axis_target_budget=per_axis_target_budget, forward_operator=forward_operator,
    )

    n = noisy_y.shape[0]
    key = jax.random.PRNGKey(seed)
    history: list[float] = []
    for i in range(n_steps):
        if batch_size is None or batch_size >= n:
            batch = noisy_y
        else:
            key, sub = jax.random.split(key)
            idx = jax.random.randint(sub, (batch_size,), 0, n)
            batch = noisy_y[idx]
        params, opt_state, loss = step(params, opt_state, batch)
        history.append(float(loss))
        if log_every and (i % log_every == 0 or i == n_steps - 1):
            print(f"[fit_noisy_tucker] step {i:5d}  nll {float(loss): .6f}")

    fitted = rebuild_tucker(init_g, params)
    return fitted, history
