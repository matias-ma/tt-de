"""Squared functional-Tucker density estimator (noise-free), see tt_density.py."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import optax
from chebtuckerge.chebtucker import ChebTucker
from chebtuckerge.factors import ChebyshevCoeffFactor

from .cheb_gram import chebyshev_gram_matrix, chebyshev_moment_vector

TuckerParams = tuple[jnp.ndarray, list[jnp.ndarray]]


def tucker_params(t: ChebTucker) -> TuckerParams:
    factor_coeffs = [factor.core.coeffs[0] for factor in t.factors]
    return t.core, factor_coeffs


def tucker_domains(t: ChebTucker) -> list[tuple[float, float]]:
    return [factor.domain for factor in t.factors]


def rebuild_tucker(template: ChebTucker, params: TuckerParams) -> ChebTucker:
    core, factor_coeffs = params
    factors = tuple(
        ChebyshevCoeffFactor(c, d)
        for c, d in zip(factor_coeffs, tucker_domains(template), strict=True)
    )
    return ChebTucker(core=core, factors=factors, domain=template.domain)


def _mode_product(tensor: jnp.ndarray, matrix: jnp.ndarray, mode: int) -> jnp.ndarray:
    tensor = jnp.moveaxis(tensor, mode, 0)
    out = jnp.tensordot(matrix, tensor, axes=([1], [0]))
    return jnp.moveaxis(out, 0, mode)


def squared_integral_tucker(t: ChebTucker) -> jnp.ndarray:
    """Exact ``integral g(x)^2 dx`` for a coefficient-tensor ChebTucker ``g``."""
    a = t.core
    for mode, factor in enumerate(t.factors):
        coeffs = factor.core.coeffs[0]  # (rank, n_coeffs)
        n_coeffs = coeffs.shape[-1]
        gram = chebyshev_gram_matrix(n_coeffs, factor.domain)
        m = coeffs @ gram @ coeffs.T  # (rank, rank)
        a = _mode_product(a, m, mode)
    return jnp.sum(a * t.core)


def marginal_tucker(t: ChebTucker, keep_modes: tuple[int, ...]) -> ChebTucker:
    """Exact marginal of the ACTUAL (already-squared) density ``t`` -- i.e.
    ``t`` should be ``square_tucker(g)``, not ``g`` itself, since marginalizing
    (a linear operation) only commutes with squaring in that order.

    Returns a new, smaller ChebTucker over just ``keep_modes`` (sorted
    ascending) representing ``integral t(x) dx_rest``. Unlike
    ``chebtuckerge.marginal``, this works directly on coefficient-tensor
    (``core_representation="coeff_tensor"``) factors -- the built-in version
    requires the ``scalar_chebfuns`` representation.
    """
    keep_modes = tuple(sorted(keep_modes))
    d = len(t.factors)
    core = t.core
    for mode in sorted(set(range(d)) - set(keep_modes), reverse=True):
        factor = t.factors[mode]
        coeffs = factor.core.coeffs[0]  # (rank, n_coeffs)
        moment = chebyshev_moment_vector(coeffs.shape[-1], factor.domain)
        integral_vector = coeffs @ moment  # (rank,)
        core = jnp.tensordot(core, integral_vector, axes=([mode], [0]))
    new_factors = tuple(t.factors[m] for m in keep_modes)
    new_domain = jnp.array([t.factors[m].domain for m in keep_modes], dtype=jnp.float64)
    return ChebTucker(core=core, factors=new_factors, domain=new_domain)


def log_density_tucker(t: ChebTucker, x: jnp.ndarray, eps: float = 1e-300) -> jnp.ndarray:
    g = t(x)
    z = squared_integral_tucker(t)
    return jnp.log(g**2 + eps) - jnp.log(z)


def density_tucker(t: ChebTucker, x: jnp.ndarray) -> jnp.ndarray:
    return jnp.exp(log_density_tucker(t, x))


def nll_tucker(params: TuckerParams, template: ChebTucker, x: jnp.ndarray) -> jnp.ndarray:
    t = rebuild_tucker(template, params)
    return -jnp.mean(log_density_tucker(t, x))


def make_train_step(template: ChebTucker, optimizer: optax.GradientTransformation):
    def loss_fn(params: TuckerParams, x: jnp.ndarray) -> jnp.ndarray:
        return nll_tucker(params, template, x)

    loss_and_grad = jax.jit(jax.value_and_grad(loss_fn))

    @jax.jit
    def step(params: TuckerParams, opt_state, x: jnp.ndarray):
        loss, grads = loss_and_grad(params, x)
        updates, opt_state = optimizer.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)
        return params, opt_state, loss

    return step


def fit_tucker(
    init_t: ChebTucker,
    x: jnp.ndarray,
    *,
    n_steps: int = 2000,
    lr: float = 3e-3,
    batch_size: int | None = None,
    seed: int = 0,
    log_every: int = 200,
    max_grad_norm: float = 1.0,
) -> tuple[ChebTucker, list[float]]:
    optimizer = optax.chain(optax.clip_by_global_norm(max_grad_norm), optax.adam(lr))
    params = tucker_params(init_t)
    opt_state = optimizer.init(params)
    step = make_train_step(init_t, optimizer)

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
            print(f"[fit_tucker] step {i:5d}  nll {float(loss): .6f}")

    fitted = rebuild_tucker(init_t, params)
    return fitted, history
