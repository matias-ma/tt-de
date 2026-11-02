"""Automatic complexity selection for the noisy (deconvolution) estimator.

Deconvolution density estimation is a classically ill-posed inverse problem:
convolving with the noise kernel U washes out exactly the "wiggliness" in
p_tilde that hurts recovery of the true clean density, so held-out NOISY
log-likelihood is nearly flat across a wide range of model complexities that
give very different clean-density recovery (we verified this empirically --
see todo.txt). This means plain cross-validation on noisy NLL alone is not a
reliable way to pick, e.g., the Chebyshev degree of the signal model g: it
readily flags badly-underfit (too-low-degree) models, but cannot distinguish
"just right" from "too flexible / wiggly" among the rest.

The classical remedy for exactly this situation is the "one-standard-error
rule" (Hastie/Tibshirani/Friedman, standard in cross-validated model
selection): among models whose held-out score is not statistically
distinguishable from the best one (within one standard error of its mean),
prefer the *simplest*. Applied here: fit several candidate degrees, score
each by mean held-out noisy NLL (with its standard error, from the per-sample
noisy log-likelihoods), and pick the lowest-degree candidate within one
standard error of the best. This does not require any clean (noise-free)
holdout data -- only the noisy observations, which is all a real deployment
would have.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import jax
import jax.numpy as jnp
import numpy as np
from chebttge.chebtt import ChebTT
from chebtuckerge.chebtucker import ChebTucker

from .init import init_tt_from_kde, init_tucker_from_kde
from .noisy_regularization import DEFAULT_REG_TARGET_BUDGET
from .noisy_tt_density import fit_noisy_tt, log_density_noisy_tt
from .noisy_tucker_density import fit_noisy_tucker, log_density_noisy_tucker
from .tt_density import tt_params
from .tucker_density import tucker_params


def select_by_paired_noisy_nll(log_densities, roughness, *, cluster_ids=None, se_factor=1.):
    """Choose the smoothest candidate within a paired SE of minimum noisy NLL.

    ``log_densities`` has shape (candidates, observations), evaluated on the
    SAME held-out observations. ``roughness`` must be a common function-space
    measure for every candidate. Optional cluster IDs allow dependence within
    groups (for example MCMC chains). The uncertainty is on paired loss
    differences, not on the raw NLL level. This is a model-selection heuristic,
    not a confidence bound after searching multiple fitted candidates.

    Returns the selected index and a JSON-compatible diagnostic dictionary.
    No clean observations or density values are accepted.
    """
    logs = np.asarray(log_densities, dtype=float)
    rough = np.asarray(roughness, dtype=float)
    if logs.ndim != 2 or logs.shape[0] == 0 or logs.shape[1] < 2:
        raise ValueError('log_densities must have candidates and at least two observations')
    if rough.shape != (len(logs),) or not np.isfinite(logs).all() or not np.isfinite(rough).all():
        raise ValueError('finite log densities and one finite roughness per candidate are required')
    if not np.isfinite(se_factor) or se_factor < 0:
        raise ValueError('se_factor must be finite and nonnegative')
    n = logs.shape[1]
    ids = np.arange(n) if cluster_ids is None else np.asarray(cluster_ids)
    if ids.shape != (n,):
        raise ValueError('one cluster ID per observation is required')
    _, clusters = np.unique(ids, return_inverse=True)
    count = int(clusters.max())+1
    if count < 2:
        raise ValueError('at least two independent clusters are required')
    means = -logs.mean(axis=1)
    minimum = int(np.argmin(means))
    rows = []
    for row, r, mean_nll in zip(logs, rough, means, strict=True):
        excess = logs[minimum]-row
        mean = excess.mean()
        sums = np.bincount(clusters, weights=excess-mean)
        se = np.sqrt(count/(count-1)*np.sum(sums**2))/n
        rows.append({'validation_nll': float(mean_nll), 'excess_nll': float(mean),
                     'paired_cluster_se': float(se), 'roughness': float(r),
                     'eligible': bool(mean <= se_factor*se+1e-12)})
    selected = min((k for k, row in enumerate(rows) if row['eligible']),
                   key=lambda k: (rough[k], means[k]))
    return selected, {'minimum_nll_index': minimum, 'selected_index': selected,
                      'clusters': count, 'se_factor': se_factor, 'candidates': rows}


@dataclass
class DegreeCandidateResult:
    n_coeffs: int
    val_nll_mean: float
    val_nll_se: float
    n_params: int
    model: ChebTT


@dataclass
class NoisyDegreeSelectionResult:
    selected: DegreeCandidateResult
    candidates: list[DegreeCandidateResult] = field(default_factory=list)
    final_model: ChebTT | None = None


def select_noisy_tt_degree(
    noisy_data: jnp.ndarray,
    u: ChebTT,
    candidate_n_coeffs: tuple[int, ...],
    *,
    signal_domain: jnp.ndarray | None = None,
    max_rank: int = 6,
    val_fraction: float = 0.2,
    n_steps: int = 1200,
    lr: float = 1e-2,
    max_grad_norm: float = 50.0,
    se_factor: float = 1.0,
    init_gd_steps: int = 10,
    init_n_starts: int = 32,
    refit_on_full_data: bool = True,
    seed: int = 0,
) -> NoisyDegreeSelectionResult:
    """Pick a signal degree for the noisy TT estimator via the 1-SE rule.

    ``u`` is the (already built) fixed noise kernel; its domain should match
    ``signal_domain`` (or ``init_tt_from_kde``'s default bounding box if
    ``signal_domain`` is None) via ``noise_kernel.build_gaussian_noise_kernel_tt``.

    Trains one model per entry of ``candidate_n_coeffs`` on a training split,
    scores each by mean +/- standard-error held-out noisy NLL on a validation
    split, and selects the *smallest* degree whose validation NLL is within
    ``se_factor`` standard errors of the best one. If ``refit_on_full_data``,
    the selected degree is refit once more on the full ``noisy_data`` for the
    returned ``final_model`` (standard practice: use held-out data only to
    pick the hyperparameter, then use all the data for the final fit).
    """
    n = noisy_data.shape[0]
    key = jax.random.PRNGKey(seed)
    perm = jax.random.permutation(key, n)
    n_val = max(1, int(n * val_fraction))
    val_idx, train_idx = perm[:n_val], perm[n_val:]
    train_data, val_data = noisy_data[train_idx], noisy_data[val_idx]

    candidates: list[DegreeCandidateResult] = []
    for degree in candidate_n_coeffs:
        g0 = init_tt_from_kde(
            train_data,
            max_rank=max_rank,
            domain=signal_domain,
            n_coeffs=(degree,),
            gd_steps=init_gd_steps,
            n_starts=init_n_starts,
            random_seed=seed,
        )
        g_fit, _ = fit_noisy_tt(
            g0, u, train_data, n_steps=n_steps, lr=lr, log_every=0, max_grad_norm=max_grad_norm
        )
        per_sample_logq = log_density_noisy_tt(g_fit, u, val_data)
        val_nll_mean = float(-jnp.mean(per_sample_logq))
        val_nll_se = float(jnp.std(-per_sample_logq) / jnp.sqrt(val_data.shape[0]))
        n_params = sum(int(jnp.size(c)) for c in tt_params(g_fit))
        candidates.append(DegreeCandidateResult(degree, val_nll_mean, val_nll_se, n_params, g_fit))

    best = min(candidates, key=lambda c: c.val_nll_mean)
    threshold = best.val_nll_mean + se_factor * best.val_nll_se
    within = [c for c in candidates if c.val_nll_mean <= threshold]
    selected = min(within, key=lambda c: c.n_params)

    final_model = selected.model
    if refit_on_full_data:
        g0_full = init_tt_from_kde(
            noisy_data,
            max_rank=max_rank,
            domain=signal_domain,
            n_coeffs=(selected.n_coeffs,),
            gd_steps=init_gd_steps,
            n_starts=init_n_starts,
            random_seed=seed,
        )
        final_model, _ = fit_noisy_tt(
            g0_full, u, noisy_data, n_steps=n_steps, lr=lr, log_every=0, max_grad_norm=max_grad_norm
        )

    return NoisyDegreeSelectionResult(selected=selected, candidates=candidates, final_model=final_model)


@dataclass
class TuckerDegreeCandidateResult:
    n_coeffs: int
    val_nll_mean: float
    val_nll_se: float
    n_params: int
    model: ChebTucker


@dataclass
class NoisyTuckerDegreeSelectionResult:
    selected: TuckerDegreeCandidateResult
    candidates: list[TuckerDegreeCandidateResult] = field(default_factory=list)
    final_model: ChebTucker | None = None


def select_noisy_tucker_degree(
    noisy_data: jnp.ndarray,
    u: ChebTucker,
    candidate_n_coeffs: tuple[int, ...],
    *,
    signal_domain: jnp.ndarray | None = None,
    max_rank: int = 6,
    val_fraction: float = 0.2,
    n_steps: int = 1200,
    lr: float = 1e-2,
    max_grad_norm: float = 50.0,
    se_factor: float = 1.0,
    init_gd_steps: int = 10,
    init_n_starts: int = 32,
    refit_on_full_data: bool = True,
    seed: int = 0,
) -> NoisyTuckerDegreeSelectionResult:
    """Tucker mirror of ``select_noisy_tt_degree`` -- identical 1-SE rule,
    rationale and caveats (see that function's docstring); only the model
    construction/fitting calls differ (``init_tucker_from_kde``/
    ``fit_noisy_tucker``/``log_density_noisy_tucker`` in place of their TT
    counterparts, and parameter count is the Tucker core tensor's own size
    plus each factor's coefficient count, since ``tucker_params`` returns
    ``(core, factor_coeffs)`` rather than TT's flat list of core arrays).
    """
    n = noisy_data.shape[0]
    key = jax.random.PRNGKey(seed)
    perm = jax.random.permutation(key, n)
    n_val = max(1, int(n * val_fraction))
    val_idx, train_idx = perm[:n_val], perm[n_val:]
    train_data, val_data = noisy_data[train_idx], noisy_data[val_idx]

    candidates: list[TuckerDegreeCandidateResult] = []
    for degree in candidate_n_coeffs:
        g0 = init_tucker_from_kde(
            train_data,
            max_rank=max_rank,
            domain=signal_domain,
            n_coeffs=(degree,),
            gd_steps=init_gd_steps,
            n_starts=init_n_starts,
            random_seed=seed,
        )
        g_fit, _ = fit_noisy_tucker(
            g0, u, train_data, n_steps=n_steps, lr=lr, log_every=0, max_grad_norm=max_grad_norm
        )
        per_sample_logq = log_density_noisy_tucker(g_fit, u, val_data)
        val_nll_mean = float(-jnp.mean(per_sample_logq))
        val_nll_se = float(jnp.std(-per_sample_logq) / jnp.sqrt(val_data.shape[0]))
        core, factor_coeffs = tucker_params(g_fit)
        n_params = int(jnp.size(core)) + sum(int(jnp.size(c)) for c in factor_coeffs)
        candidates.append(TuckerDegreeCandidateResult(degree, val_nll_mean, val_nll_se, n_params, g_fit))

    best = min(candidates, key=lambda c: c.val_nll_mean)
    threshold = best.val_nll_mean + se_factor * best.val_nll_se
    within = [c for c in candidates if c.val_nll_mean <= threshold]
    selected = min(within, key=lambda c: c.n_params)

    final_model = selected.model
    if refit_on_full_data:
        g0_full = init_tucker_from_kde(
            noisy_data,
            max_rank=max_rank,
            domain=signal_domain,
            n_coeffs=(selected.n_coeffs,),
            gd_steps=init_gd_steps,
            n_starts=init_n_starts,
            random_seed=seed,
        )
        final_model, _ = fit_noisy_tucker(
            g0_full, u, noisy_data, n_steps=n_steps, lr=lr, log_every=0, max_grad_norm=max_grad_norm
        )

    return NoisyTuckerDegreeSelectionResult(selected=selected, candidates=candidates, final_model=final_model)


@dataclass
class RegLambdaCandidateResult:
    reg_lambda: float
    val_nll_mean: float
    val_nll_se: float
    model: ChebTT


@dataclass
class NoisyRegLambdaSelectionResult:
    selected: RegLambdaCandidateResult
    candidates: list[RegLambdaCandidateResult] = field(default_factory=list)
    final_model: ChebTT | None = None


def select_noisy_tt_reg_lambda(
    noisy_data: jnp.ndarray,
    u: ChebTT,
    candidate_reg_lambdas: tuple[float, ...],
    *,
    degree: int,
    noise_std: jnp.ndarray,
    signal_domain: jnp.ndarray | None = None,
    max_rank: int = 6,
    target_budget: float = DEFAULT_REG_TARGET_BUDGET,
    val_fraction: float = 0.2,
    n_steps: int = 1200,
    lr: float = 1e-4,
    max_grad_norm: float = 50.0,
    se_factor: float = 1.0,
    init_gd_steps: int = 10,
    init_n_starts: int = 32,
    refit_on_full_data: bool = True,
    seed: int = 0,
) -> NoisyRegLambdaSelectionResult:
    """Pick the Tikhonov ``reg_lambda`` (see ``noisy_regularization.py``) for
    a FIXED signal degree/rank via the same 1-SE rule as
    ``select_noisy_tt_degree``, and for the same reason: held-out noisy NLL
    alone is nearly flat across a wide range of regularization strengths
    that give very different clean-density recovery (confirmed empirically
    on real cryo-EM data -- see todo.txt), so it can flag badly-underfit
    (too little regularization, catastrophic blowup) candidates but not
    reliably distinguish "just enough" from "more than needed" among the
    rest.

    Unlike ``select_noisy_tt_degree`` (which breaks 1-SE ties toward fewer
    parameters), ties here are broken toward the LARGEST ``reg_lambda``
    among candidates within ``se_factor`` standard errors of the best: more
    regularization means a simpler/smoother model in this context, not a
    smaller one (the signal's own degree/rank are held fixed across
    candidates, so parameter count does not vary). ``reg_lambda=0.0`` is a
    valid candidate (disables regularization) and is included for
    comparison if passed, but note the default fitting ``lr`` here is tuned
    for regularized runs (1e-4, see ``noisy_regularization.py``'s and
    ``cryoem_common.py``'s docs on why regularized deconvolution needs a
    lower learning rate than the unregularized default of ~1e-3) -- an
    unregularized candidate fit at this same low ``lr`` may just converge
    more slowly, not worse, within a fixed ``n_steps``.

    One shared KDE-cross-approximation init is reused across every
    ``reg_lambda`` candidate (unlike ``select_noisy_tt_degree``, which must
    rebuild it per degree since degree changes the model's own structure --
    here only the training objective's regularization term changes).
    """
    n = noisy_data.shape[0]
    key = jax.random.PRNGKey(seed)
    perm = jax.random.permutation(key, n)
    n_val = max(1, int(n * val_fraction))
    val_idx, train_idx = perm[:n_val], perm[n_val:]
    train_data, val_data = noisy_data[train_idx], noisy_data[val_idx]

    g0 = init_tt_from_kde(
        train_data,
        max_rank=max_rank,
        domain=signal_domain,
        n_coeffs=(degree,),
        gd_steps=init_gd_steps,
        n_starts=init_n_starts,
        random_seed=seed,
    )

    candidates: list[RegLambdaCandidateResult] = []
    for reg_lambda in candidate_reg_lambdas:
        # reg_strength=reg_lambda directly + per_axis_target_budget, NOT
        # calibrated_reg_strength: the latter computes one global scalar
        # calibrated against the single worst-case axis, which silently
        # under-regularizes every other axis whenever domain/noise_std are
        # heterogeneous across modes (see noisy_regularization.py's module
        # comment and todo.txt for the real cryo-EM dim=10 failure this
        # caused). Per-axis normalization reduces to the exact same
        # behavior as calibrated_reg_strength in the homogeneous case.
        g_fit, _ = fit_noisy_tt(
            g0, u, train_data, n_steps=n_steps, lr=lr, log_every=0, max_grad_norm=max_grad_norm,
            noise_std=noise_std, reg_strength=reg_lambda,
            per_axis_target_budget=target_budget if reg_lambda else None,
        )
        per_sample_logq = log_density_noisy_tt(g_fit, u, val_data)
        val_nll_mean = float(-jnp.mean(per_sample_logq))
        val_nll_se = float(jnp.std(-per_sample_logq) / jnp.sqrt(val_data.shape[0]))
        candidates.append(RegLambdaCandidateResult(reg_lambda, val_nll_mean, val_nll_se, g_fit))

    best = min(candidates, key=lambda c: c.val_nll_mean)
    threshold = best.val_nll_mean + se_factor * best.val_nll_se
    within = [c for c in candidates if c.val_nll_mean <= threshold]
    selected = max(within, key=lambda c: c.reg_lambda)

    final_model = selected.model
    if refit_on_full_data:
        g0_full = init_tt_from_kde(
            noisy_data,
            max_rank=max_rank,
            domain=signal_domain,
            n_coeffs=(degree,),
            gd_steps=init_gd_steps,
            n_starts=init_n_starts,
            random_seed=seed,
        )
        final_model, _ = fit_noisy_tt(
            g0_full, u, noisy_data, n_steps=n_steps, lr=lr, log_every=0, max_grad_norm=max_grad_norm,
            noise_std=noise_std, reg_strength=selected.reg_lambda,
            per_axis_target_budget=target_budget if selected.reg_lambda else None,
        )

    return NoisyRegLambdaSelectionResult(selected=selected, candidates=candidates, final_model=final_model)


@dataclass
class TuckerRegLambdaCandidateResult:
    reg_lambda: float
    val_nll_mean: float
    val_nll_se: float
    model: ChebTucker


@dataclass
class NoisyTuckerRegLambdaSelectionResult:
    selected: TuckerRegLambdaCandidateResult
    candidates: list[TuckerRegLambdaCandidateResult] = field(default_factory=list)
    final_model: ChebTucker | None = None


def select_noisy_tucker_reg_lambda(
    noisy_data: jnp.ndarray,
    u: ChebTucker,
    candidate_reg_lambdas: tuple[float, ...],
    *,
    degree: int,
    noise_std: jnp.ndarray,
    signal_domain: jnp.ndarray | None = None,
    max_rank: int = 6,
    target_budget: float = DEFAULT_REG_TARGET_BUDGET,
    val_fraction: float = 0.2,
    n_steps: int = 1200,
    lr: float = 1e-4,
    max_grad_norm: float = 50.0,
    se_factor: float = 1.0,
    init_gd_steps: int = 10,
    init_n_starts: int = 32,
    refit_on_full_data: bool = True,
    seed: int = 0,
) -> NoisyTuckerRegLambdaSelectionResult:
    """Tucker mirror of ``select_noisy_tt_reg_lambda`` -- identical 1-SE
    rule, rationale and caveats (see that function's docstring); only the
    model construction/fitting calls differ (``init_tucker_from_kde``/
    ``fit_noisy_tucker``/``log_density_noisy_tucker`` in place of their TT
    counterparts).
    """
    n = noisy_data.shape[0]
    key = jax.random.PRNGKey(seed)
    perm = jax.random.permutation(key, n)
    n_val = max(1, int(n * val_fraction))
    val_idx, train_idx = perm[:n_val], perm[n_val:]
    train_data, val_data = noisy_data[train_idx], noisy_data[val_idx]

    g0 = init_tucker_from_kde(
        train_data,
        max_rank=max_rank,
        domain=signal_domain,
        n_coeffs=(degree,),
        gd_steps=init_gd_steps,
        n_starts=init_n_starts,
        random_seed=seed,
    )

    candidates: list[TuckerRegLambdaCandidateResult] = []
    for reg_lambda in candidate_reg_lambdas:
        g_fit, _ = fit_noisy_tucker(
            g0, u, train_data, n_steps=n_steps, lr=lr, log_every=0, max_grad_norm=max_grad_norm,
            noise_std=noise_std, reg_strength=reg_lambda,
            per_axis_target_budget=target_budget if reg_lambda else None,
        )
        per_sample_logq = log_density_noisy_tucker(g_fit, u, val_data)
        val_nll_mean = float(-jnp.mean(per_sample_logq))
        val_nll_se = float(jnp.std(-per_sample_logq) / jnp.sqrt(val_data.shape[0]))
        candidates.append(TuckerRegLambdaCandidateResult(reg_lambda, val_nll_mean, val_nll_se, g_fit))

    best = min(candidates, key=lambda c: c.val_nll_mean)
    threshold = best.val_nll_mean + se_factor * best.val_nll_se
    within = [c for c in candidates if c.val_nll_mean <= threshold]
    selected = max(within, key=lambda c: c.reg_lambda)

    final_model = selected.model
    if refit_on_full_data:
        g0_full = init_tucker_from_kde(
            noisy_data,
            max_rank=max_rank,
            domain=signal_domain,
            n_coeffs=(degree,),
            gd_steps=init_gd_steps,
            n_starts=init_n_starts,
            random_seed=seed,
        )
        final_model, _ = fit_noisy_tucker(
            g0_full, u, noisy_data, n_steps=n_steps, lr=lr, log_every=0, max_grad_norm=max_grad_norm,
            noise_std=noise_std, reg_strength=selected.reg_lambda,
            per_axis_target_budget=target_budget if selected.reg_lambda else None,
        )

    return NoisyTuckerRegLambdaSelectionResult(selected=selected, candidates=candidates, final_model=final_model)
