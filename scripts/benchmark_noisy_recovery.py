"""Matched paired-GL regularization experiment; selection uses noisy data only.

Reuse a saved setup from benchmark_gl_convolution.py --rsi-setup-only.
The initializer and prescribed kernel are identical for every candidate.
Clean density diagnostics are deliberately delegated to a separate evaluator.
"""

import argparse
import json
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import optax
from chebttge.chebtt import ChebTT
from chebttge.cores import ChebyshevCoeffCore
from gl_gpu_noisy import _noniso_covariance

from ftde.cheb_product import square_tt
from ftde.fisher_visibility import FisherVisibilityPenalty
from ftde.ginzburg_landau import sample_ginzburg_landau
from ftde.noise_marginals import noise_marginals_from_tt
from ftde.noisy_model_selection import select_by_paired_noisy_nll
from ftde.noisy_regularization import tikhonov_penalty_tt
from ftde.reference_score import ReferenceScorePenalty
from ftde.sobolev import AmplitudeSobolevPenalty
from ftde.tt_convolution import TTConvolution
from ftde.tt_density import rebuild_tt, tt_params


def load_model(saved, prefix, domain):
    return ChebTT(cores=tuple(ChebyshevCoeffCore(jnp.asarray(saved[f'{prefix}_{k}']), tuple(box))
                             for k, box in enumerate(domain)), domain=jnp.asarray(domain))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--setup', type=Path, default=Path('artifacts/gl_rsi_trial/paired_setup.npz'))
    parser.add_argument('--output', type=Path, default=Path('artifacts/noisy_recovery'))
    parser.add_argument('--strengths', type=float, nargs='+', default=[0., .0001, .001, .01])
    parser.add_argument('--order', type=int, default=2)
    parser.add_argument('--n-coeffs', type=int, help='pad the same initial function to a larger basis')
    parser.add_argument('--legacy', action='store_true', help='include the original coefficient penalty')
    parser.add_argument('--steps', type=int, default=8000)
    parser.add_argument('--batch-size', type=int, default=256)
    parser.add_argument('--lr', type=float, default=.0005)
    parser.add_argument('--n-train', type=int, default=100000)
    parser.add_argument('--noise-scale', type=float, default=1.5)
    parser.add_argument('--train-seed', type=int, default=0)
    parser.add_argument('--valid-seed', type=int, default=2)
    parser.add_argument('--warm-start', type=Path)
    parser.add_argument('--reference', type=Path, help='polynomial log-density reference fitted from noisy marginals')
    parser.add_argument('--fisher-floor', type=float, help='use a Fisher visibility filter instead of reference-score derivatives')
    parser.add_argument('--checkpoint-rule', choices=('minimum-nll', 'paired-one-se'), default='minimum-nll')
    parser.add_argument('--kernel-marginals', action='store_true', help='build the prior from the prescribed noise TT without Gaussian parameters')
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    saved = np.load(args.setup)
    domain = saved['domain']
    d = len(domain)
    init = load_model(saved, 'signal', domain)
    if args.warm_start:
        warm = np.load(args.warm_start)
        np.testing.assert_array_equal(warm['domain'], domain)
        init = load_model(warm, 'deconvolved', domain)
    if args.n_coeffs is not None:
        if args.n_coeffs < max(c.coeffs.shape[-1] for c in init.cores):
            raise ValueError('--n-coeffs must preserve or increase the initial coefficient count')
        init = rebuild_tt(init, [jnp.pad(c.coeffs, ((0, 0), (0, 0), (0, args.n_coeffs-c.coeffs.shape[-1])))
                                 for c in init.cores])
    kernel = load_model(saved, 'kernel', saved['kernel_domain'])
    covariance = _noniso_covariance(d, args.noise_scale)
    noise_scales = (np.array([m.moments()[1] for m in noise_marginals_from_tt(kernel)])
                    if args.kernel_marginals else np.sqrt(np.diag(covariance)))

    def observations(seed, count):
        # Clean samples are hidden by the observation generator; no clean
        # samples/densities are exposed to fitting or selection below.
        return np.asarray(sample_ginzburg_landau(seed, count, d)) + np.random.default_rng(seed).multivariate_normal(
            np.zeros(d), covariance, size=count)

    train = observations(args.train_seed, args.n_train)
    valid = observations(args.valid_seed, 2048)
    if args.valid_seed == 2 and args.noise_scale == 1.5:
        np.testing.assert_allclose(valid[:256], saved['probe'], atol=1e-14)
    all_obs = np.concatenate([train, valid])
    obs_domain = np.stack([all_obs.min(axis=0)-.05, all_obs.max(axis=0)+.05], axis=1)
    direct = TTConvolution.from_tt(init, kernel)
    op = direct.prepare(obs_domain, tolerance=1e-14)
    smoothness = AmplitudeSobolevPenalty.from_tt(init, order=args.order,
                                               length_scales=noise_scales)
    if args.reference:
        reference = np.load(args.reference)
        np.testing.assert_array_equal(reference['domain'], domain)
        smoothness = ReferenceScorePenalty.from_tt(init, reference['log_coeffs'],
                                                   length_scales=noise_scales)
        if args.fisher_floor is not None:
            noise_arguments = {'noise_tt': kernel} if args.kernel_marginals else {'noise_std': noise_scales}
            smoothness = FisherVisibilityPenalty.from_tt(init, reference['log_coeffs'],
                **noise_arguments, visibility_floor=args.fisher_floor)
            np.savez(args.output/'fisher_spectra.npz', **{f'axis_{k}': v for k, v in enumerate(smoothness.eigenvalues)})
    elif args.fisher_floor is not None:
        raise ValueError('--fisher-floor requires --reference')
    factors = [np.linalg.cholesky(np.asarray(gram)) for gram in op.grams]
    inverses = tuple(jnp.asarray(np.linalg.inv(f)) for f in factors)
    initial = tuple(c@f for c, f in zip(tt_params(init), factors, strict=True))

    def rebuild(p):
        return rebuild_tt(init, tuple(c@f for c, f in zip(p, inverses, strict=True)))

    evaluate = jax.jit(lambda p, x: op.log_density(rebuild(p), x))
    penalty = jax.jit(lambda p: smoothness(rebuild(p)))
    legacy_penalty = jax.jit(lambda p: tikhonov_penalty_tt(
        square_tt(rebuild(p)), jnp.sqrt(jnp.diag(jnp.asarray(covariance))), per_axis_target_budget=.5))
    probe = jnp.asarray(valid[:8])
    discrepancy = float(jnp.max(jnp.abs(evaluate(initial, probe)-direct.log_density(init, probe))))
    if discrepancy > 1e-8:
        raise FloatingPointError(f'Prepared convolution discrepancy {discrepancy}')
    print('setup complete, direct log discrepancy', discrepancy, flush=True)
    train = jnp.asarray(train)
    prefix = 'fisher' if args.fisher_floor is not None else 'reference' if args.reference else f'h{args.order}'
    candidates = ([('legacy', 1.)] if args.legacy else []) + [(f'{prefix}_{s:g}', s) for s in args.strengths]
    results, scores = {}, {}
    for name, strength in candidates:
        destination = args.output/name
        destination.mkdir(exist_ok=True)
        if (destination/'result.json').exists():
            record = json.loads((destination/'result.json').read_text())
            settings = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
            previous = dict(record['settings'])
            previous.setdefault('checkpoint_rule', 'minimum-nll')
            previous.setdefault('kernel_marginals', False)
            if ({k: previous.get(k) for k in settings} != settings or set(previous)-set(settings)):
                raise ValueError('Existing result has different settings; use a new output directory')
            results[name] = record
            scores[name] = np.load(destination/'validation.npy')
            continue
        optimizer = optax.chain(optax.clip_by_global_norm(1.), optax.adam(
            optax.cosine_decay_schedule(args.lr, args.steps, alpha=.1)))
        regularizer = legacy_penalty if name == 'legacy' else penalty

        def step_fn(p, state, x, strength=strength, regularizer=regularizer, optimizer=optimizer):
            def objective(pp):
                return -evaluate(pp, x).mean() + strength*regularizer(pp)
            value, grad = jax.value_and_grad(objective)(p)
            updates, state = optimizer.update(grad, state, p)
            return optax.apply_updates(p, updates), state, value

        step = jax.jit(step_fn)

        params = initial
        state = optimizer.init(params)
        best, best_nll, best_step, best_logs = params, np.inf, 0, None
        history = []
        checkpoints, checkpoint_logs = [], []
        key = jax.random.PRNGKey(0)
        start = time.monotonic()
        for i in range(args.steps+1):
            if i % 400 == 0 or i == args.steps:
                logs = np.concatenate([np.asarray(evaluate(params, jnp.asarray(valid[j:j+256])))
                                       for j in range(0, len(valid), 256)])
                if not np.isfinite(logs).all():
                    raise FloatingPointError(f'{name}: nonfinite validation at {i}')
                nll = -float(logs.mean())
                roughness = float(penalty(params))
                history.append({'step': i, 'nll': nll, 'roughness': roughness})
                if args.checkpoint_rule == 'paired-one-se':
                    checkpoints.append(params)
                    checkpoint_logs.append(logs)
                if nll < best_nll:
                    best, best_nll, best_step, best_logs = params, nll, i, logs
                print(name, i, 'nll', nll, 'roughness', roughness,
                      'seconds', round(time.monotonic()-start, 1), flush=True)
            if i == args.steps:
                break
            key, sub = jax.random.split(key)
            batch = train[jax.random.randint(sub, (args.batch_size,), 0, len(train))]
            params, state, loss = step(params, state, batch)
            if not np.isfinite(float(loss)):
                raise FloatingPointError(f'{name}: nonfinite objective at {i}')
        minimum_nll = best_nll
        checkpoint_selection = None
        if args.checkpoint_rule == 'paired-one-se':
            selected_step, checkpoint_selection = select_by_paired_noisy_nll(
                np.stack(checkpoint_logs), [r['roughness'] for r in history],
                cluster_ids=np.arange(len(valid)) % 200)
            best, best_logs = checkpoints[selected_step], checkpoint_logs[selected_step]
            best_nll, best_step = history[selected_step]['nll'], history[selected_step]['step']
        fitted = rebuild(best)
        fine = TTConvolution.from_tt(init, kernel, quadrature_order=2*np.asarray(direct.orders))
        check = float(jnp.max(jnp.abs(evaluate(best, probe)-fine.log_density(fitted, probe))))
        record = {'name': name, 'strength': strength, 'best_nll': best_nll, 'best_step': best_step,
                      'minimum_nll': minimum_nll, 'checkpoint_selection': checkpoint_selection,
                      'roughness': float(penalty(best)), 'history': history, 'seconds': time.monotonic()-start,
                      'quadrature_log_error': check,
                      'settings': {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}}
        np.savez(destination/'fit.npz', domain=domain,
                 **{f'deconvolved_{k}': np.asarray(c.coeffs) for k, c in enumerate(fitted.cores)},
                 **{f'naive_{k}': np.asarray(c.coeffs) for k, c in enumerate(init.cores)})
        np.save(destination/'validation.npy', best_logs)
        (destination/'result.json').write_text(json.dumps(record, indent=2)+'\n')
        results[name], scores[name] = record, best_logs
    # Compare paired per-observation losses, with the sampler's 200 chains
    # as clusters. The last incomplete block has appropriate cluster weights.
    names = list(results)
    chosen, detail = select_by_paired_noisy_nll(
        np.stack([scores[name] for name in names]), [results[name]['roughness'] for name in names],
        cluster_ids=np.arange(len(valid)) % 200)
    selection = {'minimum_nll': names[detail['minimum_nll_index']], 'selected': names[chosen],
                 'rule': 'paired cluster one-SE, then minimum integrated roughness',
                 'candidates': dict(zip(names, detail['candidates'], strict=True))}
    (args.output/'selection.json').write_text(json.dumps(selection, indent=2)+'\n')
    print('SELECTION (no clean truth)', json.dumps(selection, indent=2), flush=True)


if __name__ == '__main__':
    main()
