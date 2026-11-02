"""Paired likelihood uncertainty and complexity selection; no clean GL input."""

import jax
import jax.numpy as jnp
import numpy as np
from gl_covariance_common import load_tt

from ftde.tt_convolution import TTConvolution
from ftde.tt_density import rebuild_tt, tt_params


def select_with_paired_uncertainty(directory, records, data):
    records = [r for r in records if not r['naive']]
    kernel = load_tt(directory/'noise.npz')
    valid = data['valid']
    groups, models, complexities = {}, {}, {}
    for record in records:
        name = record['name']
        g = load_tt(directory/name/'best.npz')
        models[name] = g
        key = tuple(c.coeffs.shape for c in g.cores)
        groups.setdefault(key, []).append(name)
        complexities[name] = sum(c.coeffs.size for c in g.cores)-sum(c.coeffs.shape[1]**2 for c in g.cores[:-1])-1
    scores, operators = {}, {}
    recorded_nll = {r['name']: r['best_nll'] for r in records}
    for names in groups.values():
        template = models[names[0]]
        n = template.cores[0].coeffs.shape[-1]
        if n not in operators:
            operators[n] = TTConvolution.from_tt(template, kernel).prepare(data['observation_domain'], tolerance=1e-14)
        op = operators[n]
        parameters = tuple(jnp.stack([tt_params(models[name])[k] for name in names]) for k in range(len(template.cores)))
        evaluate = jax.jit(lambda p, x, op=op, template=template: jax.vmap(
            lambda pp: op.log_density(rebuild_tt(template, pp), x))(p))
        chunks = []
        for start in range(0, len(valid), 128):
            batch = valid[start:start+128]
            padded = np.pad(batch, ((0, 128-len(batch)), (0, 0)), mode='edge')
            chunks.append(np.asarray(evaluate(parameters, jnp.asarray(padded)))[:, :len(batch)])
        logs = np.concatenate(chunks, axis=1)
        if not np.isfinite(logs).all():
            raise FloatingPointError('Nonfinite validation likelihood in model selection')
        for name, row in zip(names, logs, strict=True):
            np.testing.assert_allclose(-row.mean(), recorded_nll[name], rtol=0, atol=1e-9,
                                       err_msg=f'Batched likelihood disagrees with saved score for {name}')
            scores[name] = row
        print('likelihood uncertainty scores', names, flush=True)
        jax.clear_caches()
    best = min(scores, key=lambda name: -scores[name].mean())
    rows = {}
    # Existing GL sampler emits consecutive blocks of 200 parallel chains.
    # Treat each entire chain as one cluster, allowing temporal dependence.
    if len(valid) % 200:
        raise ValueError('This benchmark expects complete blocks of 200 validation chains')
    for name, logs in scores.items():
        excess = scores[best]-logs
        chain_means = excess.reshape(-1, 200).mean(axis=0)
        se = float(chain_means.std(ddof=1)/np.sqrt(200))
        mean = float(excess.mean())
        rows[name] = {'validation_nll': float(-logs.mean()), 'excess_nll': mean,
                      'paired_cluster_standard_error': se,
                      'tt_dimension_proxy': int(complexities[name]), 'eligible': bool(mean <= se+1e-12)}
    eligible = [name for name in scores if rows[name]['eligible']]
    chosen = min(eligible, key=lambda name: (complexities[name], rows[name]['validation_nll']))
    return chosen, {'rule': 'paired_one_standard_error_then_lowest_TT_dimension',
                    'minimum_nll_model': best, 'selected_model': chosen,
                    'validation_clusters': 200, 'candidates': rows}
