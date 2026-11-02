"""Post-selection oracle diagnostic of the data/regularization tradeoff.

Evaluate the already-frozen clean-data TT under each noisy likelihood.
It is NEVER used to initialize, train, select or update a noisy model.
"""
import hashlib
import json
from pathlib import Path

import jax
import jax.numpy as jnp
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from noise_diagnosis import ROOT, SIGMAS, GaussianMomentProblem

from ftde.direct_training import load_tt, write_record


def main():
    frozen = json.loads((ROOT/'frozen.json').read_text())
    for path, digest in frozen['files'].items():
        assert hashlib.sha256(Path(path).read_bytes()).hexdigest() == digest
    from ftde.swirl import sample_swirl
    clean = sample_swirl(732981, 4096, 4)
    standard_noise = np.random.default_rng(740981).normal(size=clean.shape)
    clean_model = load_tt(frozen['cases']['gaussian_0']['models']['seed0']['path'])
    records = []
    for sigma in SIGMAS[1:]:
        case = f'gaussian_{sigma:g}'
        model = load_tt(frozen['cases'][case]['models']['seed0']['path'])
        np.testing.assert_array_equal(model.domain, clean_model.domain)
        problem = GaussianMomentProblem(clean_model, np.full(4, sigma), strength=.001, regularizer='hessian')
        noisy_params = tuple(jnp.asarray(c.coeffs)@factor for c, factor in zip(model.cores, problem.factors, strict=True))
        training = problem.features(np.load(ROOT/case/'observations.npz')['train'])
        testing = problem.features(clean+sigma*standard_noise)
        def logs(params, features, problem=problem):
            return np.concatenate([np.asarray(problem.logs(params, tuple(f[s:s+128] for f in features)))
                                   for s in range(0, len(features[0]), 128)])
        train_clean, train_noisy = [logs(p, training) for p in [problem.params, noisy_params]]
        test_clean, test_noisy = [logs(p, testing) for p in [problem.params, noisy_params]]
        np.testing.assert_allclose(test_noisy, np.load(ROOT/case/'test_logs_seed0.npy'), atol=1e-10, rtol=0)
        energies = [np.asarray(problem.energies(p)) for p in [problem.params, noisy_params]]
        penalties = [.001*e[3]/e[0] for e in energies]
        delta = test_noisy-test_clean
        row = {'sigma': sigma, 'clean_fit_noisy_training_nll': -float(train_clean.mean()),
            'noisy_fit_training_nll': -float(train_noisy.mean()),
            'clean_fit_penalty': float(penalties[0]), 'noisy_fit_penalty': float(penalties[1]),
            'clean_minus_noisy_training_nll': float((train_noisy-train_clean).mean()),
            'clean_minus_noisy_penalty': float(penalties[0]-penalties[1]),
            'clean_minus_noisy_total_objective': float((train_noisy-train_clean).mean()+penalties[0]-penalties[1]),
            'clean_minus_noisy_test_nll': float(delta.mean()), 'paired_test_se': float(delta.std(ddof=1)/np.sqrt(len(delta)))}
        records.append(row)
        print(row, flush=True)
        del training, testing, problem
        jax.clear_caches()
    write_record(ROOT/'objective_bias.json', {'records': records, 'post_selection_oracle_diagnostic': True,
        'clean_fit_used_in_noisy_training_or_selection': False,
        'training_count': 100000, 'test_count': len(clean),
        'interpretation': 'Positive total-objective difference means the trained noisy fit is preferred over this same-capacity clean-data comparator by the actual training objective.'})
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5), layout='constrained')
    x = np.arange(len(records))
    for offset, key, label in [(-.25, 'clean_minus_noisy_training_nll', 'Noisy training NLL'),
                               (0., 'clean_minus_noisy_penalty', 'Hessian penalty'),
                               (.25, 'clean_minus_noisy_total_objective', 'Total training objective')]:
        axes[0].bar(x+offset, [r[key] for r in records], width=.25, label=label)
    axes[0].set_xticks(x, [str(r['sigma']) for r in records])
    axes[0].set(title='Training objective: clean-data TT minus noisy-data TT', xlabel='Measurement-noise σ',
                ylabel='Difference in nats\nPositive: objective prefers the noisy-trained TT')
    axes[0].legend(fontsize=8)
    axes[1].errorbar([r['sigma'] for r in records], [r['clean_minus_noisy_test_nll'] for r in records],
                     yerr=[1.96*r['paired_test_se'] for r in records], fmt='o-', capsize=4)
    axes[1].set(title='Independent noisy test NLL difference', xlabel='Measurement-noise σ',
                ylabel='Clean-data TT minus noisy-data TT\n± 1.96 paired standard errors')
    for ax in axes:
        ax.axhline(0, color='gray', ls='--', lw=1)
        ax.grid(alpha=.2)
    fig.suptitle('The fixed smoothness penalty can outweigh the likelihood benefit of latent detail', fontsize=14)
    fig.text(.02, .015, 'Diagnostic only: evaluate the frozen noise-free fit through the same noise operator; never initialize, select or train a noisy fit with it.\n'
             'Both TTs: rank 32, 49 Chebyshev modes, box [−3,3]⁴; Ntrain=100,000, Ntest=4096; Hessian λ=0.001, physical scales=0.3.\n'
             'A positive objective difference diagnoses a tradeoff for this pair of models; it is not a proof of global optimality or unavoidable statistical error.', fontsize=9, va='bottom')
    fig.get_layout_engine().set(rect=(0, .14, 1, .81))
    fig.savefig('figs/convergence/noise_regularization_bias.png', dpi=160)
    plt.close(fig)


if __name__ == '__main__':
    main()
