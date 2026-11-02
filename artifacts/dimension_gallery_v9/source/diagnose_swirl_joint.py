"""Post-selection joint-geometry diagnostic for the final 20D Swirl gallery.

Truth is used only to evaluate frozen estimators. Samples come directly from
normalized squared TTs, not from a fitted proposal or a denoising procedure.
"""
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from plot_dimension_gallery import verify
from run_dimension_gallery import ROOT, digest, read
from scipy.spatial import cKDTree

from ftde.direct_training import load_tt, write_record
from ftde.swirl import curve, sample_swirl
from ftde.tt_sampling import make_squared_tt_moments, make_squared_tt_sampler


def main():
    dim, count, seed = 20, 8192, 92637
    overrides = read(ROOT/'publication_overrides.json')
    folders = {noise: Path(overrides.get(f'swirl20_{noise}', ROOT/f'swirl20_{noise}'))
               for noise in ('free', 'mixture')}
    frozen = {noise: verify(folder) for noise, folder in folders.items()}
    # Nearest-curve evaluation is a truth-based diagnostic after freezing.
    tree = cKDTree(curve(np.linspace(0, 1, 20001), dim))
    finer = cKDTree(curve(np.linspace(0, 1, 40001), dim))
    truth = sample_swirl(seed, count, dim)
    truth_distances = tree.query(truth, workers=4)[0]
    distances_by_density = {'True latent density': truth_distances}
    threshold = float(np.quantile(truth_distances, .95))

    def describe(points):
        distances = tree.query(points, workers=4)[0]
        refined = finer.query(points[:128], workers=4)[0]
        change = float(np.max(np.abs(distances[:128]-refined)))
        if change > 1e-4:
            raise FloatingPointError('Curve discretization requires refinement')
        return {'rms_normal_distance_per_sqrt_d_minus_1': float(np.sqrt(np.mean(distances**2)/(dim-1))),
                'median_distance': float(np.median(distances)),
                'distance_p95': float(np.quantile(distances, .95)),
                'fraction_outside_true_95_percent_tube': float(np.mean(distances > threshold)),
                'curve_grid_refinement_max_distance_change': change}

    result = {'dimension': dim, 'sample_count': count, 'seed': seed,
              'true_95_percent_distance_threshold': threshold, 'truth': describe(truth),
              'models': {}, 'used_for_training_or_selection': False,
              'interpretation': 'RMS nearest-curve distance divided by sqrt(d-1); not a fitted Gaussian width parameter'}
    for noise, folder in folders.items():
        name = frozen[noise]['selected']
        model = load_tt(folder/name/'fit.npz')
        params = tuple(core.coeffs for core in model.cores)
        sampler = make_squared_tt_sampler(model)
        uniforms = np.random.default_rng(seed+1).uniform(size=(count, dim))
        points = np.concatenate([np.asarray(sampler(params, jnp.asarray(uniforms[start:start+128])))
                                 for start in range(0, count, 128)])
        mean, variance = map(np.asarray, make_squared_tt_moments(model)(params))
        z = float(np.max(np.abs(points.mean(0)-mean)/np.sqrt(variance/count)))
        if not np.isfinite(points).all() or z > 6:
            raise FloatingPointError('TT sample moments disagree with exact marginal moments')
        distances_by_density['Noise-free estimate' if noise == 'free' else 'Noisy estimate'] = tree.query(points, workers=4)[0]
        result['models'][noise] = describe(points) | {
            'selected': name, 'source': str(folder), 'selection_sha256': digest(folder/'selection.json'),
            'model_sha256': digest(folder/name/'fit.npz'), 'sampler_max_mean_standard_errors': z}
        print('JOINT GEOMETRY', noise, result['models'][noise], flush=True)
        jax.clear_caches()
    write_record(ROOT/'joint_geometry.json', result)
    print('TRUTH GEOMETRY', result['truth'], flush=True)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import plot_dimension_gallery as plotting
    fig, ax = plt.subplots(figsize=(8, 4.8))
    for label, distances in distances_by_density.items():
        ax.plot(np.sort(distances), np.arange(1, count+1)/count, label=label, lw=2)
    ax.axvline(threshold, color='gray', ls=':', label='95% of truth lies to the left')
    ax.axhline(.95, color='gray', ls=':', alpha=.5)
    ax.set(xlabel='Distance to the known 20D Swirl curve', ylabel='Fraction of samples within distance',
           xlim=(0, max(np.quantile(x, .995) for x in distances_by_density.values())), ylim=(0, 1.01),
           title='20D Swirl · recovery of the joint tube')
    ax.grid(alpha=.2)
    ax.legend(loc='lower right', fontsize=9)
    fig.text(.5, .02, '8,192 samples per density · truth used only to evaluate frozen models', ha='center', fontsize=9)
    fig.tight_layout(rect=(0, .05, 1, 1))
    destination = ROOT/'figs/convergence'
    destination.mkdir(exist_ok=True)
    fig.savefig(destination/'joint_tube_recovery.png', dpi=165)
    plt.close(fig)
    plotting.publish()


if __name__ == '__main__':
    main()
