"""NLL-only, observation-only controls for the noisy Chebyshev TT estimator.

Synthetic data generation is a separate command. Fitting reads only the
observations for its case. No clean fit initializes any noisy fit.
"""
import argparse
import hashlib
import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from objective_study import CASES

from ftde.direct_training import (
    DirectTrainingProblem,
    NoisyDirectTrainingProblem,
    load_tt,
    write_record,
)
from ftde.gaussian_convolution import GaussianConvolution, _rescale, _transfer
from ftde.simple_initialization import init_tt_constant
from ftde.tt_convolution import TTConvolution

ROOT = Path('artifacts/noise_diagnosis_v4')
SIGMAS = (0., .15, .3, .6)


class GaussianMomentProblem(NoisyDirectTrainingProblem):
    """Cache fixed Gaussian integration moments, never a fitted density.

T_i T_j = (T_{i+j}+T_|i-j|)/2 makes 2*n-1 moments sufficient per
observation and coordinate. This is algebraically the same Gaussian
forward quadrature, with all coefficient gradients retained.
"""
    def __init__(self, initial, std, *, quadrature_order=None, **kwargs):
        super().__init__(initial, None, **kwargs)
        self.quadrature = GaussianConvolution.from_tt(initial, std, quadrature_order=quadrature_order)

    def features(self, observations):
        observations = np.asarray(observations)
        result = []
        for k, (n, box) in enumerate(zip(self.sizes, self.domain, strict=True)):
            q = self.quadrature
            t = (2*np.asarray(q.nodes[k])-sum(box))/np.ptp(box)
            v = jnp.asarray(np.polynomial.chebyshev.chebvander(t, 2*n-2))
            @jax.jit
            def integrate(y, k=k, q=q, v=v):
                lw = q.log_weights[k][None]-jnp.log(q.std[k]*np.sqrt(2*np.pi))-.5*((y[:, None]-q.nodes[k])/q.std[k])**2
                mass = jax.scipy.special.logsumexp(lw, axis=1)
                h = jnp.exp(lw-mass[:, None])@v
                return jnp.concatenate([h, mass[:, None]], axis=1)
            result.append(jnp.concatenate([integrate(jnp.asarray(observations[s:s+1024, k]))
                                           for s in range(0, len(observations), 1024)]))
        return tuple(result)

    def _logs(self, params, features):
        state = jnp.ones((len(features[0]), 1, 1))
        logs = jnp.zeros(len(features[0]))
        for c, f, n, inv in zip(params, features, self.sizes, self.inverse, strict=True):
            indices = jnp.arange(n)
            gram = .5*(f[:, indices[:, None]+indices[None, :]]+f[:, jnp.abs(indices[:, None]-indices[None, :])])
            gram = jnp.asarray(inv)[None]@gram@jnp.asarray(inv.T)[None]
            state, scale = _rescale(_transfer(state, c, gram))
            logs += scale+f[:, -1]
        return logs+jnp.log(state[:, 0, 0])-jnp.log(self.energies(params)[0])


def generate():
    from ftde.swirl import sample_swirl
    ROOT.mkdir(parents=True, exist_ok=True)
    if (ROOT/'frozen.json').exists():
        raise RuntimeError('Study already frozen')
    # Freeze geometry from an existing noisy-only box, rounded outwards.
    noisy_box = np.load(Path(CASES['swirl4_mixture'])/'init.npz')['domain']
    radius = float(np.ceil(np.max(np.abs(noisy_box))))
    domain = np.tile([-radius, radius], (4, 1))
    for sigma in SIGMAS:
        folder = ROOT/f'gaussian_{sigma:g}'
        folder.mkdir(exist_ok=True)
        data = {}
        for name, seed, count in [('train', 0, 100000), ('valid', 2, 4096)]:
            latent = sample_swirl(seed, count, 4)
            data[name] = latent+sigma*np.random.default_rng(8000+seed).normal(size=latent.shape)
        np.savez_compressed(folder/'observations.npz', **data)
        write_record(folder/'data_protocol.json', {'sigma': sigma, 'domain': domain.tolist(),
            'domain_source': 'ceil(max(abs(previous noisy-only domain)))',
            'matched_latent_draws_across_levels': True, 'clean_values_available_to_noisy_trainer': False})


def run(case, name, *, seed=0, resume=None, steps=4000, lr=.001, lbfgs=False, modes_override=None):
    folder = ROOT/case
    folder.mkdir(parents=True, exist_ok=True)
    out = folder/name
    if (out/'result.json').exists():
        return
    if (ROOT/'frozen.json').exists():
        raise RuntimeError('Study already frozen')
    if sum(p.stat().st_size for p in ROOT.rglob('*') if p.is_file()) > 256*2**20:
        raise RuntimeError('Study artifact storage budget exceeded')
    if case.startswith('gaussian_'):
        protocol = json.loads((folder/'data_protocol.json').read_text())
        domain, sigma = np.asarray(protocol['domain']), protocol['sigma']
        data = np.load(folder/'observations.npz')
        rank, modes = 32, 49
    else:
        base = Path(CASES['gl6_free' if case == 'gl6_wide_free' else case])
        data = np.load(base/'data.npz')
        domain_base = Path(CASES['gl6_mixture']) if case == 'gl6_wide_free' else base
        domain = np.load(domain_base/'init.npz')['domain']
        rank, modes = (32, 49) if case.startswith('swirl') else (10, 25)
    if modes_override is not None:
        modes = modes_override
    train, valid = data['train'], data['valid']
    initial = load_tt(resume) if resume else init_tt_constant(domain, rank=rank, modes=modes, seed=seed)
    options = {'regularizer': 'hessian', 'strength': .001}
    if case.startswith('gaussian_'):
        problem = GaussianMomentProblem(initial, np.full(4, sigma), **options) if sigma else DirectTrainingProblem(initial, **options)
    elif case.endswith('free'):
        problem = DirectTrainingProblem(initial, **options)
    else:
        observation_domain = np.stack([np.minimum(train.min(0), valid.min(0))-.02,
                                       np.maximum(train.max(0), valid.max(0))+.02], axis=1)
        forward = TTConvolution.from_tt(initial, load_tt(base/'kernel.npz')).prepare(observation_domain, tolerance=1e-14)
        problem = NoisyDirectTrainingProblem(initial, forward, **options)
    out.mkdir(exist_ok=True)
    config = {'case': case, 'seed': seed, 'rank': rank, 'modes': modes, 'steps': steps, 'learning_rate': lr,
        'batch_size': 128 if lbfgs else 256, 'batch_seed': 0, 'objective': 'nll', **options,
        'resume': str(resume) if resume else None, 'initialization': 'earlier noisy NLL fit' if resume else 'constant plus .01 random coefficients',
        'uses_signal_pilot': False, 'uses_truth': False, 'domain': domain.tolist(),
        'optimizer': 'full-data L-BFGS' if lbfgs else 'Adam',
        'data_sha256': hashlib.sha256(train.tobytes()+valid.tobytes()).hexdigest()}
    write_record(out/'config.json', config)
    if lbfgs:
        from ftde.direct_lbfgs import fit_lbfgs
        fit_lbfgs(problem, train, valid, out, iterations=steps, batch_size=128, checkpoint_every=5, validation_batch_size=128)
    else:
        problem.fit_adam(train, valid, out, steps=steps, batch_size=256, learning_rate=lr,
                         checkpoint_every=400, validation_batch_size=128)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('case')
    p.add_argument('--name', default='seed0')
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--resume', type=Path)
    p.add_argument('--steps', type=int, default=4000)
    p.add_argument('--lr', type=float, default=.001)
    p.add_argument('--lbfgs', action='store_true')
    p.add_argument('--modes', type=int)
    args = p.parse_args()
    if args.case == 'generate':
        generate()
    else:
        run(args.case, args.name, seed=args.seed, resume=args.resume, steps=args.steps, lr=args.lr, lbfgs=args.lbfgs, modes_override=args.modes)
