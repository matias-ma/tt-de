"""Post-fit rank, mode-tail, and amplitude-sign diagnostics; never used to fit."""
import argparse
import json
from pathlib import Path

import jax.numpy as jnp
import numpy as np
from refresh_density_fits import read_model

from ftde.cheb_gram import chebyshev_gram_matrix
from ftde.swirl import curve


def diagnostics(model):
    arrays = [np.asarray(c.coeffs)@np.linalg.cholesky(np.asarray(chebyshev_gram_matrix(c.coeffs.shape[-1], c.domain)))
              for c in model.cores]
    # Left-canonical form makes each right environment and Schmidt spectrum
    # meaningful in the physical L2 amplitude norm.
    for k in range(len(arrays)-1):
        a, b, n = arrays[k].shape
        q, r = np.linalg.qr(arrays[k].transpose(0, 2, 1).reshape(a*n, b))
        arrays[k] = q.reshape(a, n, -1).transpose(0, 2, 1)
        arrays[k+1] = np.einsum('ab,bcn->acn', r, arrays[k+1])
    norm = np.linalg.norm(arrays[-1])
    arrays[-1] /= norm
    tails, spectra = {}, {}
    for k in range(len(arrays)-1, -1, -1):
        a, b, n = arrays[k].shape
        tails[str(k)] = float(np.sum(arrays[k][:, :, -8:]**2))
        if k:
            u, s, vh = np.linalg.svd(arrays[k].reshape(a, b*n), full_matrices=False)
            spectra[str(k)] = {'energy': (s*s).tolist(), 'beyond16': float(np.sum(s[16:]**2)),
                                    'beyond32': float(np.sum(s[32:]**2))}
            arrays[k] = vh.reshape(len(s), b, n)
            arrays[k-1] = np.einsum('abn,bc->acn', arrays[k-1], u*s)
    t = np.linspace(0, 1, 4001)
    amplitude = np.asarray(model(jnp.asarray(curve(t, len(model.cores)))))
    crossings = np.flatnonzero(amplitude[:-1]*amplitude[1:] < 0)
    return {'top_eight_mode_energy': tails, 'schmidt_spectra': spectra,
                'amplitude_sign_changes_on_true_curve': len(crossings),
                'crossing_t': t[crossings].tolist()}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('models', type=Path, nargs='+')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    out = {}
    for path in args.models:
        record = diagnostics(read_model(path))
        print(path, 'sign changes', record['amplitude_sign_changes_on_true_curve'],
              'max mode tail', max(record['top_eight_mode_energy'].values()), flush=True)
        out[str(path)] = record
    args.output.write_text(json.dumps(out, indent=2)+'\n')
