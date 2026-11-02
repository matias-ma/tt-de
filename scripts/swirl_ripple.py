"""Quantify along-ridge beading of every swirl candidate (post-selection diagnostic).

For each pair marginal, the estimate and the truth are evaluated along the
projected true path. Their difference is split with a Gaussian filter in t
(width RIPPLE_WIDTH, about the length of path covered by 1.5 tube widths
at typical speed) into a slowly varying part and a high-pass residual. The
high-pass part, relative to the truth profile norm, measures bead-like
oscillation: the truth contributes its own sharp features (path crossings,
fold caustics) to both profiles, so only the difference is filtered.
Writes <case>/ripple.json; never used for fitting or selection.
"""

import argparse
import json
from pathlib import Path

import numpy as np
from refresh_density_fits import read_model
from scipy.ndimage import gaussian_filter1d
from swirl_figures import WINDOW, ridge_profile

from ftde.marginals import squared_pair_grid
from ftde.swirl import swirl_pair_marginal

RIPPLE_WIDTH = .012


def ripple(case):
    settings = json.loads((case/'settings.json').read_text())
    selection = json.loads((case/'selection.json').read_text())
    dim = settings['dim']
    xs = np.linspace(-WINDOW, WINDOW, 361)
    t = np.linspace(0, 1, 4001)
    sigma = RIPPLE_WIDTH/(t[1]-t[0])
    out = {'filter_width_t': RIPPLE_WIDTH, 'selected': selection['selected'], 'candidates': {}}
    pairs = [(0, 1), (2, 3)] + ([(0, dim-1)] if dim >= 8 else [])
    truths = {p: ridge_profile(swirl_pair_marginal(dim, *p, xs, xs), xs, dim, *p, t) for p in pairs}
    for name in selection['names']:
        fit = read_model(case/name/'fit.npz')
        record = {}
        for i, j in pairs:
            estimate = ridge_profile(squared_pair_grid(fit, i, j, xs, xs), xs, dim, i, j, t)
            diff = estimate-truths[i, j]
            smooth = gaussian_filter1d(diff, sigma, mode='nearest')
            norm = np.linalg.norm(truths[i, j])
            record[f'{i}{j}'] = {'profile_relative_error': float(np.linalg.norm(diff)/norm),
                                 'smooth_relative_error': float(np.linalg.norm(smooth)/norm),
                                 'ripple_relative_error': float(np.linalg.norm(diff-smooth)/norm)}
        out['candidates'][name] = record
        print(case.name, name, {k: round(v['ripple_relative_error'], 4) for k, v in record.items()}, flush=True)
    (case/'ripple.json').write_text(json.dumps(out, indent=2)+'\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('cases', nargs='+', type=Path)
    for case in parser.parse_args().cases:
        ripple(case)


if __name__ == '__main__':
    main()
