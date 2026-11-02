"""Rank-10 6D noisy follow-up using the previous observation-selected recipe."""
from argparse import Namespace
from pathlib import Path

import run_tucker_gallery as gallery

from ftde.direct_training import write_record


def main():
    gallery.ROOT = Path('artifacts/tucker_noisy_rank10/r10')
    folder = gallery.ROOT/'swirl6_mixture'
    if not (folder/'protocol.json').exists():
        # Reuse the literal prescribed-kernel checkpoint as well as the data;
        # rebuilding a spectral approximation on another GPU can perturb its
        # coefficients. Rank is the intended representation change here.
        protocol = gallery.read('artifacts/tucker_capacity/r6_noisy/swirl6_mixture/protocol.json')
        protocol.update(rank=10, steps=8000)
        folder.mkdir(parents=True, exist_ok=True)
        write_record(folder/'protocol.json', protocol)
    folder, protocol = gallery.setup('swirl', 6, 'mixture', 10, 8000)
    if not (folder/'curriculum/config.json').exists():
        protocol.update(recipes=['curriculum'],
                        recipe_choice='Prior held-out winner at rank 6; fixed before rank-10 training',
                        comparison='Same observations, domain, noise kernel and loss; rank 6 to 10, updates 4000 to 8000')
        write_record(folder/'protocol.json', protocol)
    if protocol['recipes'] != ['curriculum']:
        raise RuntimeError('Existing rank-10 protocol differs')
    args = Namespace(family='swirl', dim=6, noise='mixture', rank=10, steps=8000,
                     recipe='curriculum', microbatches=1, objective='logpen', strength=None)
    gallery.fit(args)
    gallery.freeze(folder)


if __name__ == '__main__':
    main()
