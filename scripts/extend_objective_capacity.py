"""Transfer observation-selected noise-free swirl capacity to noisy controls."""
import json

from objective_study import ROOT
from run_objective_study import run

from ftde.direct_training import write_record


def main():
    capacity = json.loads((ROOT/'swirl4_free/capacity_selection.json').read_text())
    rank, modes = capacity['rank'], capacity['modes']
    protocol = {'source': 'noise-free swirl held-out NLL capacity selection',
                'rank': rank, 'modes': modes, 'uses_truth': False,
                'cases': ['swirl4_mixture', 'swirl10_mixture'],
                'recipes': [['nll', 100.], ['mmd', 100.], ['hybrid', 10.], ['hybrid', 100.]],
                'strength': .001, 'steps': 4000}
    if (ROOT/'capacity_extension.json').exists():
        assert json.loads((ROOT/'capacity_extension.json').read_text()) == protocol
    else:
        write_record(ROOT/'capacity_extension.json', protocol)
    for case in protocol['cases']:
        for objective, weight in protocol['recipes']:
            run(case, rank, modes, objective=objective, kernel_weight=weight)


if __name__ == '__main__':
    main()
