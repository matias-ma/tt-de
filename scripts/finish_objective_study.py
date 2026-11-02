"""Freeze and publish each completed case; fitting never sees these diagnostics."""
import json
import subprocess
import sys
import time

from objective_study import ROOT


def main():
    counts = {'gl6_free': 12, 'swirl4_free': 9, 'gl6_paired': 8,
              'gl6_mixture': 8, 'swirl4_mixture': 12, 'swirl10_mixture': 12}
    for case, count in counts.items():
        start = time.monotonic()
        while len(list((ROOT/case).glob('*/result.json'))) < count:
            if time.monotonic()-start > 3*3600:
                raise TimeoutError(f'Fitting has not completed: {case}')
            time.sleep(30)
        for script in ['select_objective_study.py', 'evaluate_objective_study.py', 'plot_objective_study.py']:
            print('START', script, case, flush=True)
            with (ROOT/f'{case}_{script}.log').open('w') as log:
                subprocess.run([sys.executable, '-u', 'scripts/'+script, case], stdout=log,
                               stderr=subprocess.STDOUT, check=True)
        print('PUBLISHED', case, flush=True)
    # All cases now have frozen choices, evaluated models and density figures.
    from plot_objective_study import summary
    summary(list(counts))
    (ROOT/'publication.json').write_text(json.dumps({'cases': list(counts), 'fits': sum(counts.values()),
                                                    'all_figures_published': True}, indent=2)+'\n')


if __name__ == '__main__':
    main()
