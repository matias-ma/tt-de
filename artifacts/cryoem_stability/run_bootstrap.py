import os, subprocess
from pathlib import Path
root = Path('artifacts/cryoem_stability')
env = dict(os.environ, CHEBTTGE_JAX_CACHE_DIR=f'/tmp/ftde-jax-{os.getuid()}', XLA_PYTHON_CLIENT_PREALLOCATE='false', OMP_NUM_THREADS='4', OPENBLAS_NUM_THREADS='4')
for seed in [0, 11, 23, 37]:
    name = 'dim10_rank48_h2_3e4_control' if seed == 0 else f'dim10_bootstrap_{seed}'
    cmd = ['srun', '--jobid=14407868', '--ntasks=1', '--cpus-per-task=4', '/home/ma4473/.conda/envs/chebdGE/bin/python', 'scripts/fit_cryoem_bootstrap.py', '--base', 'artifacts/cryoem_continuity/gallery_dim10', '--output', str(root/name), '--seed', str(seed), '--rank', '48', '--steps', '3200', '--strength', '.0003']
    if seed:
        cmd.append('--resample')
    with (root/(name+'.log')).open('w') as stream:
        subprocess.run(cmd, env=env, stdout=stream, stderr=subprocess.STDOUT, check=True)
    print('DONE', name, flush=True)
