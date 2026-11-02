import os, subprocess, time
from pathlib import Path
root = Path('artifacts/cryoem_stability')
start = time.monotonic()
while not (root/'dim10_bootstrap_37/marginal_89.npz').exists():
    if time.monotonic()-start > 2700:
        raise TimeoutError('Bootstrap runs did not finish within 45 minutes')
    time.sleep(10)
env = dict(os.environ, CHEBTTGE_JAX_CACHE_DIR=f'/tmp/ftde-jax-{os.getuid()}', XLA_PYTHON_CLIENT_PREALLOCATE='false', OMP_NUM_THREADS='4', OPENBLAS_NUM_THREADS='4')
prefix = ['srun', '--jobid=14407868', '--ntasks=1', '--cpus-per-task=4', '/home/ma4473/.conda/envs/chebdGE/bin/python']
boots = [str(root/f'dim10_bootstrap_{seed}') for seed in [11,23,37]]
cmd = prefix+['scripts/compare_cryoem_stability.py', '--base', 'artifacts/cryoem_continuity/gallery_dim10', '--output', str(root/'comparison_dim10'), '--candidates', str(root/'dim10_rank48_h2_3e4_control'), str(root/'dim10_rank64_h2_3e4'), *boots, '--ensemble', *boots]
with (root/'comparison_dim10.log').open('w') as stream:
    subprocess.run(cmd, env=env, stdout=stream, stderr=subprocess.STDOUT, check=True)
print('DONE all-models comparison', flush=True)
cmd = prefix+['scripts/refresh_density_figures.py', str(root/'gallery_dim10'), str(root/'gallery_dim20'), '--figures', str(root/'gallery')]
with (root/'render.log').open('w') as stream:
    subprocess.run(cmd, env=env, stdout=stream, stderr=subprocess.STDOUT, check=True)
print('DONE gallery staging', flush=True)
