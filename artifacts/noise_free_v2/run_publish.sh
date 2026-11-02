#!/bin/bash
set -e
export JAX_ENABLE_X64=1 JAX_COMPILATION_CACHE_DIR=/tmp/ftde_v2_jax CUDA_CACHE_PATH=/tmp/ftde_v2_cuda XDG_CACHE_HOME=/tmp/ftde_v2_cache MPLCONFIGDIR=/tmp/ftde_v2_mpl XLA_PYTHON_CLIENT_PREALLOCATE=false OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
PY=/home/ma4473/.conda/envs/chebdGE/bin/python
$PY scripts/plot_noise_free_v2.py > artifacts/noise_free_v2/plot.log 2>&1
$PY scripts/plot_noise_free_ablation.py >> artifacts/noise_free_v2/plot.log 2>&1
$PY scripts/plot_noise_free_objectives.py >> artifacts/noise_free_v2/plot.log 2>&1
$PY scripts/plot_noise_free_initialization.py >> artifacts/noise_free_v2/plot.log 2>&1
$PY -u scripts/publish_noise_free_cryoem.py > artifacts/noise_free_v2/cryoem_publish.log 2>&1
$PY scripts/audit_noise_free_v2.py --archive-old > artifacts/noise_free_v2/audit.log 2>&1
