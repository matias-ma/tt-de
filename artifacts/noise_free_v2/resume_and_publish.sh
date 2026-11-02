#!/bin/bash
set -e
export JAX_ENABLE_X64=1 JAX_COMPILATION_CACHE_DIR=/tmp/ftde_v2_jax CUDA_CACHE_PATH=/tmp/ftde_v2_cuda XDG_CACHE_HOME=/tmp/ftde_v2_cache MPLCONFIGDIR=/tmp/ftde_v2_mpl XLA_PYTHON_CLIENT_PREALLOCATE=false OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
/home/ma4473/.conda/envs/chebdGE/bin/python -u scripts/resume_noise_free_cryoem.py dim20 > artifacts/noise_free_v2/cryoem_resume.log 2>&1
bash artifacts/noise_free_v2/run_publish.sh
