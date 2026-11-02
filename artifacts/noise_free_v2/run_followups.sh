#!/bin/bash
set -e
export JAX_ENABLE_X64=1 JAX_COMPILATION_CACHE_DIR=/tmp/ftde_v2_jax CUDA_CACHE_PATH=/tmp/ftde_v2_cuda XDG_CACHE_HOME=/tmp/ftde_v2_cache MPLCONFIGDIR=/tmp/ftde_v2_mpl XLA_PYTHON_CLIENT_PREALLOCATE=false OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
PY=/home/ma4473/.conda/envs/chebdGE/bin/python
$PY -u scripts/run_noise_free_v2.py --case swirl10 --names baseline hessian_0.0001 hessian_0.001 hessian_0.01 h1_0.01 h1_0.1 conditional_1 seed_1 seed_2 perturbation_0.001 perturbation_0.1 long_adam > artifacts/noise_free_v2/grid10.log 2>&1
$PY -u scripts/run_noise_free_followups.py --case swirl4 --names lbfgs_continue lbfgs_hessian_continue lbfgs_flat adam_logpen > artifacts/noise_free_v2/lbfgs4.log 2>&1
$PY -u scripts/run_noise_free_followups.py --case swirl10 --names sweep_flat sweep_hessian_continue lbfgs_hessian_continue > artifacts/noise_free_v2/follow10.log 2>&1
$PY -u scripts/run_noise_free_followups.py --case gl6 --names baseline hessian > artifacts/noise_free_v2/gl6.log 2>&1
