#!/bin/bash
set -e
export JAX_ENABLE_X64=1 JAX_COMPILATION_CACHE_DIR=/tmp/ftde_v2_jax CUDA_CACHE_PATH=/tmp/ftde_v2_cuda XDG_CACHE_HOME=/tmp/ftde_v2_cache MPLCONFIGDIR=/tmp/ftde_v2_mpl XLA_PYTHON_CLIENT_PREALLOCATE=false OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
PY=/home/ma4473/.conda/envs/chebdGE/bin/python
$PY -u scripts/run_noise_free_followups.py --case swirl4 --names sweep_hessian_flat_long sweep_hessian_strong_flat_long > artifacts/noise_free_v2/sweep_long4.log 2>&1
$PY -u scripts/run_noise_free_followups.py --case swirl10 --names sweep_hessian_flat_long sweep_hessian_medium_flat_long sweep_hessian_strong_flat_long > artifacts/noise_free_v2/sweep_long10.log 2>&1
$PY -u scripts/run_noise_free_followups.py --case swirl10 --names adam_h1_long_continue adam_hessian_long_continue > artifacts/noise_free_v2/long10.log 2>&1
$PY -m pytest tests/test_direct_training.py tests/test_core_sweeps.py -q > artifacts/noise_free_v2/final_tests.log 2>&1
$PY -u scripts/select_noise_free_v2.py swirl4 swirl10 gl6 > artifacts/noise_free_v2/selection.log 2>&1
$PY -u scripts/evaluate_noise_free_v2.py swirl4 swirl10 gl6 > artifacts/noise_free_v2/evaluation.log 2>&1
$PY -u scripts/run_noise_free_cryoem.py dim4 path_dim4 dim10 dim20 > artifacts/noise_free_v2/cryoem.log 2>&1
