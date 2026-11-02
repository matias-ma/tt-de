#!/bin/bash
set -euo pipefail
export JAX_ENABLE_X64=1
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export JAX_COMPILATION_CACHE_DIR=/tmp/ftde_v3_jax
export CUDA_CACHE_PATH=/tmp/ftde_v3_cuda
export XDG_CACHE_HOME=/tmp/ftde_v3_cache
export MPLCONFIGDIR=/tmp/ftde_v3_mpl
export OPENBLAS_NUM_THREADS=4
export OMP_NUM_THREADS=4
exec "${PYTHON:-python}" "$@"
