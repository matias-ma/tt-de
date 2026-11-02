# RSI convolution figures

See [the figure index](../artifacts/figure_refresh/previous_figs/README.md) for PNGs. Run artifacts are optional
and live in the ignored `artifacts/` directory.

## Method and selection

These runs use `chebttge.computations.convolve_tt(method="rsi")`. Adaptive
interpolation rows are selected once, outside JAX tracing; subsequent forward
and gradient evaluations recompute interpolation with those fixed rows. The
normalizer remains the exact integral of the squared signal TT.

Both `--rsi-max-rank` and `--rsi-sketch-size` are exposed in the TT runners.
The latter sets the number of random suffix projections; a larger rank cap
alone need not resolve a sketch that misses relevant directions. The GL
benchmark's `--rsi-setup-only` option checks the initial convolution against
the direct operator before training, using only noisy observations and the
prescribed noise kernel. Frozen rows must be reused with the same sketch size.

The integrated-product convolution uses the interpolation machinery from
[the RSI paper](https://arxiv.org/abs/2602.17974). Integration couples the input
bonds, so this implementation does not inherit the paper's cubic cost for
pointwise multiplication. Its speed and accuracy must be measured separately.

GL retains the original examples, seeds, sample sizes and optimizer settings,
with noise standard deviations multiplied by 1.5. Checkpoints are selected by
held-out noisy likelihood. Cryo-EM retains the existing `noise1` path data,
diagonal prescribed covariance, signal-domain trimming, and per-dimension
rank/degree settings. Its checkpoint selection uses direct Gaussian noisy
likelihood as an independent check on the RSI objective. No clean GL density
or cryo-EM state label enters fitting or checkpoint selection.

The GL plot errors are relative errors of full-density values on clean samples,
not integrated marginal errors. Cryo-EM path errors follow the existing
notebook convention: discard endpoints and normalize estimated anchor values.
They are a diagnostic, not a density-fitting objective.

## Reproduce on Della

Follow [instructions.txt](../instructions.txt), allocate once, and reuse it:

```bash
salloc --partition=cryoem --gres=gpu:1 --cpus-per-task=4 --mem=32G --time=06:00:00
conda activate chebdGE
export XLA_PYTHON_CLIENT_PREALLOCATE=false

srun python scripts/benchmark_gl_convolution.py --dim 10 --noise mixture \
  --noise-scale 1.5 --rank 6 --n-coeffs 19 --naive-steps 0 --deconv-steps 8000 \
  --convolution-backend rsi --rsi-max-rank 64 \
  --output artifacts/gl_rsi_trial/mixture_rsi.json

XLA_PYTHON_CLIENT_PREALLOCATE=true XLA_PYTHON_CLIENT_MEM_FRACTION=0.90 \
srun python scripts/benchmark_gl_convolution.py --dim 10 --noise paired \
  --noise-scale 1.5 --rank 10 --n-coeffs 17 --kernel-rank 12 \
  --naive-steps 0 --deconv-steps 8000 --convolution-backend rsi --rsi-max-rank 256 \
  --rsi-sketch-size 24 --output artifacts/gl_rsi_trial/paired_rsi_accurate.json

srun python scripts/benchmark_gl_convolution.py --dim 10 --noise paired \
  --noise-scale 1.5 --rank 10 --n-coeffs 17 --kernel-rank 12 \
  --convolution-backend rsi --rsi-max-rank 256 --rsi-sketch-size 24 \
  --rsi-setup-only --output artifacts/gl_rsi_trial/paired_setup.json
srun python scripts/check_gl_rsi_checkpoint.py \
  artifacts/gl_rsi_trial/paired_rsi_accurate.json \
  artifacts/gl_rsi_trial/paired_setup.npz

srun python scripts/plot_gl_convolution_checkpoint.py \
  artifacts/gl_rsi_trial/mixture_rsi.json artifacts/gl_rsi_trial/paired_rsi_accurate.json \
  --figure-dir figs/ginzburg_landau/rsi/scale_1p5

srun python scripts/benchmark_cryoem_convolution.py --dim 4 --rank 4 --degree 13 \
  --steps 1500 --batch-size 2048 --backend rsi --rsi-max-rank 32 \
  --output artifacts/cryoem_rsi/path4_rsi.json
srun python scripts/benchmark_cryoem_convolution.py --dim 4 --rank 4 --degree 13 \
  --steps 1500 --batch-size 2048 --backend naive \
  --output artifacts/cryoem_rsi/path4_naive.json

srun python scripts/benchmark_cryoem_convolution.py --dim 10 --rank 8 --degree 5 \
  --steps 1500 --batch-size 2048 --backend rsi --rsi-max-rank 64 \
  --output artifacts/cryoem_rsi/path10_rsi.json
srun python scripts/benchmark_cryoem_convolution.py --dim 10 --rank 8 --degree 5 \
  --steps 1500 --batch-size 2048 --backend naive \
  --output artifacts/cryoem_rsi/path10_naive.json

srun python scripts/plot_cryoem_rsi_checkpoint.py \
  artifacts/cryoem_rsi/path4_rsi.json artifacts/cryoem_rsi/path4_naive.json
srun python scripts/plot_cryoem_rsi_checkpoint.py \
  artifacts/cryoem_rsi/path10_rsi.json artifacts/cryoem_rsi/path10_naive.json
```

`--degree` in the cryo-EM benchmark is the legacy name for the **number of
coefficients**: 13 means polynomial degree 12, and 5 means degree 4.

Run these commands sequentially on the allocation. The larger paired fit
ran out of GPU memory during its first gradient when it overlapped another
fit with preallocation disabled. The completed retry used the GPU alone and a
90% preallocated pool; preallocation reduces fragmentation as described in
[JAX's memory guide](https://docs.jax.dev/en/latest/gpu_memory_allocation.html).

For a check over all 2,048 noisy validation observations, use
`scripts/check_gl_rsi_checkpoint.py RECORD SETUP_NPZ`. The setup file is
created by `benchmark_gl_convolution.py --rsi-setup-only` with the same
example settings. This checks positivity and density/log-density errors
using the training interpolation rows; it does not refit or use clean labels.

## Completed checks

- Six tests in `tests/test_tt_convolution.py` pass, including RSI gradients and
  optimizer updates, independent convolution cubature, and prepared moments.
- The mixture GL fit's maximum log-density discrepancy from direct convolution
  is `8.28e-7`. RSI and direct clean-sample relative errors are `0.4582` and
  `0.4573`, respectively. This run establishes accurate convolution, without
  an accuracy or speed gain over the direct implementation.
- The paired GL fit completed all 8,000 steps without nonfinite losses,
  selecting step 5,000 by noisy validation likelihood. With rank cap 256 and
  sketch size 24, every one of the 2,048 noisy validation density values was
  positive and finite. Against direct convolution, the relative density RMS
  error was `8.93e-8`, the 99th-percentile absolute log error was `8.30e-6`,
  and the maximum was `1.74e-4`. RSI and direct validation NLLs differed by
  `2.11e-7`. The prescribed paired kernel's relative sample error was
  `3.79e-7`, and doubling quadrature changed log density by at most `2.38e-13`.
  The fit took 6,041 seconds, including RSI setup.
- Greater convolution accuracy did **not** fix density recovery under the
  stronger paired noise: clean-sample relative error was `1.4532`, versus
  `0.9997` for KDE and `0.9996` for the untrained TT initializer. The earlier
  rank-cap-128/default-sketch run had relative error `1.4161` but a much
  larger probe log discrepancy (`0.0875`). These are final diagnostics;
  neither the rank/sketch choice nor checkpoint selection used clean truth.
- In 4D cryo-EM, RSI and direct Gaussian convolution agree to `3.6e-14` in log
  density. Path relative errors are `0.645` for RSI deconvolution, `0.298` for
  naive TT, and `0.323` for KDE. Accurate convolution does not make
  deconvolution statistically beneficial on this example.
- In 10D cryo-EM, RSI and direct Gaussian convolution agree to `1.5e-12` in
  log density. Path relative errors are `0.333` for RSI deconvolution, `0.273`
  for naive TT, and `0.081` for KDE. Neither cryo-EM example improves on its
  naive baselines with these deconvolution settings.
