# Dimension-10 noisy Ginzburg–Landau rerun (2026-09-22)

## Method and limitation

Each run uses the repository's Ginzburg–Landau sampler, 2,048 separate noisy
validation observations, and 4,000 common clean diagnostic samples. The noisy
training set contains 20,000 or 100,000 observations. Checkpoints are selected
by noisy validation NLL every 200 steps. All runs set `--naive-steps 0` and
use 8,000 deconvolution steps, except the 20k pilot (4,000 steps). The
"naive" metric is the untrained KDE-initialized TT, not a fitted naive NLL
estimator. Rank, polynomial degree, and regularization were varied after the
first results. This is a single-seed exploratory sweep, and the same clean
diagnostic sample was used to compare configurations.

The fits use `TTConvolution`, which directly contracts two signal cores with
the prescribed noise TT. At the time of these runs, RSI construction failed
inside the JAX training trace because row selection used host NumPy. These
results therefore measure the direct contraction. The later fixed-row RSI
training comparison is in [the RSI trial](../gl_rsi_trial/summary.md).

## Results

`n_coeffs` counts Chebyshev coefficients, so polynomial degree is
`n_coeffs - 1`. "Sample L2" is the square root of the project's relative
squared error evaluated at clean GL-distributed points; it is distinct from
the integrated Lebesgue L2 error. Lower is better.

| Noise | Train points | Signal rank | Coefficients | Penalty multiplier | Selected step | Sample L2 |
|---|---:|---:|---:|---:|---:|---:|
| mixture | 20,000 | 8 | 17 | 1 | 600 | 0.2780 |
| mixture | 100,000 | 8 | 17 | 1 | 2,600 | 0.2130 |
| mixture | 100,000 | 6 | 17 | 1 | 5,000 | 0.1849 |
| **mixture** | **100,000** | **6** | **19** | **1** | **5,000** | **0.1439** |
| mixture | 100,000 | 6 | 19 | 0.5 | 5,000 | 0.1443 |
| mixture | 100,000 | 6 | 21 | 1 | 5,000 | 0.1677 |
| paired | 100,000 | 6 | 19 | 1 | 5,000 | 1.6913 |
| paired | 100,000 | 8 | 17 | 1 | 5,000 | 0.4286 |
| paired | 100,000 | 8 | 19 | 1 | 5,000 | 1.6384 |
| **paired** | **100,000** | **10** | **17** | **1** | **5,000** | **0.4055** |
| paired | 100,000 | 12 | 17 | 1 | 5,000 | 0.4360 |

The two selected 100k fits have untrained-naive sample L2 errors of 0.9966
(mixture) and 0.9988 (paired). Deterministic integrated relative L2 errors
are 0.1782 (mixture rank 6, 19 coefficients) and 0.3971 (paired rank 10,
17 coefficients). The alternative mixture 21-coefficient fit scores 0.2067
integrated; paired rank 8, 17 coefficients scores 0.4030. Doubling the
integrated quadrature order from 64 to 128 changed the underlying integrals
by at most 7.9e-13 relative to the reference `integral p^2` across those
checked fits.

The mixture fit's general TT-kernel convolution agreed with an analytic
Gaussian-mixture operator within 1.2e-12 in log density on probe points.
The rank-12 paired noise kernel had sampled relative error 1.65e-6 against
the prescribed correlated Gaussian, with no negative values on 2,000 probe
points. Convolution quadrature-doubling errors were below 5.5e-13 in log
density for all runs. The paired inverse fit remains substantially harder:
raising degree from 16 to 18 produced a large clean-density error despite
similar noisy validation NLL. Kernel approximation error does not explain
that outcome.

## Artifacts and reproduction

Each configuration has a JSON record and an NPZ with its selected TT cores
in this directory. Integrated checks are in `*_integrated_l2.json`. The
selected 2D marginal figures are in `figs/mixture/` and `figs/paired/`.
Every panel, including the noise-kernel marginal, uses the same x and y
display bounds from the fitted signal domain.

The key runs can be reproduced on a GPU in the `chebdGE` environment:

```bash
python scripts/benchmark_gl_convolution.py --dim 10 --noise mixture --n-train 100000 --rank 6 --n-coeffs 19 --naive-steps 0 --deconv-steps 8000 --output results/gl_rsi_followup/gl10_mix_100k_r6_n19.json
python scripts/benchmark_gl_convolution.py --dim 10 --noise paired --kernel-rank 12 --n-train 100000 --rank 10 --n-coeffs 17 --naive-steps 0 --deconv-steps 8000 --output results/gl_rsi_followup/gl10_paired_100k_r10_n17_k12.json
python scripts/evaluate_gl_l2.py results/gl_rsi_followup/gl10_mix_100k_r6_n19.json results/gl_rsi_followup/gl10_paired_100k_r10_n17_k12.json
python scripts/plot_gl_convolution_checkpoint.py results/gl_rsi_followup/gl10_mix_100k_r6_n19.json results/gl_rsi_followup/gl10_paired_100k_r10_n17_k12.json --pair 0 1
```
