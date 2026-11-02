# RSI convolution in noisy Ginzburg–Landau training

Figures using the same noisy-GL panel layout as the earlier results:
[direct](figs/direct/mixture/noisy_dim10_marg_01.png),
[RSI cap 32](figs/rsi_cap32/mixture/noisy_dim10_marg_01.png), and
[RSI cap 64](figs/rsi_cap64/mixture/noisy_dim10_marg_01.png).
They were rendered with `scripts/plot_gl_convolution_checkpoint.py`.

`scripts/benchmark_gl_convolution.py` now accepts `--convolution-backend rsi`
and `--rsi-max-rank`. The RSI path constructs the squared signal, selects
interpolation rows once from the initial model outside JAX tracing, and uses
those fixed rows for differentiable convolution during training. Its output
domain covers every training and validation observation. The direct backend
and all other training settings are unchanged.

These runs used one cryoem GPU allocation (one GPU, four CPUs, 32 GB), the
`chebdGE` environment, the same seeded GL observations and KDE initialization,
the same Gram-preconditioned optimizer and cosine schedule, and checkpoint
selection by held-out noisy NLL. The clean GL density was evaluated only after
selection. Sample L2 below is the square root of the squared relative error
at 1,000 held-out clean GL samples, not an integrated Lebesgue L2 norm.

| 10D mixture, 100k observations, signal rank 6, 19 coefficients, 1000 steps | Selected step | Clean sample L2 ↓ | Exact held-out noisy NLL ↓ | Fit seconds ↓ | Maximum RSI vs direct log-density error |
|---|---:|---:|---:|---:|---:|
| Direct | 1000 | 0.210951 | 13.665852 | 20.18 | — |
| RSI, cap 32 | 800 | 0.248003 | 13.674897 | 234.70 | 0.2511 |
| RSI, cap 64 | 1000 | 0.210954 | 13.665852 | 241.06 | 2.10e-7 |

The cap-32 RSI checkpoint had a lower *approximate* validation NLL (13.663643)
than direct, but its exact NLL was worse (13.674897). Its approximate NLL
also rose sharply at step 1000. Raising the cap to 64 made the convolution
effectively agree with direct and removed the fit gap, at about 12 times the
fit time. The initial RSI validation relative RMS errors were 0.00436 and
2.47e-9 for caps 32 and 64. RSI did not improve this GL fit.

A smaller matched check (10D, 20k observations, rank 4, nine coefficients,
500 steps, 500 held-out samples) gave clean sample L2 0.960999 for direct
versus 0.961047 for RSI cap 16. Fit times were 8.67 and 35.87 seconds.

The JSON and NPZ files here contain the exact settings, training histories,
checks, and selected coefficients. The representative commands were:

```bash
python scripts/benchmark_gl_convolution.py --dim 10 --noise mixture \
  --n-train 100000 --n-test 1000 --rank 6 --n-coeffs 19 \
  --kernel-n-coeffs 257 --naive-steps 0 --deconv-steps 1000 \
  --batch-size 512 --convolution-backend direct \
  --output results/gl_rsi_trial/gl10_r6n19_1000_direct.json
python scripts/benchmark_gl_convolution.py --dim 10 --noise mixture \
  --n-train 100000 --n-test 1000 --rank 6 --n-coeffs 19 \
  --kernel-n-coeffs 257 --naive-steps 0 --deconv-steps 1000 \
  --batch-size 512 --convolution-backend rsi --rsi-max-rank 64 \
  --output results/gl_rsi_trial/gl10_r6n19_1000_rsi64.json
```
