# Noisy GL, stronger mixture noise

The original 10D mixture example was rerun with its noise standard deviations
multiplied by 1.5. All other settings match the earlier selected setup:
100,000 training observations, 4,000 clean diagnostic samples, signal rank 6,
19 Chebyshev coefficients, 8,000 deconvolution steps, the same KDE
initialization, optimizer, regularization, seeds, and noisy-NLL checkpoint
selection. Clean GL values were used only after checkpoint selection.

| Convolution | Selected step | Clean sample L2 | Fit seconds |
|---|---:|---:|---:|
| Direct | 5,000 | 0.457327 | 129 |
| RSI, rank cap 64 | 5,000 | 0.458168 | 1,693 |

RSI's initial convolution error was 2.09e-9 relative RMS against the exact
represented convolution. At its selected checkpoint, the maximum log-density
difference from direct convolution on the probe batch was 8.28e-7. Thus the
RSI approximation was accurate, but did not improve this fit and was slower.

The original-format figures are [direct](../../figs/archive/gl_rsi_trial/direct/mixture/noisy_dim10_marg_01.png)
and [RSI](../../figs/ginzburg_landau/rsi/scale_1p5/mixture/noisy_dim10_marg_01.png). The JSON files retain the
configuration and training history; the NPZ files retain the selected TT cores.
