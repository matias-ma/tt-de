# Figures

RSI baselines and a direct-convolution regularization follow-up. All figures are PNGs.

## Strong-noise recovery follow-up

The [investigation](../../../docs/noisy_recovery.md) compares the original paired
RSI fit with a single-TT fit using integrated amplitude smoothness. Candidate
selection uses noisy validation, before evaluating clean-density errors.

- [Matched 2D marginals](ginzburg_landau/noisy_recovery/paired_comparison.png)
- [Full-density errors, with both L2 conventions](ginzburg_landau/noisy_recovery/recovery_errors.png)
- [Noisy validation curves](ginzburg_landau/noisy_recovery/validation_curves.png)

## Noisy Ginzburg–Landau

Original 10D examples, 100,000 observations, noise standard deviations scaled
by 1.5. The familiar panels show truth, clean/noisy observations, the prescribed
noise kernel, KDE, the TT initializer and the selected deconvolved TT.

| Noise | Marginal 0–1 | Marginal 2–3 | Marginal 0–9 |
| --- | --- | --- | --- |
| Mixture | [PNG](ginzburg_landau/rsi/scale_1p5/mixture/noisy_dim10_marg_01.png) | [PNG](ginzburg_landau/rsi/scale_1p5/mixture/noisy_dim10_marg_23.png) | [PNG](ginzburg_landau/rsi/scale_1p5/mixture/noisy_dim10_marg_09.png) |
| Paired | [PNG](ginzburg_landau/rsi/scale_1p5/paired/noisy_dim10_marg_01.png) | [PNG](ginzburg_landau/rsi/scale_1p5/paired/noisy_dim10_marg_23.png) | [PNG](ginzburg_landau/rsi/scale_1p5/paired/noisy_dim10_marg_09.png) |

The errors printed on GL figures compare the full density on held-out clean
samples; they are not errors of the displayed 2D marginal. Clean density values
are used only after checkpoint selection by noisy validation likelihood.
The stronger paired-noise fit recovers the marginal modes but has poor
full-density error. Its RSI convolution agrees closely with direct convolution;
numerical accuracy alone does not resolve the statistical difficulty.

## Cryo-EM

The original `noise1` path examples, with the prescribed diagonal noise model.
All model panels show exact marginals of the fitted density. Path labels are
used only for the final diagnostic.

| Dimension | Marginal 0–1 | Marginal 2–3 | Path diagnostic | Training |
| --- | --- | --- | --- | --- |
| 4D | [PNG](cryoem/rsi/cryoem_dim4_tt_marg_01.png) | [PNG](cryoem/rsi/cryoem_dim4_tt_marg_23.png) | [PNG](cryoem/rsi/cryoem_path_dim4.png) | [PNG](cryoem/rsi/cryoem_training_dim4.png) |
| 10D | [PNG](cryoem/rsi/cryoem_dim10_tt_marg_01.png) | [PNG](cryoem/rsi/cryoem_dim10_tt_marg_23.png) | [PNG](cryoem/rsi/cryoem_path_dim10.png) | [PNG](cryoem/rsi/cryoem_training_dim10.png) |

Also: [10D marginal 8–9](cryoem/rsi/cryoem_dim10_tt_marg_89.png).

[Settings and reproduction commands](../../../docs/rsi_figures.md).

## Earlier figures

`archive/` preserves figures formerly stored alongside experiment outputs.
Existing figures directly under `ginzburg_landau/` and `cryoem/` are unchanged.
Optional checkpoints and logs are kept separately in `../artifacts/`.
