# Density estimation: 4D, 10D, 20D

Fresh single squared-Chebyshev-TT fits from near-uniform starts. Full-Hessian regularization is the default. Noise kernels retain the earlier benchmark scales, including the correlated Swirl mixture.

| Benchmark | Dimension | Noise-free relative L2 | Noisy relative L2 |
|---|---:|---:|---:|
| Ginzburg–Landau | 4 | [0.0367](ginzburg_landau/noise_free_dim4_marg_01.png) | [0.1672](ginzburg_landau/noisy_gl4_mixture_marg_01.png) |
| Ginzburg–Landau | 10 | [0.0815](ginzburg_landau/noise_free_dim10_marg_01.png) | [0.2892](ginzburg_landau/noisy_gl10_mixture_marg_01.png) |
| Ginzburg–Landau | 20 | [0.1027](ginzburg_landau/noise_free_dim20_marg_01.png) | [0.3751](ginzburg_landau/noisy_gl20_mixture_marg_01.png) |
| Swirl | 4 | [0.1155](swirl/noise_free_dim4_marg_01.png) | [0.2961](swirl/noisy_swirl4_mixture_marg_01.png) |
| Swirl | 10 | [0.2272](swirl/noise_free_dim10_marg_01.png) | [0.5909](swirl/noisy_swirl10_mixture_marg_01.png) |
| Swirl | 20 | [0.2561](swirl/noise_free_dim20_marg_01.png) | [0.7810](swirl/noisy_swirl20_mixture_marg_01.png) |

These are unsquared **full-dimensional latent** errors, evaluated after observation-only choices were frozen. Each benchmark folder contains two pair marginals and a convergence figure per condition. True and estimated latent densities share their color scale; reconvolved fits and observed densities share another.

Cryo-EM retains its previous 4D/10D/20D noisy fits and training figure; no clean-density error is available. Its fits were not rerun in this synthetic matrix.

See [protocol and interpretation](../docs/dimension_gallery.md), [error comparison](convergence/dimensional_errors.png), and [archived previous gallery](../artifacts/gallery_archive/before_dimension_gallery_v9/). Records, models and logs live in `artifacts/dimension_gallery_v9`.

The [20D joint-tube diagnostic](convergence/joint_tube_recovery.png) compares samples from the frozen clean and noisy estimates with the true Swirl tube.

## Tucker comparison

The separate [Tucker gallery](TUCKER.md) reports feasible-rank GL, Swirl and
Cryo-EM fits. Tucker figures use `tucker_` prefixes; the TT results above
remain the default.
