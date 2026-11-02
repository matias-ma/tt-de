# Tucker density figures

TT remains the default. Native Tucker fits use the same squared-density likelihood and full-Hessian penalty.
Mode ranks are limited by the dense core; see [protocol and implementation](../docs/tucker_density.md).

| Benchmark | Dimension | Mode rank | Clean relative L2 | Noisy relative L2 |
|---|---:|---:|---:|---:|
| GL | 4 | 5 / 5 | [0.0392](ginzburg_landau/tucker_noise_free_dim4_marg_01.png) | [0.2561](ginzburg_landau/tucker_noisy_dim4_marg_01.png) |
| GL | 8 | 3 / 3 | [0.0975](ginzburg_landau/tucker_noise_free_dim8_marg_01.png) | [0.2003](ginzburg_landau/tucker_noisy_dim8_marg_01.png) |
| Swirl | 4 | 24 / 24 | [0.1137](swirl/tucker_noise_free_dim4_marg_01.png) | [0.2246](swirl/tucker_noisy_dim4_marg_01.png) |
| Swirl | 6 | 10 / 10 | [0.2799 ± 0.0039](swirl/tucker_noise_free_dim6_marg_01.png) | [0.5065 ± 0.0047](swirl/tucker_noisy_dim6_marg_01.png) |

Each condition has two pair-marginal figures (`marg_01`, `marg_23`) and a convergence plot, with `tucker_` prefixes.
Errors are unsquared full-dimensional latent L2, evaluated only after held-out selection.

[Dimensional error comparison](convergence/tucker_dimensional_errors.png).
Models, protocols, numerical checks and logs: `artifacts/tucker_gallery`.

Ranks are listed as clean / noisy. Swirl uses the higher-rank follow-up; the original low-rank fits remain archived.
The ± values are numerical standard errors from randomized importance quadrature in 6D; pair marginals remain exact.
[Rank and training comparison](../artifacts/tucker_4d_tuning/report.md).

[4D Cryo-EM marginals](cryoem/tucker_dim4.png) · [training](cryoem/tucker_training.png). No clean-density error is available.
