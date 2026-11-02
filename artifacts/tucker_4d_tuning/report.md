# 4D Swirl Tucker tuning

Higher ranks, a Hessian-penalty sweep, longer noisy training, and larger-batch refinement were tested on the same observations, domain, polynomial basis and prescribed noise kernel as TT. TT remains the default and its fits and figures are unchanged.

The original experiment plan was saved before fitting. Three clean fits use rank 32, 20,000 updates and Hessian strengths 0.003, 0.01 and 0.03. After measuring noisy rank-24 runtime, the noisy plan was amended before truth evaluation: three fits now start from an exact rank expansion of the published rank-16 Tucker fit, with 4,000 additional updates at strengths 0.00001, 0.0001 and 0.001. All use 49 modes and the log-penalty objective. The interrupted noisy scratch run remains archived and is not a candidate.

For each condition, the best held-out fit (including the prior published Tucker fit) then receives lower-learning-rate refinement: 5,000 clean updates with batch 4,096; three independent noisy branches of 2,000 updates with batch 256 and marginal-curvature strengths 0, 0.0003 and 0.001. Clean selection minimizes held-out NLL. The main noisy selection uses the same rule as TT: minimum marginal curvature among candidates within one paired standard error of the best observation NLL. The strict minimum-NLL choice is also frozen and reported below. This selection amendment was saved before evaluating any new fits against truth. Baseline truth errors were known before this development follow-up.

| Condition | TT relative L2 | Previous Tucker | Tuned Tucker | Tuned rank |
|---|---:|---:|---:|---:|
| clean | 0.115548 | 0.119776 | 0.113680 | 24 |
| noisy | 0.296052 | 0.310783 | 0.224603 | 24 |

The noisy strict-minimum-NLL control has relative L2 0.220842 and held-out NLL 3.862005, from `artifacts/tucker_4d_tuning/noisy_refine_m0/swirl4_mixture`. It is reported as a control, not selected by its truth error.

L2 values are unsquared full-dimensional latent relative errors, not errors of projected marginals. The 4D Tucker self-integral uses degree-exact blocked quadrature; the cross integral is checked by doubling path quadrature from 4,000 to 8,000 points.

| Condition | TT held-out NLL | Tucker held-out NLL | Tucker − TT | Paired SE |
|---|---:|---:|---:|---:|
| clean | 1.229805 | 1.227535 | -0.002270 | 0.001537 |
| noisy | 3.863900 | 3.862347 | -0.001553 | 0.001091 |

Lower values are better. Paired standard errors describe observation-wise likelihood differences on the same 4,096 validation observations; they are not adjusted for model selection and are not an independent test-set significance claim. This compares the newly tuned Tucker fits with the existing TT gallery, not equal tuning budgets or parameter counts. Noisy observation likelihood need not order latent-density error identically.

| New candidate | Rank | Hessian strength | Marginal strength | Best held-out NLL |
|---|---:|---:|---:|---:|
| clean_r32_h0.003 | 32 | 0.003 | 0 | 1.245806 |
| clean_r32_h0.01 | 32 | 0.01 | 0 | 1.253094 |
| clean_r32_h0.03 | 32 | 0.03 | 0 | 1.269830 |
| noisy_r24_h1e-05 | 24 | 1e-05 | 0 | 3.863079 |
| noisy_r24_h0.0001 | 24 | 0.0001 | 0 | 3.862872 |
| noisy_r24_h0.001 | 24 | 0.001 | 0 | 3.863302 |
| clean_refine | 24 | 0.01 | 0 | 1.227535 |
| noisy_refine_m0 | 24 | 0.0001 | 0 | 3.862005 |
| noisy_refine_m0.0003 | 24 | 0.0001 | 0.0003 | 3.862021 |
| noisy_refine_m0.001 | 24 | 0.0001 | 0.001 | 3.862347 |

Selected checkpoints:

- clean: `artifacts/tucker_4d_tuning/clean_refine/swirl4_free`, checkpoint 750/5000 in its final training stage.
- noisy: `artifacts/tucker_4d_tuning/noisy_refine_m0.001/swirl4_mixture`, checkpoint 400/2000 in its final training stage.

Exactly separable noise-mixture blocks now use successive factor-sized actions instead of a dense pair matrix. The real-kernel benchmark agrees with the previous calculation to about 1e-15 in gradients and measures about a 1.25× value-and-gradient speedup at rank 24 on the allocated A100. Rank expansion zero-pads the old core and adds orthogonal factor directions; it preserves the initial represented function. These are computational changes, not convolution approximations.

[Current figures](../../figs/TUCKER.md) · [Previous 4D figures](previous_figs) · [Frozen selection](selection.json) · [Machine-readable comparison](comparison.json)

Reproduce with `scripts/tune_tucker_4d.py`, then `scripts/publish_tucker_4d_tuning.py`, using `scripts/run_noise_diagnosis_gpu.sh` on an allocated GPU.
