# Rank 10 for noisy 6D Tucker

The earlier cutoff at rank 6 was a conservative run-budget choice, not a demonstrated hardware or mathematical limit. Rank 4 was the original baseline. Clean point evaluation reduces one core mode at each contraction; noisy evaluation applies quadratic moment matrices while retaining the full core.

Both runs below use exactly the same observations, latent box, prescribed noise kernel, 49 polynomial modes, log-penalty objective and weak-penalty degree-curriculum recipe. The recipe was fixed from the previous held-out winner before rank-10 fitting. Rank 10 uses 8,000 updates instead of 4,000 and the same batch of 256 observations.

| Noisy 6D fit | Core entries | Updates | Held-out NLL | Full latent relative L2 |
|---|---:|---:|---:|---:|
| Rank 6 | 46,656 | 4,000 | 4.832957 | 0.7792 ± 0.0012 |
| Rank 10 | 1,000,000 | 8,000 | 4.758941 | 0.5065 ± 0.0047 |

The published noisy 6D model comes from `artifacts/tucker_noisy_rank10/r10/swirl6_mixture`, selected by held-out NLL before evaluating the new fit against truth. The ± values are numerical integration standard errors, not statistical uncertainty of the fitted model.

Rank 10 has a 7.63 MiB dense core, versus 0.36 MiB at rank 6. The expensive part is batched noisy intermediates and reverse-mode differentiation. Pair contraction work grows approximately as rank^(dimension+2), while its core-sized intermediate storage grows as batch × rank^dimension, times the remaining noise bond. Neither formula includes fixed polynomial-moment preparation costs.

Independent kernel mixture components now use separate exact pair contractions. Contributions sharing the final output bond are summed; no approximation or rank truncation is introduced. Tests compare values and gradients with dense paired and sequential contractions, including four- and six-dimensional mixtures.

[Profiling measurements](profiles) report warmed value-gradient times and compiled temporary storage for the actual prescribed kernel. These are not process peak-memory measurements or full Adam iteration timings.

[Current figures](../../figs/TUCKER.md) · [Rank-10 figures](r10/figs/swirl) · [Previous rank-6 figures](previous_figs) · [Earlier rank comparison](../tucker_capacity/report.md)

Reproduce on the allocated GPU with `scripts/run_tucker_noisy_rank10.py`, then `scripts/publish_tucker_noisy_rank10.py`. TT remains the default; its figures are unchanged.

| Rank | Likelihood | Warm value + gradient (16 observations) | Compiled temporary storage |
|---|---|---:|---:|
| 6 | clean | 1.15 ms | 13.3 MiB |
| 6 | noisy | 8.34 ms | 117.9 MiB |
| 10 | clean | 3.03 ms | 320.4 MiB |
| 10 | noisy | 15.85 ms | 1642.3 MiB |

The measured rank-10 full-batch training run used roughly 40 GiB of GPU allocation on an 80 GiB H100. That is distinct from the 32 GiB host-memory allocation and from the compiled-temporary figures above. Smaller microbatches trade runtime for GPU memory; the rank-10 limit was not a lack of mathematical support for convolution.
