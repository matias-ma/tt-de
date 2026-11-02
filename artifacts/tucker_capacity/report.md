# Higher-rank Tucker Swirl follow-up

The initial ranks were too restrictive. A 12,000-update rank-8 control improves held-out NLL from 1.468 to 1.390; rank 16 with the same objective and update count reaches 1.239. Rank therefore matters beyond extra training time.

Rank 24 in 4D and ranks 8/10 in 6D use 20,000 updates and the TT core-sweep log-penalty objective. Noisy candidates use 4,000 updates and both strong and degree-curriculum recipes. All starts use the same observation-only box, seed and near-uniform construction; no true density or clean pilot enters fitting.

| Condition | Selected rank | Old relative L2 | New relative L2 | Old held-out NLL | New held-out NLL |
|---|---:|---:|---:|---:|---:|
| swirl4_free | 24 | 0.4330 | 0.1198 | 1.4680 | 1.2283 |
| swirl6_free | 10 | 0.8915 | 0.2799 ± 0.0039 | 2.0955 | 0.2862 |
| swirl4_mixture | 16 | 0.5133 | 0.3108 | 3.8802 | 3.8645 |
| swirl6_mixture | 6 | 0.9274 | 0.7792 ± 0.0012 | 5.1406 | 4.8330 |

The 4D full-density error uses exact blocked polynomial quadrature. The 6D fourth moment uses eight independent scrambled Sobol importance estimates (1,048,576 total samples); ± denotes one numerical standard error. The cross integral is deterministic, refined path/Gaussian quadrature. Every displayed pair marginal is exact.

The noisy 6D recovery improves but remains poor in full-dimensional L2. These results do not establish a best attainable Tucker error. In 4D, the updated clean/noisy errors are close to the TT gallery (0.116 / 0.296).

Frozen candidate hashes and observation-only choices: [selection.json](selection.json). Original figures: [baseline_figs](baseline_figs). Published figures: [Tucker index](../../figs/TUCKER.md).

Each fit is reproduced with `scripts/run_tucker_gallery.py --action fit --root artifacts/tucker_capacity/ROOT --family swirl --dim D --rank R --noise free|mixture --steps N --objective nll|logpen`; noisy runs also use `--microbatches 4` and both `--recipe strong` and `--recipe curriculum`. Full per-run settings are in each frozen `config.json`. Publish with `scripts/publish_tucker_capacity.py` after fitting.

TT remains the default. Its fits and figures are unchanged.

The exact coordinate-pair contraction is 2.45× faster than sequential contraction in the rank-16, 49-mode, batch-16 real-kernel value/gradient benchmark. Maximum absolute gradient difference: 2.4e-15. These are warm microbenchmark timings on the allocated GPU; see [record](pair_benchmark.json).

The prepared polynomial basis also uses an exact doubling recurrence, with value and endpoint-gradient regression tests. [Data equivalence checks](data_equivalence.json) confirm identical observations, domains, and noise covariances across the original and updated comparisons.
