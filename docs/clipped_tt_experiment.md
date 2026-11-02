# Direct density TT with clipping during optimization

This experiment follows the request to optimize an unsquared functional TT
and use `max(f_TT, 0)` for nonnegativity. The learned Gaussian-mixture option
remains available, but no learned signal mixture is used in these trials.

The fitted model is

\[
p_\theta(x)=\frac{\max(f_\theta(x),0)}{\int_D\max(f_\theta(z),0)\,dz},\qquad
q_\theta(y)=\int_D p_\theta(x)u(y-x)\,dx.
\]

Both training and validation use this clipped, normalized, convolved model.
Clipping the observation density, clipping displayed marginals, or fitting a
signed density and clipping afterward would optimize different models. The
earlier `test_cryoem_unsquared.py` experiment did the latter; this is a new
objective, not a rerun of that surrogate.

## Results

**Removing the square and clipping did not improve the tested 4D recovery.**
The best noisy-validation candidate learned from the KDE initializer has
full-dimensional relative L2 **0.3371**, versus **0.2162** for the original
squared estimator. Its small validation advantage does not reproduce on
the independent observations. Both warm-started fits retain **step zero**;
the subsequent 3,000 optimization steps do not improve their validation NLL.

All clipped values below use the finer 129-node integration. Checkpoint
selection used the training integration order, before these audits.

| Candidate | Selected step | Noisy validation NLL | Full 4D relative L2 |
|---|---:|---:|---:|
| Original squared TT | previous selection | 3.861538 | 0.216188 |
| Direct r32, original H2 normalization | 1200 | 3.861182 | 0.377340 |
| Direct r32, positive-part H2 normalization, selected | 2800 | 3.860238 | 0.337147 |
| Direct r64, H2 1e-4, negative-mass penalty .001 | 2000 | 3.862973 | 0.336267 |
| Direct r64, H2 1e-5, negative-mass penalty .001 | 2000 | 3.862425 | 0.298614 |
| Recovery warm start, r64, H2 1e-4 | 0 | 3.862015 | 0.216377 |
| Recovery warm start, r64, H2 1e-5 | 0 | 3.862015 | 0.216377 |

For the selected direct fit, the paired log-likelihood gain over the original
is **+0.001300 ± 0.000722** on validation and **−0.000109 ± 0.000562** on the
independent test set (one standard error). This does not establish a noisy
prediction improvement, and the known clean-density error is worse.

The selected raw function discards about 11.43 times as much negative mass
as it retains positive mass. This is not negative probability in the fitted
model: its positive part is explicitly normalized throughout. It shows how
different the raw TT can become when only its positive regions contribute.
The negative-mass penalty reduces this ratio but does not beat the original
clean error in the tested trials.

The selected fit's 97-to-129-node changes are only 2.99e-6 in validation NLL
and 2.07e-6 in relative L2. Across all six candidates, the corresponding
maximum changes are 3.18e-6 and 3.80e-5. The regression is not explained by
coarse numerical integration.

This qualifies the earlier amplitude-zero diagnosis. Squaring explains why
an amplitude sign crossing forces a density hole; it does **not** establish
that removing the square will fix fitting. Hard clipping can also create
zero-density regions, and its derivative is zero wherever the raw function
is negative. The experiments support keeping the square's role in
nonnegativity separate from the unresolved optimization and deconvolution
issues. They do not rule out a better direct-density optimizer, nor establish
what would happen in 10D or 20D.

The existing gallery was not replaced with these weaker recoveries. Three
additional figures show the density comparison, training/test likelihood,
and marginal ridge profiles in `figs/swirl`. No cryo-EM rerun was triggered
because this change did not improve swirl.

## Implementation and scope

`src/ftde/clipped_tt_density.py` implements joint Gauss--Legendre integration
in four dimensions. It supports prescribed Gaussian noise mixtures whose
covariances are block diagonal in coordinate pairs, including the swirl's
correlated component. The noise component label is shared across pairs.

Clipping destroys the separability that makes the original squared-TT
normalizer and convolution tractable. This implementation therefore makes
an explicit **4D-only** restriction. It is a reference experiment, not a
scalable 10D/20D implementation. At 97 nodes per axis the joint grid already
has about 88.5 million values; naively extending that grid is not practical.
No 10D/20D clipped fits or cryo-EM improvements are claimed.

The original squared estimator and the optional signal-mixture code are
preserved. `ClippedTTQuadrature4D` is exported as a separate option. A direct
density checkpoint has the same coefficient file format as an amplitude
checkpoint, so it must be evaluated through the clipped operator, **not**
through `density_tt`, which would square it again.

## Training comparisons

Six fits use the existing 100,000 training observations, 4,096 validation
observations, latent domain, and prescribed noise from
`artifacts/swirl/swirl_mixture_dim4`. Neither training nor checkpoint
selection evaluates the target density or its curve.

Four fits start from the original KDE-based initializer. Its existing square
is materialized once and physically compressed into a direct density TT;
subsequent trainable coefficients parameterize the density function itself.
The ranks are 32 or 64, with 49 modes. Training uses 65 or 81 quadrature
nodes per axis, 1,600 or 4,000 Adam steps, and batches of 256 or 1,024.

Two further fits start from the original recovered squared-TT density,
converted into a direct rank-64, 65-mode TT. The physical relative L2 change
of representation is 0.1643% before clipping/renormalization; polynomial
projection alone contributes 0.01895%. This is a conversion of a previously
data-fitted TT, not a Gaussian-mixture initialization. These fits use 97
quadrature nodes, 3,000 steps, batch size 2,048, and learning rate 2e-5
decaying to 2e-6.

The first trial reused the old relative H2 penalty. It revealed a weakness:
normalizing raw derivative energy by the raw function's squared norm lets
a large discarded negative part affect the regularizer's scale. Subsequent
trials divide by the positive part's squared norm instead. Two also penalize
the ratio of discarded negative mass to retained mass, with strength .001.
These are regularizers of the raw function; none changes the definition of
the clipped probability density.

Checkpoint selection minimizes noisy validation NLL. The final candidate
selection is frozen in `artifacts/clipped_tt/selection.json` before the
remaining target-density evaluations. The first exploratory trial's clean
error is also retained; it is not used as a selection criterion.

## Validation and reproducibility

The selected checkpoints are reintegrated at 97 and 129 nodes per axis.
Full-dimensional clean L2 uses the actual clipped grid, the independent
analytic target norm, and a 1,600-node path integral evaluated only after
fitting. It is not the error of a two-dimensional picture.

An independent 8,192-observation test set uses seed 73129. Paired likelihood
differences compare each clipped candidate and the original squared model
on exactly the same rows. This set is not used for training or selection.

Tests cover positive and discarded mass, marginal normalization, support,
independent truncated-normal convolution formulas, correlated noise-mixture
convolution, finite-difference likelihood gradients, and the conversion of
an independently known squared TT into a direct TT.

The seven clipped/linear-convolution numerical tests pass. The broader run
has 14 passes and one failure in the existing 5D correlated-Gaussian squared
estimator accuracy test (mean absolute log error 0.5734 versus its 0.4
threshold). An isolated copy with the new clipped export removed reproduces
the identical failure and value; the original squared implementation and
that test were not changed. Logs are `tests_full.log` and
`test_preexisting_5d.log` under `artifacts/clipped_tt/logs`. Ruff passes for
the new and modified implementation files.

Example fit from the original KDE initializer:

```bash
python scripts/fit_clipped_tt.py \
  --base artifacts/swirl/swirl_mixture_dim4 \
  --output artifacts/clipped_tt/reproduce_cold \
  --rank 32 --order 81 --steps 4000 --strength 1e-5
```

Example warm start and continuation:

```bash
python scripts/initialize_direct_tt_grid.py \
  --source artifacts/swirl/swirl_mixture_dim4/selected.npz \
  --output artifacts/clipped_tt/reproduce_initializer.npz --rank 64 --modes 65
python scripts/fit_clipped_tt.py \
  --base artifacts/swirl/swirl_mixture_dim4 \
  --initial artifacts/clipped_tt/reproduce_initializer.npz --initial-is-density \
  --output artifacts/clipped_tt/reproduce_warm \
  --rank 64 --order 97 --steps 3000 --batch-size 2048 --lr .00002 --strength 1e-5
```

All commands use the `chebdGE` environment. Computational outputs and logs
stay in `artifacts/clipped_tt`; only figures go into `figs/swirl`.
