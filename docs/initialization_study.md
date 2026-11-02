# Initialization sensitivity of the squared Chebyshev TT

> Historical report: the fitted-pilot implementations and study scripts have
> been removed. Commands and implementation references below describe the
> retired experiment, not runnable current APIs. Saved results remain under
> `artifacts/initialization_study`; see the
> [current direct-optimization workflow](noise_free_optimization.md).

This study keeps `p = a_TT² / integral(a_TT²)` and the existing Chebyshev
coefficient cores and prescribed-noise TT convolution. It changes how the
amplitude starts. A fitted Gaussian mixture, when used, is only a pilot:
the model subsequently optimized and evaluated is an ordinary squared TT.

The practical general-purpose candidate is a **noise-corrected moment
Gaussian**, using sample-guided TT conversion. Ordinary Gaussian conversion
is close; neither dominates clean-density recovery on every case. For the
swirl, a tangent-mixture pilot is better: median 10D relative L2 drops from
0.696 for KDE to 0.368 at the same TT capacity and update budget. That gain
comes from the initial TT, which subsequent optimization does not improve.
See the [measured results](#results) and
[comparison figure](../artifacts/gallery_archive/before_noise_free_v2/convergence/initialization_recovery.png).

The protocol is recorded in `artifacts/initialization_study/plan.json` before
latent-density evaluation. Numerical checkpoints and diagnostics stay under
that directory. Figures are added to the existing `figs/convergence`,
`figs/ginzburg_landau`, and `figs/swirl` directories; earlier PNGs are retained.

## Controlled comparison

Eight cases cover six families: correlated Gaussian (4D), four anisotropic
clusters (4D), two independent banana-shaped pairs (4D), a noisy ring with
independent remaining coordinates (4D), Ginzburg–Landau (6D and 10D), and the
swirl–pretzel (4D and 10D).

The six lower-dimensional cases share a prescribed two-component diagonal
Gaussian noise mixture. The two 10D cases retain their earlier prescribed
noise: GL's isotropic scale mixture and the swirl's mixture with correlated
coordinate pairs. Every case uses 30,000 training observations. The 10D cases
reuse the first 30,000 observations from their existing training splits;
their old validation observations remain separate. Other cases use new
independent training and validation seeds.

Eight initialization recipes are repeated with seeds 0, 1, and 2:

| Recipe | Construction |
|---|---|
| Random | Smooth random Chebyshev coefficients, then normalization and orthogonalization |
| Moment Gaussian | Estimate mean and covariance from observations, subtract the prescribed noise covariance, floor unresolved covariance eigenvalues, and cross-approximate the square-root Gaussian |
| Gaussian, guided TT | The same moment Gaussian, converted using pilot means and draws to guide Chebyshev fibers |
| KDE | Existing training-selected diagonal-bandwidth KDE, cross-approximated as a square root; the library's standard 5,000-center computational cap |
| Diagonal EM | 32-component signal mixture with diagonal covariances, fitted through the known noise, then separable-amplitude projection and physical TT compression |
| Tangent EM | 32-component signal mixture with learned local directions and one shared transverse variance, fitted through the known noise, then quadrature, amplitude projection, and TT compression |
| Full-covariance EM | 32-component signal mixture with full covariance matrices, fitted through the known noise, then square-root TT cross-approximation |
| Full EM, guided TT | The same full-covariance pilot and seed, converted using Chebyshev fibers with suffix samples drawn from that pilot |

EM uses 150 iterations, retaining the best noisy-validation checkpoint.
Early throughput runs used a 2,000-center KDE cap. Those KDE starts and fits
are repeated at 5,000 centers before selection and truth evaluation; the
earlier artifacts remain under `legacy_kde2000_*` for transparency. No other
recipe or optimizer setting changes in this amendment.
Mixture projection is part of the initialization recipe: its approximation
error can change the initial TT likelihood. The study scores the *actual
converted TT*, not just the pilot. Tangent and diagonal recipes use the
previously implemented positive separable-amplitude conversion; the general
full-covariance recipe uses functional cross-approximation. Consequently,
this compares complete usable recipes, not only covariance parameterizations.
The `full_guided` ablation changes only conversion: pilot component means
and 128 pilot draws guide the suffix columns of a one-pass row-interpolative
TT cross. Truncated SVD and pivoted QR choose interpolation rows. This avoids
relying on a uniform box search to locate a narrow density, but does not
guarantee a globally accurate approximation. It was added before truth
evaluation, and earlier cases are backfilled under the same protocol.

A second conversion control, `gaussian_guided`, was added before independent
test or truth evaluation after the 10D swirl showed a large construction
loss: its single Gaussian pilot scored NLL 9.850826, but the converted TT
scored 13.032294. It uses the identical moment Gaussian with sampled-suffix
TT cross, without extra mixture components. All eight cases and three seeds
are backfilled; the other settings are unchanged.

All starts have maximum rank 32 and 49 Chebyshev modes. Physical-L2 QR
orthogonalization and unit amplitude norm standardize core scaling. Small
perturbations activate missing ranks. This convention does not imply that
all possible internal orthogonal gauge choices are eliminated.

Each start receives the same 1,200 Adam updates, the same sequence of
128-observation minibatches, learning rate `1e-3` with cosine decay, amplitude
H² strength `1e-4`, and Sobolev preconditioning. Checkpoints are compared every
200 steps using noisy validation NLL plus the fixed smoothness penalty.
Step zero is eligible. The last iterate is also retained, so a selected
step-zero result cannot be mistaken for successful TT refinement.

These are initialization-seed repetitions on fixed observations, not three
independent training datasets. They measure initialization and construction
variability under a fixed computational budget; they do not prove which
initializer is optimal after arbitrarily long optimization.
Construction/training failures remain recorded and are not silently retried
on resume. A failed seed gives that recipe an infinite mean selection score
for the case; recovery summaries identify missing seeds explicitly.

For the two 10D cases, an additional control continues the validation-selected
KDE run and the best non-KDE recipe for 2,400 more updates at learning rate
`2e-4`, restarting Adam at the selected coefficients. The alternative recipe is chosen by its mean validation score across
seeds. These continuations keep the data, rank, modes, and penalty fixed, and
are selected before independent evaluation. They are separate from the
matched-budget primary comparison.

## Selection and independent evaluation

Two practical policies are recorded before evaluating truth:

1. Select the lowest-scoring initial TT, then use its trained checkpoint.
2. Train all candidates and select the lowest-scoring resulting checkpoint.

The second spends more TT fitting work and is not described as a free
initialization improvement. A single-recipe comparison averages the ranks of
mean validation scores across cases, weighting each case equally.

A screening control, specified before independent-test or truth evaluation,
uses seed zero only. It compares choosing a start immediately, choosing by the
best validation score observed through 400 updates, and training all eight
starts. Screening is a retrospective simulation on the stored trajectories;
it cannot see later scores of discarded candidates. Its maximum TT update
cost is `8*400 + 800 = 4000`, versus 1200 for a single chosen start and 9600
for training all eight. Constructing all eight starts is an additional cost.
This control does not alter the primary fixed-recipe recommendation.

For the earlier request to transfer swirl improvements to cryo-EM, a separate
recipe is chosen by the same mean-validation-rank rule on the two swirl
cases. This choice is also frozen before truth evaluation. It is a
curved-density transfer choice, distinct from the general-purpose ranking
over all eight cases.

The construction-cost figure reports observed wall times, including pilot
fitting and TT conversion. Shared compilation caches and some overlapping
execution on the single GPU mean these are approximate costs, not isolated
cold-start timing benchmarks. Common data/noise-kernel preparation is excluded.

After those decisions are frozen, 8,192 fresh noisy observations measure
prediction. GL uncertainty calculations account for its 200 independent
sampling chains. The known signal enters only the subsequent recovery
evaluation and marginal plots.

Full-dimensional L2, squared Hellinger distance, and total variation use a
defensive importance proposal with half its mass on the known signal and
half on the fitted TT. Each stratum has 16,384 draws. TT samples come from
conditional inverse CDFs of the actual squared polynomial amplitude;
Chebyshev antiderivatives supply these CDFs, with 36 bisection iterations.
This includes spurious fitted mass that a true-curve-only diagnostic could
miss. Evaluation reports Monte Carlo standard errors and independently
checks the known squared norm of the true density. GL's true-sample
contribution uses chain-cluster standard errors.

No recovery error chooses a fit, seed, pilot, bandwidth, rank, or checkpoint.
Independent test observations are also excluded from all selection.

An independent prescribed-noise check compares each saved TT kernel with
the analytic noise mixture at 1,024 draws per case. All sampled values are
positive; the largest relative discrepancy is `4.3e-9`. This check uses the
known noise only, and supports interpreting the much larger pilot-to-TT
likelihood losses as initialization/conversion effects.

Independent evaluation also exposed an extreme-tail issue: the polynomial
noise approximation produced one nonfinite prediction for a Gaussian start
on a fresh banana observation, although the fitted TT's predictions were
finite. For diagonal Gaussian-mixture noise, evaluation now checks every
prediction below log density -25, and any nonfinite value, using direct
positive quadrature with the prescribed analytic noise. Doubling its order
must agree within `1e-7` in log density. This resolves tiny values below the
absolute polynomial-kernel error without clipping densities. The same rule
is applied to every initialization and fitted model; counts and corrections
are recorded. All model and selection hashes remain fixed. This numerical
measurement correction uses the supplied noise, not the clean signal.

## API and reproduction

`ftde.initialize_tt_from_pilot(...)` exposes the eight recipes. It accepts
training/validation observations, the signal box, and a prescribed zero-mean
Gaussian-mixture noise specification. The Gaussian moment starts use only
the noise covariance and also apply to independent zero-mean noise with that
finite covariance. The EM starts assume the Gaussian-mixture noise family;
the underlying TT convolution remains more general.

`ftde.select_tt_initialization(candidates, validation, noise_tt=...,
noise_scales=...)` selects among converted TT amplitudes using their actual
noisy likelihood and common smoothness penalty. It leaves the original KDE
initializer available and does not change any historical gallery defaults.
This helper chooses by the initial score; it does not predict which start
will optimize best. The benchmark includes counterexamples to that shortcut.
The general-purpose recommendation instead uses aggregate scores *after*
the matched training budget.

For the general-purpose start selected in this study:

```python
from ftde import initialize_tt_from_pilot

start, diagnostics, pilot = initialize_tt_from_pilot(
    train, valid, domain, noise_covariances, noise_weights,
    method="gaussian_guided", rank=32, modes=49, seed=0,
)
# Pass `start` to the existing squared-TT optimizer and noise convolution.
```

This start does not require a KDE bandwidth. It estimates a coarse signal
Gaussian from the observed moments and the prescribed noise covariance;
the subsequent free TT optimization can learn non-Gaussian structure.

Using the `chebdGE` environment on one GPU allocation:

```bash
export JAX_ENABLE_X64=1
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
srun --ntasks=1 python scripts/run_initialization_study.py --finish
# Resume only the backfill/long-control/evaluation stage if needed:
srun --ntasks=1 python scripts/finish_initialization_study.py
# Regenerate figures from completed evaluations:
srun --ntasks=1 python scripts/plot_initialization_study.py
```

The finisher freezes model hashes and data-only choices before evaluating
truth. Under Slurm it checks the remaining allocation time before starting
the longer controls; a stopped run can resume on the next allocation.
The runner creates a missing protocol from
`scripts/initialization_protocol.py`; use `--root artifacts/initialization_repeat`
for a fresh set of fits. The two 10D cases require the existing GL/swirl
gallery data, domain, and noise-kernel artifacts described above. A fresh
run starts with all eight recipes and the 5,000-center KDE cap; the protocol
also records how the original study acquired its conversion controls.

`tests/test_pilot_initialization.py` checks noise moment subtraction, physical
normalization, density-preserving gauge changes, activation of missing ranks,
and validation selection. `tests/test_tt_sampling.py` checks inverse CDFs
and correlated moments against independent analytic/quadrature calculations.
`tests/test_sampled_tt_cross.py` checks reconstruction of a low-rank
multivariate polynomial on independent points and one-dimensional interpolation.

## Results

All 192 primary attempts are complete: 189 produced fitted TTs. The three
failures are the ordinary full-covariance pilot conversion on 10D swirl;
each reported a zero or invalid physical amplitude norm. The corresponding
three sample-guided conversions all succeeded. These failures are included
in the ranking, rather than replaced by favorable retries.

The frozen general-purpose choice is **the noise-corrected moment Gaussian
with sample-guided TT conversion** (`method="gaussian_guided"`). Its mean
validation rank is 3.125, versus 3.250 for ordinary Gaussian conversion and
4.500 for KDE. The small rank difference between the Gaussian variants is
not evidence of a substantial universal accuracy advantage. Leaving out
one case at a time selects one of these two Gaussian recipes in all eight
checks; equal weighting of the six distribution families also selects the
guided Gaussian. This supports the Gaussian *family* more strongly than one
specific conversion implementation.

Across cases, the span of mean validation NLLs between recipes is larger
than the median within-recipe range over three seeds. For 10D swirl those
quantities are 0.1583 and 0.0082; for GL10, 0.0305 and 0.0026. Choosing the
initialization recipe matters more here than simply changing its seed.
These comparisons exclude failed conversions from the finite NLL ranges.

Construction matters independently of pilot fitting. On 10D swirl the same
moment Gaussian has noisy validation NLL 9.8508, but ordinary conversion
starts at 12.4559–13.3683 and guided conversion at 11.8711–13.1779. Guidance
helps some starts but does not reliably preserve the pilot in this harder
case. A single Gaussian is also a poor description of the swirl itself.
The frozen swirl-specific choice is instead the tangent-mixture pilot,
converted to an ordinary Chebyshev TT before any TT training.

Independent full-dimensional relative L2 errors are below. Entries are
medians over the three initialization seeds; smaller is better. Every
checkpoint was chosen before seeing these errors.

| Case | KDE | Moment Gaussian | Guided Gaussian | Tangent pilot |
|---|---:|---:|---:|---:|
| Correlated Gaussian, 4D | 0.414 | 0.012 | 0.012 | 0.066 |
| Multimodal, 4D | 0.394 | 0.374 | 0.384 | 0.121 |
| Banana, 4D | 0.284 | 0.180 | 0.180 | 0.407 |
| Ring, 4D | 0.318 | 0.282 | 0.353 | 0.483 |
| GL, 6D | 0.554 | 0.571 | 0.419 | 0.774 |
| Swirl, 4D | 0.330 | 0.353 | 0.355 | 0.284 |
| GL, 10D | 0.471 | 0.604 | 0.607 | 0.619 |
| Swirl, 10D | 0.696 | 0.783 | 0.775 | 0.368 |

The tangent recipe reduces median swirl error by approximately 14% in 4D
and 47% in 10D. Its validation-selected seed also improves fresh noisy-test
log likelihood over KDE by `0.00949 ± 0.00147` and `0.07469 ± 0.00363`,
respectively (paired standard errors, per observation). All six tangent
swirl fits retain **step zero**: these are improvements from constructing
the starting TT, not evidence that the TT optimizer successfully refines
that start.

Longer optimization reduces the selected 10D swirl KDE error from 0.690 to
0.655, while the tangent continuation retains 0.368. GL10's two continuations
retain their source checkpoints (KDE 0.471, full-covariance pilot 0.494).
Thus the tested additional budget does not erase the initialization gap;
this does not rule out improvements with a different optimizer or schedule.

The 400-step screening policy improves the initial-score choice on GL6 and
GL10, but still keeps inferior recovery on banana and ring. Training all
eight candidates is also not an oracle: in the seed-zero multimodal control,
it chooses a Gaussian-derived fit with worse recovery than the tangent
start. More candidate fitting improves the search over noisy validation
scores; it cannot guarantee the best clean-density reconstruction.

The true-ridge diagnostic, used only after fitting, finds 14–19 amplitude
sign crossings for ordinary Gaussian starts after fitting 10D swirl, 10–12
for guided Gaussian, and zero for the tangent starts. However, the KDE runs
have only 0–6 crossings and still recover poorly. Amplitude zeros contribute
to the diagnosis but do not explain every recovery error.

Figure summaries:

- [Recovery errors and seed ranges](../artifacts/gallery_archive/before_noise_free_v2/convergence/initialization_recovery.png)
- [Training trajectories](../artifacts/gallery_archive/before_noise_free_v2/convergence/initialization_training.png)
- [Independent noisy-test comparisons](../artifacts/gallery_archive/before_noise_free_v2/convergence/initialization_test_likelihood.png)
- [Construction cost and conversion losses](../artifacts/gallery_archive/before_noise_free_v2/convergence/initialization_construction.png)
- [Longer-training controls](../artifacts/gallery_archive/before_noise_free_v2/convergence/initialization_long_controls.png)
- [Initialization screening](../artifacts/gallery_archive/before_noise_free_v2/convergence/initialization_screening.png)
- Swirl marginals: [4D spiral](../artifacts/gallery_archive/before_noise_free_v2/swirl/initialization_swirl4_marg_01.png),
  [4D pretzel](../artifacts/gallery_archive/before_noise_free_v2/swirl/initialization_swirl4_marg_23.png),
  [10D spiral](../artifacts/gallery_archive/before_noise_free_v2/swirl/initialization_swirl10_marg_01.png),
  [10D pretzel](../artifacts/gallery_archive/before_noise_free_v2/swirl/initialization_swirl10_marg_23.png).

These matched-budget experiments do not replace the earlier gallery fits:
the smaller cases use new observations and noise settings, and 1,200 updates
are a deliberately limited common budget. Some historical gallery fits use
more data, longer optimization, and separately selected regularization.

The Gaussian recommendation is a practical default chosen by observable
validation scores, not a claim of uniformly smallest clean-density error.
For example, guided Gaussian improves the Gaussian, banana and GL6 cases,
but KDE recovers GL10 better under this protocol. Ordinary and guided
Gaussian conversion also differ on the ring. Small noisy-likelihood
differences can coexist with large recovery differences in this inverse
problem; validation cannot reliably identify every latent artifact.

The tangent pilot allows one elongated local direction and a shared
transverse variance. That is a useful starting assumption for the swirl's
single tube, but more restrictive for two independent curved pairs or a
ring accompanied by additional broad coordinates. These restrictions apply
to the pilot, not to the subsequent TT parameterization. Their influence
can nevertheless persist under finite-budget optimization.

## Cryo-EM transfer

The frozen swirl-specific recipe (`tangent`) was transferred to the existing
4D, 4D-path, 10D and 20D cases. These keep their original data, prescribed
noise, domains, 49 modes, and maximum ranks 32, 32, 64 and 48. Each new start
received 1,200 TT updates with the existing cryo-EM smoothing strength
`3e-4`, Sobolev preconditioning and learning rate `2e-4`.

After the 10D candidate was still improving at that limit, both
higher-dimensional cases received a further 2,400 updates from the selected
primary checkpoint, restarting Adam at the same learning rate. This
follow-up is based only on cryo-EM development likelihoods. The original
1,200-step fits and comparisons remain saved separately. The synthetic
benchmark and its frozen recommendations are unchanged by this follow-up.

Candidates are compared with the previously retained fits by paired noisy
validation likelihood, choosing the smoother amplitude among fits within
one standard error. These are practical transfer comparisons against
historical fits with different training histories, not matched-budget
initialization experiments. The development validation observations have
been reused, so their scores are not independent-test results. No clean
cryo-EM density or state labels enter these runs.

All four transfers and both longer continuations are complete. Lower noisy
validation NLL is better:

| Case | Previous fit | New start, 1,200 updates | New start, +2,400 updates | Retained fit |
|---|---:|---:|---:|---|
| 4D | 3.531477 | 3.534704 | — | Previous |
| 4D path | 3.539250 | 3.539623 | — | New, by smoothness within one SE |
| 10D | 9.857318 | 9.874981 | 9.864500 | Previous |
| 20D | 24.656633 | 24.677509 | 24.652809 | New continuation |

The 4D-path candidate has comparable likelihood: its change is
`-0.000373 ± 0.001187` in mean log likelihood (new minus previous, paired SE).
Its amplitude H² roughness falls from 37.482 to 33.915, so the one-SE rule
selects it. This is a smoothness improvement, not a likelihood improvement.
The 20D gain is `0.003824 ± 0.002794`, approximately 1.37 paired standard
errors. That is enough for the specified selection rule, but is modest
evidence on a reused development split; its amplitude roughness increases
from 152.610 to 158.621. The 4D and 10D comparisons retain the previous fits.
These results do not establish better recovery of the unknown clean density.

Comparison figures: [4D](../artifacts/gallery_archive/before_noise_free_v2/cryoem/initialization_study_dim4.png),
[4D path](../artifacts/gallery_archive/before_noise_free_v2/cryoem/initialization_study_path_dim4.png),
[10D](../artifacts/gallery_archive/before_noise_free_v2/cryoem/initialization_study_dim10.png), and
[20D](../artifacts/gallery_archive/before_noise_free_v2/cryoem/initialization_study_dim20.png).

For example, reproduce the 10D continuation with:

```bash
srun --ntasks=1 python scripts/transfer_initialization_cryoem.py --case dim10 --extra-steps 2400
```

The `dim4`, `path_dim4` and `dim20` cases use the same script. Omitting
`--extra-steps` resumes an existing continuation if one has already been
saved. New candidate/selected checkpoints stay under
`artifacts/initialization_study/cryoem_<case>`; comparison figures stay under
`figs/cryoem`. Previous gallery models and figures are preserved.

## Verification

The 18 focused initialization, TT-cross, sampling, local-mixture and
TT-convolution tests pass. Seven additional Gaussian-convolution tests pass
for the independent tail calculation. Source lint checks also pass.

All 193 synthetic benchmark evaluations, including its four longer
continuations, verify their frozen model hashes.
Doubling the prescribed-TT convolution quadrature at eight fresh test
observations per model changes log densities by at most `1.1e-12`.
The defensive importance estimates of the known signal squared norm are
within 2.1 Monte Carlo standard errors of their independently computed
values. The largest fitted-sample mean deviation is 3.39 standard errors
relative to exact TT moments. These checks support the reported numerical
accuracy; they do not remove uncertainty from the finite training data.

`scripts/audit_initialization_study.py` checks the saved data and model
hashes, verifies preservation of all 101 earlier PNGs, and records source,
dependency, test and figure hashes under `artifacts/initialization_study`.
Only figures and the existing figure index are written under `figs`.
