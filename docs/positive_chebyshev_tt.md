# Positive polynomial cores, retaining the Chebyshev functional TT

This implements the positive-core idea from [the positivity review](positivity_options.md),
adapted to the user's requirement to retain Chebyshev polynomials. It does **not**
implement the spline basis proposed in that review. Both storage and convolution
remain Chebyshev functional TTs.

**Outcome:** the positivity and normalization machinery passes its numerical
checks, but the tested fits recover noisy swirl **worse** than the original
squared TT in both 4D and 10D. This remains an optional experimental estimator;
the existing defaults and gallery fits are retained. It was not promoted to
cryo-EM because the requested prerequisite—better swirl recovery—was not met.

The new density is the TT contraction itself:

\[
p(x)=G_1(x_1)\cdots G_d(x_d),\qquad
G_k(x_k)_{ab}=\sum_j c_{k,abj}T_j(t_k(x_k)).
\]

Every scalar entry of every core is a positive polynomial throughout its interval.
Outside the existing latent box the density is zero. There is no final square,
clipping, softplus, or exponential, and no fitted Gaussian-mixture signal pilot.
The prescribed noise TT, including its correlations, is unchanged.

## Enforcing positivity without changing the model basis

Chebyshev coefficients can have either sign. Constraining the coefficients
themselves to be positive would not ensure positive polynomials, since the
Chebyshev basis functions change sign.

Instead, a fixed linear matrix maps each polynomial's Chebyshev coefficients
to its Bernstein coefficients on 16 subintervals. Bernstein basis functions
are nonnegative and sum to one. Therefore positive coefficients in this
*certificate* imply that the original Chebyshev polynomial is positive at
every point, including between quadrature nodes and at endpoints.

These are sufficient linear inequalities; they are not a claim that every
positive polynomial passes at a fixed subdivision count. Finer subdivision
reduces conservatism. The matrices are built with a local recurrence, avoiding
unstable conversion through high-degree global power coefficients. The model
is never evaluated or convolved in a Bernstein or spline basis.

Floating-point arithmetic still applies. Updates preserve a small interior
margin (`1e-8` in dimensionless core-value coordinates), and each saved model
is checked again after conversion to its stored Chebyshev coefficients.

## Exact normalization and optimization

Each incoming core row satisfies

\[
\sum_b\int G_k(x)_{ab}\,dx=1.
\]

These are linear equalities using the exact Chebyshev integration moments.
The integrated cores are stochastic matrices with boundary ranks one, so the
whole density integrates to one. This prevents scale escape in the barrier
objective. Such normalization is available for any strictly positive-core TT
by a positive suffix-integral gauge transformation.

With other cores fixed, every noisy likelihood value is linear in the active
core. An alternating block minimizes noisy negative log likelihood, a log
barrier on the polynomial certificates, and an optional fixed quadratic core
curvature penalty. The individual block problem is convex; the entire TT
fitting problem is still nonconvex.

The implementation uses matrix-free Newton-CG with small QR factors to scale
the core directions. Equality constraints are projected out, a fraction-to-
boundary step preserves positivity, and Armijo backtracking verifies objective
decrease. Nodal values at Gauss points are **optimizer coordinates only**:
the invertible map back to Chebyshev coefficients leaves the represented
polynomial and its degree unchanged. The curvature penalty is on the actual
core polynomial; the old amplitude-relative Sobolev quotient is not reused.
Specifically, it averages the squared physical second derivative of
`u = right_rank * interval_width * G` over edges and the uniform coordinate
measure. This is a fixed rescaling within a fit; its strength should not be
interpreted as identical to the old amplitude penalty or automatically
equivalent across different ranks.

`PositiveTTForward` reuses `LinearTTConvolution` and the existing prepared
one-coordinate Chebyshev/noise moments. It works beyond 4D without a full
tensor-product integration grid. Dense block design matrices currently make
memory grow with batch size times adjacent signal ranks times mode count;
the implementation does not yet stream Newton products over the dataset.

The comparison baseline is the original squared TT in each frozen
`artifacts/swirl/swirl_mixture_dim*/selected.npz`, not the later
mixture-initialized squared-TT alternative. That alternative remains available
and is documented separately in `swirl_improvement.md`.

## Initialization and model handling

K-means on observed training points supplies starting locations. Broad positive
Chebyshev core polynomials and a small transition floor give an initial TT.
This is only a data-based initialization: no Gaussian signal mixture is fitted
by EM, no known latent density/curve enters it, and all core transition
polynomials subsequently train independently against the noisy likelihood.
This start can still bias local optimization; it is not a global convergence
guarantee.

`PositiveChebyshevTT` is a separate wrapper with its own serialization tag.
Call its `density`, `log_density`, and `pair_grid` methods. Do **not** pass its
underlying `.tt` to the old `density_tt` helper: that helper intentionally
squares an amplitude. Generic SVD TT rounding also does not preserve the
new model's positivity or stochastic-row constraints.

Example fit (inside the existing allocation/environment):

```bash
python scripts/fit_positive_tt.py \
  --base artifacts/swirl/swirl_mixture_dim4 \
  --output artifacts/positive_tt/example \
  --rank 32 --modes 49 --sweeps 100 --batch-size 8192 \
  --inner-steps 6 --barrier .001 --barrier-final .0001 --strength 1e-8
```

`scripts/evaluate_positive_tt.py` first freezes selection using noisy validation
likelihood, then computes post-fit truth diagnostics and writes comparison
figures to the existing `figs/swirl` or `figs/cryoem` folder. All model files,
logs, and numerical records go under `artifacts/positive_tt`.

## Scope of the validation

The numerical tests cover polynomial certificate identities through 97 modes,
normalization and support, serialization, Gaussian and correlated-polynomial
noise convolution against independent integration, nonzero likelihood gradients,
single-core linearity, and feasibility/objective decrease of a constrained step.

For fitted swirl models, full-dimensional L2 uses the exact TT Gram contraction
for the fitted density's squared norm and one-dimensional quadratures for its
cross integral with the known target. Refinement checks the cross integral and
the prescribed-noise convolution. The independent 8,192-observation test set
is generated only for post-selection evaluation.

Positivity alone does not guarantee improved recovery. Positive-core TTs are a
more restricted model class than arbitrary signed-core TTs, and their useful
ranks need not match those of a squared-amplitude TT. The completed benchmark
outcomes are recorded below.

Mode counts also need care: squaring a 49-mode amplitude gives a density of
polynomial degree up to 96 in each coordinate. A direct 49-mode density has
degree only 48. The resolution ladder therefore includes 97 direct modes
to match the old density's maximum polynomial degree. Equal degree still
does not make positive and squared-TT ranks equivalent.

The degree controls use the same parent checkpoint, data order, regularization,
and number of updates. A tiny move into the strict interior is applied at
initialization; the post-fit capacity audit records any resulting difference
between their initial Chebyshev coefficients. Thus these controls also separate
increased degree from simply doing more optimization.

## Completed swirl results

The observation-only search covered ranks 8, 16, and 32 in 4D, rank 32 in
10D, 33/49/65/97 modes, broad and narrower initializations, and matched-budget
degree continuations. The final comparisons use the completed, stabilized
Newton implementation. Early numerical debugging runs are excluded. Both
dimensions select rank 32 with 97 modes by noisy validation likelihood.

| Case | Original relative L2 | Positive-core relative L2 | Original independent-test NLL | Positive-core test NLL |
| --- | ---: | ---: | ---: | ---: |
| Noisy 4D swirl | 0.216188 | 0.484651 | 3.909332 | 3.917886 |
| Noisy 10D swirl | 0.487772 | 0.806761 | 6.275878 | 6.313321 |

Lower is better in both metrics. The new validation NLLs are 3.870608 and
6.320802, compared with 3.861538 and 6.284202 for the original fits. On the
independent 8,192-row test sets, the new model's log-likelihood changes are
`-0.008554 ± 0.001401` in 4D and `-0.037443 ± 0.003255` in 10D (± one paired
standard error). The test set and clean density were used only after fitting
and checkpoint selection.

The controlled continuations show modest resolution gains, rather than a cure:

| Case | Degree continuation | Noisy validation gain over same-budget control | Lower-degree L2 | Higher-degree L2 |
| --- | --- | ---: | ---: | ---: |
| 4D | 49 → 65 modes | 0.001968 | 0.514529 | 0.517273 |
| 4D | 65 → 97 modes | 0.000369 | 0.488168 | 0.484651 |
| 10D | 49 → 65 modes | 0.006968 | 0.895167 | 0.889684 |
| 10D | 65 → 97 modes | 0.000973 | 0.807434 | 0.806761 |

Each row compares runs from a common parent checkpoint, with the same data
order and update budget. The two stages have different parent checkpoints.
Their interior initialization moves differ by at most `3.2e-6` in a core
coefficient; the full differences are recorded in the capacity audits.

The images still show missing mass and beads despite the absence of a final
square. Eliminating squaring is therefore **not sufficient** to remove these
artifacts. The 10D fit is much less concentrated as a joint density:
`integral q² = 3.782`, versus `28.834` for truth and `17.721` for the original
fit. Reasonable-looking pair marginals do not establish accurate recovery of
the full thin tube.

These experiments do not isolate whether the remaining limitation is mainly
the nonnegative-core rank requirement, the conservative polynomial positivity
cone, initialization, or incomplete nonconvex optimization. The final direct
rank is 32, whereas the old rank-64 amplitude can represent a density TT of
rank up to 4096. Matching polynomial degree is **not** matching those model
classes. It would be incorrect to claim that positivity alone fixes the
original artifacts, or that these results rule out all positive-core methods.

Both selected models integrate to one to within `1e-15`. Their smallest
positive core certificate exceeds `6.9e-11`, and core-row mass errors are below
`3.6e-15`. Doubling convolution quadrature changes checked log densities by
less than `5.6e-13`; refining the latent cross integral changes it by less than
`3.9e-14` relative to the true squared norm. These discrepancies are far too
small to explain the recovery errors.

Fourteen focused tests pass:

```bash
python -m pytest tests/test_positive_chebyshev.py \
  tests/test_linear_tt_convolution.py tests/test_tt_convolution.py -q
```

The final models, frozen selection, numerical audits, and capacity comparisons
are in `artifacts/positive_tt/swirl4_comparison` and `swirl10_comparison`.
Every fit has its command settings, seed, data hash, checkpoints, and history
under `artifacts/positive_tt`; `search_plan.json` records the observation-only
search and controls. The older mixture and clipped options remain available.

Figures are in the existing `figs/swirl` directory:

- `noisy_mixture_positive_dim{4,10}.png`: truth/original/new pair marginals.
- `noisy_mixture_positive_dim{4,10}_training.png`: training and held-out likelihood.
- `noisy_mixture_positive_dim{4,10}_ridges.png`: densities along the true path,
  used solely as a post-fit diagnosis.
- `noisy_mixture_positive_capacity_dim{4,10}.png`: the matched-budget controls.

All pre-existing PNGs are preserved. Numerical artifacts stay outside `figs`.
