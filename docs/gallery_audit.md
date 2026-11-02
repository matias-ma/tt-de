# Audit of GL accuracy and cryo-EM appearance

Audit date: 2026-09-25. No fitted models or published images changed.

## Frozen 14D paired-noise GL fit

The saved full-dimensional, unsquared Lebesgue relative L2 is
0.2306560401. It is the error of the latent density, not its convolution
or a two-coordinate marginal. The integral calculation includes the
estimated density's full declared box, taking it to be zero outside;
the GL normalizer uses [-3,3] in each coordinate, where the quartic
potential makes omitted tails negligible. Refining the saved cross and
fourth-power quadrature from orders 65 to 130 changes the integrals by
1.15e-9 relative to the truth's squared norm.

An additional post-fit check sampled 100,000 independent configurations
from the GL transfer distribution on a 601-point grid on [-3,3], seed
938125. This avoids the training MCMC sampler. NumPy evaluated the frozen
TT cores directly; Gauss-Legendre contractions normalized its square.
Importance sampling estimated the squared error as
`mean((q(X)-p(X))**2/p(X))/integral(p**2)`. The resulting relative L2 was
0.226510, with delta-method Monte Carlo standard error 0.002830.
This corroborates the deterministic result; finite importance sampling can
miss rare tail contributions and is not a replacement for that integral.
Results are saved in `artifacts/gallery_audit/gl_paired_dim14.json`.
All three tests in `tests/test_gl_l2.py` pass, including independent 2D
tensor cubature and exact-zero boundary handling.

The noise scale 1.5 multiplies the base standard deviation
`sqrt(.7*.3**2+.3*.9**2)`: actual coordinate standard deviation is 0.829759.
Transfer integration gives signal standard deviations 0.842416–0.859215.
Noise-to-signal coordinate variance ratios are 0.933–0.970. Correlated
noise can still be worse in particular directions; a coordinate variance
comparison does not describe all recoverable features.

## Information available to the estimator

The gallery fit receives noisy training/validation observations and the
prescribed noise TT. Its initialization uses the noisy-data KDE. The
pilot compares exponential-polynomial marginal degrees 2, 4, 6, 8,
shared or separate, by noisy validation likelihood. In the 14D paired
case it selected **separate degree-4 marginals**. The Fisher penalty
shrinks weakly observed amplitude directions toward that learned reference.
It does not simply penalize high polynomial degrees. GL truth is used to
generate benchmark observations and for subsequent validation, not as the
automated fitting/selection objective.

This is nevertheless a substantial structural preference. GL's quartic
site potential, smooth nearest-neighbor interactions and natural coordinate
ordering are favorable to the chosen pilot and TT representation. The
workflow has also been developed repeatedly using GL error diagnostics.
Noisy-only selection within a run does not remove development-time
adaptation to the benchmark. Published accuracy currently lacks independent
training-seed replications. The 100,000 training observations come from
200 thinned MCMC chains, not 100,000 certified independent draws.

The generator also restarts the same numeric RNG seed separately for signal
sampling and additive noise. These streams should be explicitly separated
in a fresh benchmark; their reuse is a simulation-design issue, not evidence
that clean density values entered the optimizer. Preserve existing artifacts
when changing the simulation protocol.

Older module comments describe a squared error evaluated at sampled points
and claim direct comparability with another implementation. Those comments
do not describe the current gallery's integrated metric. Comparing against
another estimator requires matching its actual error implementation,
sampling, domain, normalization and tuning protocol. Increasing dimension
also changes the GL interaction coefficient because h=1/(d+1).

## Cryo-EM beads

The latent estimate remains a normalized squared Chebyshev TT, not a
Gaussian mixture. Gaussian KDE is used for initialization; the four-component
Gaussian mixture specifies the **noise**. The displayed latent marginals
are polynomial marginals obtained by integrating the squared TT, without
Gaussian smoothing of the output. Rank is 20 and amplitude degree is 48.

The 10D marginal (0,1) visibly breaks into bright patches along a curved
ridge; the 4D version is more continuous. Smooth polynomials can produce
oscillations and near-zero valleys, and squaring the amplitude preserves
those valleys. A blurred likelihood may distinguish such alternatives
poorly. A thin curved ridge may require substantial separation rank and
is not made easier solely by adding polynomial modes. These are plausible
contributors, not a completed causal decomposition of the remaining beads.

The implemented curvature penalty also penalizes sharp transverse structure;
it does not independently control variation along a ridge. The noise model
is an approximation to heterogeneous per-observation uncertainties and
omits within-component correlations and possible state dependence.
Residual model error can be absorbed into the latent density. Improvements
in noisy validation likelihood therefore do not validate the individual
latent patches.

Useful next experiments are independent observation/bootstrap and optimizer
replications, capacity/stability checks, and controlled noise-model checks.
Any geometry-aware smoothing should be judged by these diagnostics, not
chosen just to make the curve look continuous. The supplied noise function
remains the estimator input; no clean signal or signal covariance is needed.
