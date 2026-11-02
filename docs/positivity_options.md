# Positivity alternatives for the functional TT density estimator

Research date: 2026-09-26. This is a literature review and a proposed
algorithm, not a claim of new swirl or cryo-EM fitting results.

**Implementation update:** the subsequent user request requires retaining the
Chebyshev basis. The [implemented positive-polynomial-core variant](positive_chebyshev_tt.md)
therefore enforces positivity through interval-wide linear certificates on
Chebyshev coefficients. It guarantees positivity but did not improve swirl
recovery. These experiments do not establish that a basis change is needed:
entrywise positive cores are a stronger restriction than a positive final
density, and initialization and optimization remain unresolved. The spline
proposal below remains historical research context, not a demonstrated fix.

The initial research recommendation was a **direct functional TT with positive
spline cores, normalized locally, and trained with alternating constrained
Newton steps**. This removes the final square and hard clipping while
preserving tensor contraction for integration and noise convolution.

We currently square the density *amplitude*, not the optimization objective.
The relevant questions are how to parameterize a positive density and how
to optimize that representation. The visual improvement from clipping is
worth investigating independently of the full-dimensional L2 error. Neither
that observation nor the old amplitude zeros proves that squaring alone
causes every blob.

## Literature that directly applies

* **Novikov, Panov, Oseledets, UAI 2021:** their TTDE paper explicitly identifies
  positive cores combined with nonnegative basis functions as an alternative
  to squaring. This is already part of the density-TT literature.
  [Paper, section 2.2](https://proceedings.mlr.press/v161/novikov21a/novikov21a.pdf).
* **Tang, Dwaraknath, Ying, 2025 preprint:** fit a nonnegative TT to an existing
  discrete TT using log barriers and alternating Newton updates. Their
  experiments favor this over multiplicative updates. The continuous
  functional extension is explicitly future work in section 5; they do not
  demonstrate our continuous deconvolution problem.
  [Paper](https://arxiv.org/html/2507.21519v1).
* **Glasser et al., NeurIPS 2019:** positive TT and squared-amplitude models
  have different expressive power; nonnegative TT corresponds to a hidden
  Markov representation. Rank requirements can differ substantially.
  [Paper](https://proceedings.neurips.cc/paper_files/paper/2019/file/b86e8d03fe992d1b0e19656875ee557c-Paper.pdf).
* **Kooperberg and Stone, 1991:** logspline estimation models a log density
  and exponentiates it. This supplies another established positivity route,
  but it does not solve the multidimensional TT partition-function problem.
  [Authors' copy](https://research.fredhutch.org/content/dam/research/kooperberg/publications/1991logspline.pdf).
* **Tang, Khoo, Ying, 2025:** study initialization and optimization failures
  of MPS probabilistic models, including poor conditioning under changes of
  core scale and traps affecting long-range dependencies. This cautions
  against attributing every fitting failure to the positivity map.
  [Paper, sections IV–V and appendix B](https://arxiv.org/html/2505.06419v1).

## Options and their consequences for this repository

The computational assessments in this table are deductions for our model,
not reported swirl results from the cited papers.

| Construction | Positivity and gradients | Integration/convolution | Assessment |
|---|---|---|---|
| Global `exp(h_TT)` | Strictly positive for finite scores; no sign folding or hard inactive half-space | Exponentiation generally destroys the low-rank product representation | Established idea, but a major change to high-dimensional integration |
| Global `softplus(h_TT)` | Smooth and strictly positive; derivative is sigmoid, which can still become tiny | Same loss of separability as global clipping | Useful 4D control, not a solution to the scaling bottleneck |
| Entrywise softplus **inside each univariate core** | Positive matrix-valued cores; no global sign cancellation; possible saturation remains | Core integrals remain one-dimensional | Closest route to retaining Chebyshev functions; nonlinear per-core optimization |
| Positive B-spline coefficients in every core, with a log barrier | Positive throughout the box with overlapping basis coverage; no clipped negative region | Exact spline normalization and ordinary TT contraction; noise moments computed per coordinate | Preferred: also permits convex individual-core likelihood updates |
| Barrier only at sampled joint points | Keeps the checked points positive | Preserves the raw TT's cheap integral | Does not guarantee positivity between points or in unseen regions |

For softplus, `s(t)=log(1+exp(t))` and `s'(t)=sigmoid(t)>0` at every finite
real t. Numerical underflow and small derivatives in the negative tail are
still possible. The derivative of the *log* link is different:
`d log(s(t))/dt=sigmoid(t)/s(t)`, approaching one in the negative tail.
Neither scalar derivative alone proves good conditioning of a normalized,
convolved TT likelihood. Stable evaluation is essential.

Positive Chebyshev **coefficients** do not imply a positive function because
Chebyshev basis functions change sign. This does not require a basis change
or a positive link: suitable constraints on signed Chebyshev coefficients
can certify positivity of the polynomial, as in the implemented variant.
That certificate is sufficient and conservative; requiring every core entry
to be positive also restricts the model beyond positivity of the final
contracted density. Cubic B-splines supply local,
nonnegative basis functions and C2 continuity at simple interior knots;
extrapolation beyond the model's box must be disabled. See the
[SciPy basis definition and integration API](https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.BSpline.html).

## Proposed continuous model: derivation for this project

This section specifies our adaptation, rather than attributing a continuous
noisy-likelihood algorithm to the discrete NTT paper.

Let `b_kj(x_k)` be nonnegative cubic B-splines normalized so each basis
function integrates to one on coordinate k's interval. Set

\[
G_k(x_k)_{ab}=\sum_j C_{k,abj}b_{kj}(x_k),\qquad
p(x)=G_1(x_1)G_2(x_2)\cdots G_d(x_d),
\]

with boundary ranks `r_0=r_d=1`. Maintain

\[
C_{k,abj}>0,\qquad \sum_{b,j}C_{k,abj}=1
\quad\text{for every incoming state }a.
\]

All matrix entries are positive wherever the spline basis covers the
interval. Thus every summand in the joint density is positive. A zero of
an unconstrained amplitude cannot create a density hole, and there is no
negative region discarded by an output activation. This does not guarantee
that optimization finds the right density or avoids all spurious bumps.

Define `M_k=integral G_k(x_k) dx_k`. Each M_k has row sums one, so

\[
\int p(x)\,dx=M_1 M_2\cdots M_d=1.
\]

There is no high-dimensional quadrature for normalization. This stochastic
normalization is a gauge choice, not an additional restriction on strictly
positive functional TTs in this basis: for any such unnormalized model,
compute positive suffix integrals `z_k=M_k z_(k+1)`, with `z_(d+1)=1`, and
replace `C_k,abj` by `C_k,abj z_(k+1),b / z_k,a`. The resulting product is
the original model divided by its normalizer.

This normalization also prevents a concrete failure of naive barrier use:
adding `-mu sum(log C)` to a scale-invariant normalized likelihood with
unrestricted core magnitudes can reward scaling coefficients upward without
changing the density. Keeping coefficients on positive simplices removes
that scale escape direction.

## Why the existing noise problem remains tractable

If the known noise kernel has functional TT cores U_k, the noisy density is

\[
q(y)=\prod_{k=1}^d
\left[\int G_k(x_k)\otimes U_k(y_k-x_k)\,dx_k\right].
\]

Each bracket is a one-dimensional matrix integral; the product is a TT
contraction. This identity includes correlated TT noise and does not assume
independent coordinates. Because G_k is linear in C_k, the spline/noise
integrals can be prepared before optimization. The existing
`LinearTTConvolution` supplies the contraction pattern, but its Chebyshev
moment construction would need a spline implementation.

The positivity guarantee is for the model and the exact positive noise
kernel. Numerical kernel compression and quadrature still need the same
accuracy checks as before. Generic SVD rounding is not positivity-preserving
and must not be silently applied to the new cores.

## A better-conditioned individual-core optimization problem

Fix every core except C_k and flatten it to a vector c. The noisy density
at observation y_n is a **linear** function `a_n^T c`. The local row-sum
constraints keep the whole model normalized during that update. Therefore
the proposed core objective is

\[
L_k(c)=-\frac1N\sum_n\log(a_n^T c)
      +\lambda R_k(c)-\mu\sum_i\log c_i,
\qquad c>0,\quad A c=\mathbf1.
\]

For a convex quadratic smoothness penalty R_k, this is a convex constrained
problem. Its Hessian is

\[
\nabla^2 L_k(c)=\frac1N\sum_n\frac{a_n a_n^T}{(a_n^T c)^2}
 +\lambda\nabla^2R_k+\mu\operatorname{diag}(c_i^{-2}).
\]

The terms are positive semidefinite, and the barrier is strictly convex for
mu>0. Use an equality-constrained Newton/CG direction and a line search
that stays strictly inside the positive simplex; sweep through the cores
and reduce mu as the inner problems converge. The barrier derivative
`-mu/c` grows near zero instead of creating clipping's inactive region.
For the general interior-point method see
[Boyd and Vandenberghe, chapter 11](https://web.stanford.edu/~boyd/cvxbook/bv_cvxbook.pdf).

An integrated squared derivative of the **density**, with other cores fixed,
is a convex quadratic penalty. Dividing it by a variable squared norm would
generally lose this convexity. The original amplitude-relative H2 penalty
must not be reused without checking this change.

The **whole TT problem remains nonconvex** because all cores interact.
Newton updates do not establish a global convergence guarantee. They do
give a more structured alternative to simultaneous Adam updates and avoid
the specific hard-clipping dead zone.

At useful ranks, explicitly storing the core Hessian would be too expensive.
Implement Newton-CG through Hessian-vector products and left/right TT
environments, projecting directions onto the subspace with zero row sums.
The likelihood contribution to a Hessian-vector product is
`mean_n a_n (a_n^T v)/(a_n^T c)^2`; the full matrix need not be formed.
Line search must also enforce `c + alpha v > 0`; merely adding a barrier to
unconstrained Adam would not ensure feasible steps.

## What changes, and what does not

The signal would be learned directly by the TT's noisy likelihood. No fitted
Gaussian-mixture pilot or true curve is used. Positive TT models do admit
a latent-state sum/product interpretation; calling them entirely unrelated
to all mixture representations would be inaccurate. This proposal is not
the previously fitted Gaussian-mixture estimator converted into a TT.

The model class does change: positive cores are more restrictive than signed
cores at the same rank. Also, a rank-r amplitude can square into a density
TT with ranks as high as r squared. Comparing direct and squared models only
at equal numeric rank would not be a sufficient capacity comparison. We
should compare validation, parameter count, runtime, and a rank ladder.

The minimal-change control would retain univariate Chebyshev polynomials
H_k but use `G_k(x)=softplus(H_k(x))` entrywise. This keeps separability and
can use one-dimensional quadrature, unlike `softplus(H_TT(x))`. It has a
less convenient optimizer and cannot inherit the spline-core convexity
argument, but would help separate the effects of the basis change and the
positivity change.

## Concrete next experiment

1. Implement the normalized positive spline cores and known-noise moments.
   Verify positivity, normalization, gradients, and convolution independently.
2. Fit the existing swirl observations directly, with data-only
   initialization, a rank/resolution ladder, and several starts. Select on
   held-out noisy likelihood, never known clean density.
3. Compare against the frozen squared and clipped fits. Report full-density
   error and ridge continuity separately after selection, so a smoother
   picture is not confused with a better density estimate.
4. Check 10D integration/memory feasibility early. The proposed contractions
   avoid the 4D clipped method's full tensor grid, but practical cost still
   depends on the required ranks and noise representation.
5. Run cryo-EM only if swirl provides a convincing improvement.

The companion `scripts/positivity_research.py` checks the proposed
normalization, single-core linearity, and convexity algebra on a small
constructed tensor. It is **not** a density-recovery experiment. Its
illustrative map/gradient figure is saved under `figs/swirl`; numerical
notes stay outside `figs`.

The algebra check returned integral 1.0, a maximum discrepancy of
3.47e-18 between direct convolution and its single-core linear form, and
a positive minimum Hessian eigenvalue of 0.1592 for the constructed case.
These are implementation-independent feasibility checks for the proposed
formulas, not evidence that the estimator will improve swirl.
