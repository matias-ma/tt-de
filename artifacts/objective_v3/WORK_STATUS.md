# COMPLETE — objective and capacity study

Completed 2026-09-28T01:11:31.075304+00:00.

- All61fits:4000Adamupdates each, simple near-uniform initialization, one squared ChebyshevTT. No signal pilots.
- Sixcases: GL6free, swirl4free, GL6pairedGaussian, GL6broadmixture, swirl4/10correlatedmixture.
- Separate NLL/MMD selections frozen before truth; all366candidatefilehashes and source data hashes verified.
- 46checkpoints evaluated; no step-zero retained checkpoints; 14tests passed; Ruff andgitdiffcheckpassed.
- Sixindependent bounded-box swirl cross integrals agree within1.10e-8 relative totruthsquarednorm. Fourier andnoiseCF refinements passed. Eleven observed train/validation controls confirmGLoverfitcomponentandswirlunderfittingatlowrank.
- 34PNGs + README infigs; all PNGs readable and report/gallery local links resolve. Gallerystructure preserved. Older cryo/noise-free10D andablationfigures explicitlyidentified as latestavailablev2, notrerunhere.
- Numerics~85.1MiB; figures~13.8MiB. Node /tmp/ftde_v3_cuda andmplcache removed; jax/cache dirs neverpersisted. Ownallocation14585312 released andconfirmed absentfromsqueue. Otheruserjobs14452283and14590922untouched.
- No outstanding processes or required work for this study.

Main result: capacity tuning matters more than the modest hybrid-objective gains. GL rank10/25 L2.07710 vs rank32/25.12157; selectedrank5/25lambda.01 reaches.05324. Noise-free swirl hybrid32/49 reaches.09300 vsNLL.10547. Noisy4swirl.29476 vsmatchedNLL.29839 (compactNLL.39322). Noisy10swirl.70992 vsmatchedNLL.71204 (compactNLL.78907), stillpoorfulljointrecovery. GLmixture NLL-selection.79599 vsMMD-selection.30901, butneitherselectioncriterionwinsuniversally.

See docs/objective_study.md, figs/README.md, audit.json, tests_complete.log, lint.log.
