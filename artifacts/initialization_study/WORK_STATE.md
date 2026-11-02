# Active task state

User asks to keep best positivity choice and test various initializations across a broad range of distributions, finding a general-purpose start; preserve Chebyshev functional TTs and figs structure. We chose original normalized squared Chebyshev TT. No agents permitted. No skill needed.

## Running resources
- Allocation 14492351, node della-l07g2, 1 A100 80GB GPU, 4 CPUs, 32GB RAM, partition cryoem.
- Started 2026-09-26 14:54:12 local; expires 18:54:12 (4 hours). In-place extension to 6h was denied by Slurm ACL. If necessary, finish on a new single-GPU allocation after releasing this one; do not request user permission for that.
- Main srun session 35320 runs scripts/run_initialization_study.py --finish, logs/runner.log. At ~58 min elapsed it is finishing ring4; GL6, swirl4, GL10, swirl10 follow.
- Earlier swirl10 random_s0 timing run completed and will be reused: 1200 steps, ~236 sec including first compile; 0.16 sec/update after compile.
- User's unrelated job 14452283 md-committor on della-l07g3 must NEVER be touched.
- Use chebdGE: /home/ma4473/.conda/envs/chebdGE/bin/python. Env JAX_ENABLE_X64=1 XLA_PYTHON_CLIENT_MEM_FRACTION=.7 OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4.

## Scientific protocol
- artifacts/initialization_study/plan.json frozen before ANY truth evaluation, with explicit pre-evaluation amendments.
- Eight cases: gaussian4, multimodal4, banana4, ring4, gl6, swirl4, gl10, swirl10.
- Seven starts: random, gaussian (noise-corrected full-covariance moment Gaussian), kde, diagonal EM, tangent EM, full-covariance EM, full_guided (same full EM pilot, data-informed TT cross).
- Three initialization seeds 0/1/2, fixed data and identical minibatch sequence (seed41023), rank32,49 amplitude modes, 1200 Adam updates, lr1e-3 cosine, H2=1e-4, Sobolev preconditioning, batch128.
- 30000 train; validation4096 except old GL10 validation2048. Old10D problems retain old noise; others known two-component diagonal Gaussian noise. New data seeds1701/2701.
- Select checkpoints incl step0 by validation NLL + fixed H2. General fixed recipe selected by equal-case average rank of mean validation scores. Initial-only and full-refit portfolio selectors also recorded. NO known density/curve/labels in initialization/training/selection.
- EM32 components,150 iterations, validation checkpoint every10. Pilots only initialize final ordinary TT.
- All starts unit physical amplitude norm and left-orthogonal cores with jitter to activate missing ranks. This doesn't remove all orthogonal gauge freedom.
- KDE cap was initially2000. Restored historical default5000 BEFORE evaluation; finisher renames earlier runs legacy_kde2000_s* and refits. Gaussian4,multimodal4,banana4 processes loaded old2000; ring onward use5000. Earlier KDE source didn't check rank collapse; current code matches old swirl rank-collapse/bandwidth broadening safeguards. Earlier completed Gaussian/multimodal KDEs were not collapsed (good initial likelihoods).
- Added full_guided BEFORE truth evaluation; early first3 cases need backfill. Main runner in memory calls benchmark defaults, so ring onward includes7 starts; finisher fills missing methods for old cases. Plan records amendment.
- Full_guided uses sampled_tt_cross: Chebyshev fibers, pilot means+128 pilot draws as suffix columns, truncated SVD + pivoted QR row interpolation. Same EM fit as full, no actual signal information. Ring seed0 saved pilots confirmed EXACTLY identical.
- Longer controls planned GL10/swirl10: KDE and best non-KDE recipe chosen by mean validation, each from its validation-selected primary checkpoint, additional2400 updates lr2e-4. scripts/extend_initialization_study.py.

## Automation
- run_initialization_study.py runs prepare/fit all cases then finish_initialization_study.py when --finish.
- Finisher restores KDE5000/backfills full_guided, freezes selection.json, runs longer controls if enough time (estimated controls+1800sec reserve), then evaluates all cases and plots.
- If insufficient remaining allocation, finisher throws BEFORE truth evaluation; resume it on a fresh allocation to finish controls then evaluation. Never skip completed controls merely because computation is slow.
- Initial banana prepare failed noise kernel accuracy (wider box). Fixed initialization_data.py to try noise kernel modes513,1025,2049. Signal modes unchanged49. Data completion sentinel now data.json. Runner restarted and completed banana preparation successfully; old failure remains in logs.
- A premature finisher attempt failed selection due unfinished cases; NO evaluation or truth scoring happened.

## Evaluation and figures (NOT YET RUN at this state)
- evaluate_initialization_study.py requires frozen main selection, loads optional frozen long_controls/selection.json. Verifies hashes.
- Fresh8192 noisy test observations seed83171. GL paired SEs cluster over200 independent MCMC chains.
- Full-dimensional L2/Hellinger/TV from defensive half-true/half-fitted-TT proposal,16384 samples per stratum; exact known p2. True samples seed95247. This is strictly POST-FIT evaluation.
- New tt_sampling.py samples actual normalized squared TT by polynomial conditional CDF (Chebyshev antiderivatives,36 bisections). Also exact marginal mean/variance from polynomial quadrature. Evaluator requires sample means within6SE of exact moments, records max z.
- Refined convolution check doubles quadrature on8 test observations, tolerance1e-7.
- Evaluation records p2 importance ratio and SE; manually audit these before final. It also counts initial/final swirl amplitude sign crossings on true curve only after fit.
- Expected new figures14: four summaries (recovery,training,test_likelihood,construction) in figs/convergence;8 family marginal01 galleries (GL/swirl in existing respective folders, others convergence); extra pair23 for swirl4/10. Seven estimates+truth+observed histogram ->3x3 galleries.
- All existing figs hashed in previous_figures.json. Preserve every old PNG. No numerical files go in figs (PNG + existing README only).

## New code
src/ftde/pilot_initialization.py, sampled_tt_cross.py, initialization_selection.py, tt_sampling.py.
Exports in src/ftde/__init__.py: initialize_tt_from_pilot, select_tt_initialization. Original defaults retained.
Scripts: initialization_data.py, initialization_benchmark.py, run_initialization_study.py, select_initialization_study.py, extend_initialization_study.py, finish_initialization_study.py, evaluate_initialization_study.py, plot_initialization_study.py.
Optional prepared transfer_initialization_cryoem.py (NOT RUN): takes validation-chosen global recipe, keeps previous rank/modes/data/noise, fits1200 steps lr2e-4,H2=3e-4, compares old/new by paired one-SE then H2. Cases dim4,path_dim4,dim10,dim20. Saves new figs/cryoem/initialization_study_*.png, preserves old. Earlier user condition says rerun cryo if swirl improves; assess after frozen evaluation, potentially needs another allocation. Do not claim transfer run unless actually done.

## Tests / docs
- 18 focused tests passed before exact sampler moment helper addition; targeted3 sampling tests including new helper passed after. Final full18 run advisable after source freezes. See logs/final_tests.log and sampling_moments_tests.log.
- Ruff on all new files passes.
- docs/initialization_study.md contains full protocol/API/reproduction; RESULTS SECTION STILL SAYS IN PROGRESS. Must replace with measured conclusions and links.
- figs/README.md and root README not yet updated for this task. Update when results verified.
- Need final manifest (settings/source hashes/tests/figure hashes/resource cleanup), verify old PNG hashes, inspect new PNGs with view_image, cancel own allocation when work complete.
- Repo is heavily dirty from earlier work; preserve it all; no commits requested.

## Preliminary validation-only observations (not clean recovery claims)
- Gaussian4 moment start NLL4.979545, selectedstep0; KDE2000 final~4.995; mixture starts~4.985-4.988 oftenstep0.
- Multimodal4 all recipes end close~4.140-4.143; random/Gaussian/KDE refine while mixtures oftenretain0.
- Banana4 finals~4.314-4.323, no truth yet.
- Ring full/full_guided seed0 identical pilot NLL3.926012826; converted TT3.926376 vs3.925322. Init times12.21s vs3.34s (cache/order limitations). Random/Gaussian ring ends~3.923; no clean-error conclusion yet.
- Swirl10 random_s0 starts15.944346, reaches6.448126 after1200, stillimproving.

Keep user informed regularly. No questions/approval needed. Finish the actual experiments, evaluation and figures before final; don't merely propose more work.

## Later additions / progress
- ~1h13 elapsed: GL6 is running its second seed. No truth or independent test evaluation yet.
- Added screening control before evaluation to plan and select_initialization_study.py: seed-zero-only choice at initial score, best validation score through400updates, or final score. Retrospective simulation, no future scores of discarded arms; fixed-method recommendation unchanged. Plot initialization_screening.png compares their latent recovery after evaluation, with explicit max update costs1200/1200/3600/8400.
- Added plot initialization_long_controls.png for the actual extra2400-update GL10/swirl10 controls. Total expected new primary figures now16 (6 summaries +10 marginal galleries).
- Exact squared-TT marginal means/variances now added to tt_sampling.py and checked by quadrature unit test; evaluator rejects sampler mean discrepancy>6SE. Targeted3 tests pass. Full focused suite had18passing immediately before this helper, rerun full once when final source frozen.
- GL6 seed0: random7.24166, Gaussian7.24397, KDE7.25004 (selected~1000); diagonal/tangent~7.266(selected200) despite much better initial likelihoods. This motivated screening control; no truth-based decisions.
- In-place Slurm time extension was denied. Finisher now reserves1800seconds for evaluation and refuses to evaluate if longer controls cannot fit; resume on a new allocation instead of dropping the controls.

- At allocation elapsed ~1h25: GL6 complete, swirl4 seed0 in progress. First-three-case KDE5000/full_guided backfill now runs concurrently on the SAME GPU in srun session79543/step9? (check squeue steps), JAX fraction.2 and1CPU; main fraction.7. Logs early_backfill.log plus per-case backfill logs. No extra allocation. Finisher will see completed backfills and skip. Figure/docs mark timings approximate because of cache sharing and overlapping execution.
- Final focused suite rerun after sampler moments: 18passed in95.19sec; CPU srun session58860 completed. Ruff new code passed. README now links initialization study/API; docs reproduction updated to runner --finish and includes longer controls. Results still pending.

- At elapsed2h: all first6cases have21complete primary fits each (gaussian4,multimodal4,banana4,ring4,gl6,swirl4), including all KDE5000/full_guided backfills. Session79543 completed successfully. Main session35320 now GL10/kde_s1 construction; GL10seed0all7 done plus random/gaussianseed1. KDE10D construction is expensive: firstGL10KDE416sec, no fallback. GPUactive, notstalled. Remainingallocation~2h, mayneednewallocationforfinisher/cryo. Userjob14452283untouched.
- Pre-truth selection amendment added to plan/selectscript: separate curved_transfer_method by equal-case average validation rank on swirl4/swirl10. Generalmethodstillall8case ranks. Transfer script now usescurved_transfer_method. ThispolicyisfrozenBEFOREanytruthevaluation.
- Data-only preliminarymeanregularizedvalidation winners: Gaussian4=gaussian; multimodal4=full; banana4=gaussian; ring4=gaussian; GL6=gaussian; swirl4=tangent. Gaussianlikelygeneralwinnerbuttwo10Dcasespending.
- Swirl4 selectedNLL(seed0/1/2): random3.641575/3.641234/3.641872(step1200all); gaussian3.626183/3.625697/3.626299(step1200all); KDE3.623555/3.623938/3.623589(step1200all); diagonal3.614616/3.614174/3.615520(step0all); tangent3.608353/3.608994/3.608767(step0all); full3.620473/3.619157/3.618158(steps1200/0/1200); fullguided3.619944/3.611771/3.613023(step0all). NO independenttest orcleanrecoveryyet; cannotclaimlowerlatenterrorfromNLL.
- GL10seed0 score/NLL: random15.47075/15.45140, Gaussian15.46993/15.45585, KDE15.46894/15.46021, diagonal15.49017/15.47563, tangent15.49428/15.48113, full15.46341/15.45455, fullguided15.46871/15.45899. RegularizedscorevsrawNLLcanrankdifferently; reporthonestly.
- Cryotransferbasefilesallpresent inclbandwidth; oldranks32/32/64/48,49modes. canonical_startcanreduceboundaryrank64to49naturally; forwardoldandnewcomputedseparatelysofine. No cryotransferrunsyet.
- Previousfiguremanifestcontains101PNGs. figs currentlyonlyPNG+Markdown; nonewplotsgeneratedyet. DocsResultsstillpending.

- IMPORTANT pre-truth amendment at ~2h20elapsed: added eighth recipe gaussian_guided after observing swirl10 single-Gaussian pilot NLL9.850826 vs ordinaryTTinit13.032294. It uses exactsame noise-corrected Gaussian and existing guided_cross_pilot (no mixture components). Allcases3seeds willbebackedup. Plan/docs/README/plot labelsupdated8methods. Screeningcostnow8*400+800=4000 vs9600trainall. Galleries2x5, conversionplotincludesGaussianpilot losses via independentScipyGaussianconvolution. No truth/testevaluatedyet.
- Main swirl10process loadedOLD7methods; finisher willbackfillGaussian_guided on swirl10 afterprimary. Newbackfill session98658 runs first7cases sequentially on SAMEallocationGPU fraction.2/1CPU; mainfraction.7. Logs gaussian_guided_backfill.log andcase_gaussian_guided.log. Earlier session79543 finished.
- Original7methodGL10allcomplete. Swirl10gaussian_s0 complete:13.032294→6.368784,1200steps199sec. MainnowKDEs0 construction. Swirl10random_s0 reusedearlier236secfit.
- Beforeaddingnewrecipe, seven-case mean ranks: Gaussian2.429,full3.286,fullguided3.571,KDE3.857,tangent4.571,random/diagonal5.143. Newvariantwillchange ranks; recomputeonlyafterall8methodscomplete.
- Ruffpassesafter8thmethodwiring. Existing18tests pass beforethissimplebranchaddition; rerunfocusedtestswhenfinalsourcefrozen. No numericalkernelchanges.

- At elapsed~2h46: allfirst7cases8methodscomplete (168fits). Gaussian_guidedbackfill98658completed. Final18testsre-runafternewbranch:18pass22.51sec. No numericalsourcechangesaftertest. Newsource scripts/initialization_protocol.py copiescurrentplan andrunnerbootstrapsmissingroot/logs/plan; sourceplanequalityverified. Docsrepro nowenvX64andsrun.
- MainSwirl10seed0: Gaussianbest6.368784; KDE6.354258; diagonalretainsstep0NLL6.281482/score6.287975; tangentinitial6.278744,trained~6.289949. full_s0FAILED INITIALIZATION zero/invalidphysicalnorm; full_guided_s0successinitial6.338634, trainingongoing. First constructionfailuremustremainreported. Benchmarkresume nowskipsfailure.jsonaswellasfit.jsonsoitdoesnotsilentlyretryfailedseeds. ActiveSwirl10processhasoldcodebutnoin-processduplicateiterations. Finisheralreadycountsfailurescomplete.
- Eight-recipe7casepreliminarymeanranks: gaussian_guided2.714,gaussian3.0,full4.0,fullguided4.286,KDE4.571,tangent5.429,random/diagonal6.0. Gaussian4 differencesguidedvstandardtiny; doNOTtreatordinalrankwinthereasmeaningfulaccuracygain.
- StillNO independenttest/truthevaluation. Figuresstillnotgenerated. Allocationremaining~74min; mayexpirebeforeallSwirl10/backfillsfinish. ResumeonfreshsingleGPUallocationifneeded, preservingcompletedfits/failures. Needfinisherlongcontrols/evaluation/plots, thencryoifjustified, finaldocs/audit/resourcecleanup.

## CRITICAL current phase after primary completion
- All192primaryattemptscomplete:189fits,3full-covariance/uniform-cross failures(allSwirl10seeds:zero/invalidphysicalnorm). All8recipes3seeds8casesfinished inclGaussianGuided. NO independenttest/truthevaluationyet atthisentry.
- selection.json FROZEN. Globalmean-rankwinner gaussian_guided3.125, gaussian3.25,KDE4.5,fullguided4.125,full4.5,tangent4.875,diagonal5.5,random6.125. Curvedtransfermethod tangent. Failuresnowserializedasnullmeansplusfailed_seed_counts; internalrankingstillinfinity. extendfiltersnullmeans.
- validation_summary.json: leave-one-case-outalwayschoosesGaussianfamily(4ordinary,4guided); equal-familyrankalsoguidedwins. Initial/postfitselectorsdifferbanana,ring,GL6,GL10; short400-stepseed0screenfixesGLchoicesbutnotbanana/ring. No truthused.
- GaussianGuidedSwirl10initialNLLs11.871/11.883/13.178 vsordinary13.032/13.368/12.456, pilot9.850826. Guidancepartiallyimprovesconversionbutnotuniformly; doNOTclaimcompletefix. FinalGGfitsremainnearordinaryGaussian, specializedtangentbettervalidation. Full-guidedall3constructsuccessversusfullall3fail.
- Primaryrunner35320completedwithEXPECTEDfinisherbudgetguardfailure beforetruth(noSGDfailure). Oldallocation14492351CANCELEDbyusaftercompletion. Userjob14452283untouched.
- NEWallocation14502932, SAMEdella-l07g2,1GPU4CPU32GB,3h,granted~18:43local; remaining~2h59. New srun finisher_resume.log session ID fromtool juststarted (recordbelow). Itre-freezesidenticalchoices, runsGL10/swirl10longcontrols, then8caseindependentevaluations, then16plots.
- Latest18focusedtestsPASS22.51sec afterGaussianGuidedbranch. Afterwardschangesonlybenchmarkresumeskipfailure,selectorstrictJSONfailurehandling,sourceprotocolbootstrap,andplots/docs. Ruffchecksallmodifiednewcodepassrecently; finalstylecheckadvisable.
- New source initialization_protocol.py matchesplan; runnerbootstrapsmissingplan/logs. Docsrepro usesX64+srun.
- Prescribed-noisecheckon1024draws/casepassed: maxrelativekernelerror4.294e-9(GL10), Swirl10max2.87e-12, allpositive. Resultsnoise_checks.json/logs; seed73261. Itusedknownnoiseonly, notsignaltruth. CPUchecksession89138finished.
- Addedplotsfailuremarkers, readable8-methodlabels,2x5marginalgalleries,MCSE2significantdigits; longcontrolcaptionsstateAdamrestart.
- STILLREMAIN: finishlongcontrols+eval, auditMCnormratios/samplermoments/quadrature, generate/inspect16plots, conditionalcryo(singlefrozenCURVEDrecipe=tangent)4casesviaexistingtransfer script, updateDocsResults/figsREADME/rootrecommendation, finalmanifest+old101PNGhashes, release14502932. DoNOTspawnagents. Preservealloldwork.

- Newfinisheractiveexecsession30724 on14502932; logfinisher_resume.log.

- Continuation update: GL10 and swirl10 long controls COMPLETE. Finisher now starts gaussian4 independent evaluation; all primary/extension choices frozen before truth. Swirl10 KDE longer NLL~6.337 versus tangent retained~6.279. Active session30724, allocation14502932, >2.5h remains.
- Added scripts/audit_initialization_study.py: verifies frozen models, original data hashes, previous101PNGs, expected figures, noise/sampler/quadrature checks; writes source/data/figure provenance underartifacts. Runafterevaluationandcryotransfers with --allocation-ids 14492351 14502932. Noauditrunyet.
- DocsResults nowcontainsdata-onlyrecommendation(GG)andconstruction/sensitivityfindingsbutindependentresultsstillpending. RootREADMEaddsGGrecommendation; docshasexplicitAPIexample. figsREADMEstillneedsnewsection. LateststylechecksPASS; no numericalsourcechanges.

## Final phase after independent evaluation
- All primary + longer controls evaluated:193models. Finisher92531COMPLETED;16figuresgenerated. Sourceevaluationnowversion2: bananaGaussianinitial hadnonfiniteoneextremetailpointfromnoiseTT. DirectanddoublednoiseTTalsoNaN; addedconsistentmeasurement-onlyanalyticpositiveGaussian-noisequadratureforlogs<-25/nonfiniteondiagonalnoise(allbutSwirl10),checkedbydoubling1e-7. Allmodelsrescored; fits/selectionunchanged. NoNaNremaining. Additional7GaussianconvolutiontestsPASS23.18sec; previous18pass. Tailcounts/correctionsstoredinevaluationJSONs. Fixedtailbatch16avoidcompilemanyshapes.
- Primaryauditran:189success,193includingcontrols,101oldPNGsunchanged,maxquad1.052e-12,maxsamplermeanz3.387,no p2normalizationcheckabove6SE (GL10max2.086SE). RerunauditFINALafterallfigs/cryo/docsasinitialauditoverlappedsummaryfigurewriting.
- New16primaryPNGs reviewed: all6summaryplots+all10galleries havebeenviewed(actualGaussandlikelihoodplotupdatedandre-viewed). FinalCPUreplot9228complete(logfigures_final). Titlesnow3linesavoidcollision;testlabelsrotated35deg;failedpanelNoFit;colorsaturatestruthpeakcaption. Allfigurecontentlooksvalid. Needperhapsre-viewfinalSwirl10NoFitbutlayoutpatternchecked.
- MedianL2 KDE/Gaussian/GG/Tangent: gauss4 .4138/.0117/.0117/.0655;multi4 .3944/.3741/.3842/.1215;banana4 .2839/.1800/.1803/.4073;ring4 .3176/.2824/.3529/.4827;GL6 .5539/.5710/.4193/.7736;Swirl4 .3295/.3527/.3554/.2844;GL10 .4712/.6044/.6072/.6191;Swirl10 .6957/.7828/.7752/.3683. IMPORTANTGGnotuniversalrecoverywinner; frozenGGrecommendationbyobservablevalidationranks. OrdinaryGaussiancloseandsometimesbetter. KeepKDEoptionespeciallyGL10. DoNOToverstate.
- TangentSwirlall6fitsretainstep0. Swirl10KDEselected .6896→.6553 after2400extra;Tangentretains .3683. IndependenttestgainsTangentvsKDE:Swirl4 .0094906±.0014747;Swirl10 .0746863±.0036322. Data-onlyselectionhashesstillfrozen.
- DocsResultsnowfullprimarytable/analysis/links,no primaryplaceholderleft. READMErecommendationandfigsREADMEprimaryindexcomplete. NeedCRYOresultssectionandlinks+validationsectionfinal.
- CryotransferAUTHORIZEDconditionalfulfilledSwirl4median14%improvement,Swirl1047%. QueuedthenstartedSAMEallocation14502932 atstep.5,driverexecsession56994. Logs/cryoem_transfer.log loopsdim4,path_dim4,dim10,dim20. Latestdim10RUNNING. Currentallocationelapsed~54min,2h06remaining. NeedfinishthenCANCELown14502932(leavemdjob14452283untouched).
- Cryodim4done:oldNLL3.53147672,new3.53470401,gain-.00322729±.00155947→retainprevious. NewTTbeststep1200.
- Cryopathdim4done:old3.53924959,new3.53962290,gain-.00037331±.00118725→selectNEWbyoneSEthenlowerH2(37.482→33.915). Beststep1200. Donotclaimlikelihoodimprovement. Bothcryofiguresviewedlayoutgood;higherdimfigsstillpending. Transfersreusehistoricaldevelopmentvalidation,no freshcryo testororacle;oldgallerymodels/PNGspreserved,newselectedonlyunderartifacts/initialization_study/cryoem_CASE/selected.npz.
- Sourceauditnewcli --allocation-ids 14492351 14502932 writesmanifestoutsidefigs. Recordsdata/source/model/figurehashes,failurecounts,numericchecks,testlogs,optionalcryoselections. NeedfinalrunAFTERallfigwritesdone,thenfinalresourcecleanuprecord. Needfinalruffafterallsourceedits,linkcheckdocs,finaltestslogsno needrepeatnumericifunchanged. Sourceplotsandmeasurementwrapperonlychangedsince18tests,measurementuseslibraryverified7tests+observedquadraturechecks.

- IMPORTANT extra cryo follow-up: after dim10 primary stillimproving at1200 (NLL9.87498 vsold9.857318), decideddata-only toadd2400updates to BOTHdim10anddim20. Thisisnotchosenusingtruth, primarysyntheticprotocolunchanged. Scripttransfer_initialization_cryoem.py now--extra-steps, savescontinuation.json/sourcefithash, training_continuation/, primary_selection.json beforeupdatingselectedcomparison. Itautomaticallyresumesexistingcontinuationifflagomitted. FigurebudgetcaptionstatesAdamrestart+2400. No mathematicalkernelchanges. Ruffpasses.
- NewQUEUEDcontinuationdriverexecsession94290 onSAMEjob14502932 afterprimarydriver56994; logs/cryoem_continuations.log andcryoem_CASE_continuation.log. Loopdim10then20with--extra-steps2400. Mustfinishbothbeforefinaldecisions/audit/cancelallocation.
- Cryodim10PRIMARYdone:oldNLL9.85731783045,new9.87498081132,SE.0041032866,gain-.01766298→oldselected; roughnessold96.2632,new144.077. Savefinalcontinuationresultsseparatelyinthetablewithprimarycolumn.
- Cryodim20PRIMARYcurrentlytraining:step0NLL24.7786376 at73sec;step400NLL24.7116307at294sec. Expected1200+evaluationthendrivercontinuations. Allocationelapsed~1h05,~1h55remaining; ample for~20minextra.
- All16primaryimagesnowinspected;fixedtinytestxtickoverlapvia35degrotationandgallerylongtitlesvia3lines. FinalCPUreplot9228complete. Viewedall6summaries+all10galleries(actualfigs). Existingcryo4D2figsviewedlayoutgood. Primary10Dcryofigavailablebutneedsinspectaftercontinuationoverwritesfinal.20Dpending.
- Docsnowfullprimaryresults+figurelinks+caveats+Verification; figsREADMEprimaryindexcomplete; addedCryoEMtransferPROCEDUREincl2400followup,butNOcryoresultstableyet. Needfinaltable4rows old/1200/extra/selected,4linksfigsREADME,andactualconclusions. RootREADMErecommendationdone.
- Auditmaxnormz2.0854845,maxanalytic-taildoubledquad1.862e-10,allbelowthreshold. Numericcheck193models maxTTquad1.052e-12,meanmaxz3.387. Stylesallpassafterlatestmathwrapper; additionalauditfile/sourceexptsmallchangesruffpass. 25focusedtests(18+7)passed. Auditnowhashesalltestedsourcefiles+README/docs/figsREADME+stylelog; needsFINALrunafterallwritescomplete. Earlierauditoutputmodelchecksvalidbutfigurehashesmightstaleafterreplot. Finalupdateaftercryo+docs required.
- No missinglinksindocs/figsREADMEcurrently, andfigshasonlyPNG+MD. Previous101PNGspreserved. Finalshouldhave20newPNGs(total121), with2higherdimcryoPNGsupdatedinplaceaftercontinuations. Cryoselectedcheckpointsnewunderstudyonly; oldgalleryuntouched.

## COMPLETE
- All primary, synthetic continuation, independent evaluation, and cryo transfer work completed. Final20D continuationselectednewNLL24.652808659vsold24.656632940; pairedgain.003824281±.002793530, modest1.37SE, higherH2(new158.621vsold152.610).10Dretainsoldafter2400extra.4Dretainsold;path4newchosenbylowerroughnesswithin1SE.
- All20newPNGsinspected;101oldPNGsunchanged. Finaldocsandfigsindexincludecryoresultstableandcaveats. Relative links pass. Final audit afterallwrites passed:189primaryfits,3reportedconversionfailures,193syntheticevaluations,20newfigures,4cryotransfers.25focusedtestsandruffpass.
- Ownallocations14492351and14502932released; resource_cleanup.json recordsSlurmstates. Userjob14452283untouched. No remainingrequiredwork.
