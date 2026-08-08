# Run plan: objectives, status, and what is missing

**Living document. Last updated 2026-08-07.** This is the handoff page: what the
two planned runs are FOR, what is built, and what still blocks each. Numbers here
were measured on this machine unless marked otherwise.

Companion pages: `DATA.md` §0 (as-built artifact register), `DATA.md` §7 (the
rebuild chain), `rebuild_plan_2026-08-06.md` (the full ordered plan).

---

## 0. BLOCKERS — what stands between here and a launch

Nothing below is a bug. 170 tests pass, `AIONFLOW_STUB_ENCODER=1` runs the whole
loop AND the eval/posterior path on CPU, and a Run A smoke has completed on GPU
with the real encoder. What remains is listed per run.

### RUN A — no model-side blockers remain (2026-08-07)

All three landed and a 1-epoch GPU smoke ran the intended configuration end to
end on the cluster (job 37695391, exit 0), with `config.json` confirming
`fixed_combo ['spectra','z']`, `joint_dims ['logmstar_cigale','log_sfr','log_lx']`,
`joint_marginal ['log_sfr']`, `select_metric joint_complete`, `inject False`.

| # | was | status |
|---|---|---|
| A1 | joints not declarable | **DONE.** `--joint` declares it; `configure_heads_from_config` now CONSUMES `config.json`'s `joint_dims` instead of re-deriving from the module constant. Verified by reloading a checkpoint under a DIFFERENT same-arity `JOINT_PAIR`: dims come back correct under `strict=True`. `--joint-marginal` recovers rows with M* but no SFR (train split 12,803 -> 14,326). |
| A2 | no combo control | **DONE.** `--fixed-combo` reaches training AND both validation loops. Was worse than documented: there had been no combo control anywhere in `train-multi`. |
| A3 | selection on a pooled mean | **DONE.** `--select-metric`. On the GPU smoke: complete branch 1.885 (n=1,570) vs quadrature 1.566 (n=195) vs pooled 1.844 -- selection now uses the complete branch. |

Run A's invocation:
```
--fixed-combo spectra+z --joint-marginal log_sfr --select-metric joint_complete \
--drop-heads logmstar log_mbh_pan25 log_mbh_vo09 log_flux_p3 --no-inject
```

**What is NOT settled is the SCIENCE, not the machinery.** See the soundness
threats in section 2.

### RUN B — the head exists; the surrounding path does not

| # | was | status |
|---|---|---|
| B1 | counts columns not carried | **DONE** in `make_dr2_targets.py` (44 columns). Chain re-run into the LIVE sidecar still pending. |
| B2 | no Poisson head | **DONE.** Latent log-rate flow, Poisson marginal likelihood on observed counts by adaptive quadrature. Recovery verified: bias +0.009 dex, rms 0.193 against irreducible 0.200, posterior WIDTH 0.2005 vs true 0.200, calibration 0.964. N=0 gives an upper limit that tightens with exposure. |
| B3 | needs A1 | **DONE** with A1. |

**Still blocking a Run B smoke:**
1. The chain re-run, so the counts columns reach the live sidecar. Poisson heads
   are opt-in precisely because an absent column is a hard error.
2. **A count-marginal-likelihood eval path.** `eval_core` now REFUSES a Poisson
   checkpoint (`assert_no_poisson_heads`): it scores a density AT a target value
   and these heads have none, so it would have emitted a complete, plausible,
   entirely wrong table.
3. **An HR-from-rates path.** `hr_from_joint.py` refuses any joint that is not
   `(log_flux_p2, log_flux_p3)`, so it fails safe on a rate joint.

Run B's invocation:
```
--add-heads log_rate_p2 log_rate_p3 --joint log_rate_p2,log_rate_p3 \
--joint-quad-nodes 24 --select-metric joint_complete
```
`--joint-quad-nodes 24`, not 48: measured max error 4.9e-3 nats against 5.2e-3
at 48, i.e. past 24 the floor is float32 rather than quadrature. The FIXED grid
could not integrate a Poisson at all -- 7,424 nats of error at N=40,000 -- so
node placement is now per-source Laplace. See commit 5240425.

### Scope limiters, not launch blockers

| | limits what | note |
|---|---|---|
| **Merged source HDF5** (plan step 29) | Sample size only, and it is NOT blocked on any download. | **Measured 2026-08-07: 100% of the merged split's spectra are already on disk** (81,000 of 103,800 from the new shards, the rest from the old source HDF5, zero missing). Two mechanical steps remain: assemble the merged source HDF5, then re-run `build_manifest` twice — `manifest_merged.csv` currently has `split` 100% NaN and `source_row >= 0` on only 24,058 of 129,486 rows, and the stager filters on both, so merged staging would emit nothing today. **Image cutouts do not gate this**: `require_cutout` is off by default and image-less rows stage with a zero image, and Run A is spectra+z, so the cutout fetch (~5 more days) blocks nothing. Payoff is 6,717 galaxies against 1,407. |
| **eval path never run on a real model** | Reporting, not training. | Both eval scripts are fixed and tested, but no evaluation has ever executed on a DR2 checkpoint. Every performance number we hold describes a different model. |
| ~~**Presence-aware combo sampling + `encode_tokens` masking**~~ | **DONE 2026-08-08** — see `decisions.md` §12. | The four presence flags now reach the training loop (batch element 10), combos are drawn only from what a source HAS, `--fixed-combo` DROPS rows it cannot honour (count printed per split at startup), and a `[presence]` line per epoch reports what fraction of draws availability changed. Verified a no-op on dr2v2 (draw for draw identical when all four are present) and firing on a merged-like fixture: 41.7% image coverage, 58.3% of rows dropped under `--fixed-combo spectra+image`. |
| **Which HR to headline** | Run B's framing, not its code. | P2/P3 wins on measurability (median 68% width 0.697). Which is most AGN-diagnostic is unanswered — the literature agent died mid-search. Relaunch it. |

### Cheap and unresolved — reviewed 2026-08-07, only one survives

**Do:** switch the reliability cut to `p_any > NWAY_threshold6` row-wise instead
of our flat 0.5. Theirs is calibrated at the purity/completeness intersection
(median 0.0385) by a randomised-catalogue test; ours was chosen by eye and costs
GALAXY 12.5% against QSO 1.2%, a 10x asymmetry falling on the science arm.

**Dropped: the standardized-residual recomputation.** It tested whether the
1.5-6.6x conditional-vs-residual gap was a heteroscedasticity artifact, but that
gap was measured on the old model, sample and split -- all being discarded. The
detuid regroup alone moved 7,667 of 22,800 targetids. It would characterise a
model we are throwing away, and the question gets answered on the new posteriors
anyway.

**Reclassified, not dropped: the estimator null tests.** A permutation null and a
synthetic-independent null are a property of the ESTIMATOR, not the model, and
Run A's claim ("which objects get negative correlation") is a claim about the
SKEW of the per-source correlation distribution -- uninterpretable without one.
But they run AGAINST Run A's posteriors, so they are a **gate on reading the
result**, not work sitting in front of us. **BUILT 2026-08-07**, both of them,
in `scripts/posterior_correlation.py` (§2) -- so the gate is now a command to
run rather than code to write.

**Demoted: the alpha_ox existential check.** It was existential when the framing
was "predict X-ray luminosity from optical". Run A claims a posterior correlation
at fixed mass and Run B claims calibrated counts; alpha_ox speaks to neither. Keep
it as a methods-section baseline in case a referee asks. If ever run: only
computable for z >~ 0.45 (rest-frame 2500 A falls below DESI's 3600 A cutoff), and
the literature's 0.2-0.24 dex is monochromatic 2 keV on a clean blue QSO sample,
not comparable to our broad-band within-window RMSE without recomputing on our
own sources.

---

## 1. The scientific question, in one paragraph

Does AGN activity suppress star formation in the same galaxy? We attack it by
predicting a JOINT posterior over X-ray and host properties from optical/IR data
and reading the **posterior correlation** between them, per source. A negative
r(accretion, sSFR) at fixed stellar mass is the quenching signature. The whole
project exists because a joint posterior can express that and independent
marginals cannot.

---

## 2. RUN A — the collider run

**Objective.** Learn `p(log Lx, log SFR, log M* | spectrum, z)` and measure the
per-source posterior correlation `r(log Lx, log SFR | M*)`. Conditioning on M*
makes this identical to r(Lx/M*, sSFR), which is the quantity of interest. The
science claim is about the DISTRIBUTION of that correlation across sources:
which objects come out negative, and what else is true of them.

**Estimator, decided.** Two, computed post hoc from the same posterior draws, no
retraining, and the choice is deferred until real posteriors can be eyeballed:
- partial correlation `rho = (r_LS - r_LM r_SM)/sqrt((1-r_LM^2)(1-r_SM^2))`,
  equivalently `-P_ij/sqrt(P_ii P_jj)` from the precision matrix. This is exactly
  "linear regression with M* as a control". It equals the true conditional
  correlation ONLY under joint Gaussianity.
- kernel-weighted conditional covariance, which yields **rho as a function of M***
  and is both the honest estimand and the diagnostic for whether the Gaussian
  shortcut held.
Sanity check to hold: within a posterior r_LM and r_SM are both positive, so the
partial correlation MUST come out more negative than the raw one.

**BUILT 2026-08-07: `scripts/posterior_correlation.py`.** Both estimators, both
nulls (permutation and synthetic-independent), the sanity check, a per-source CSV
stratified by spectype and carrying z / log M*, and the distribution figure with
the nulls overlaid. Covered by `tests/test_posterior_correlation.py` (20 tests,
including recovery of a conditional correlation that is known by construction and
a joint whose rho FLIPS SIGN with M*, where the kernel estimator succeeds and the
partial correlation averages to zero -- the case that justifies carrying two).

Two operational facts that bind the run:
- **`posterior_structure.py` must be invoked with `--save-draws`.** Its `.npz`
  otherwise stores only per-source correlation MATRICES, which fix the estimator
  to the linear/Gaussian one; the kernel estimator and both nulls need the raw
  draws. Cost is ~140 MB at N=22,800 x S=512 x D=3. Without the flag the analysis
  script exits with instructions rather than silently reporting half the answer.
- **The sanity check is not quite the theorem it is written as above.** With
  c = r_LM r_SM and d = sqrt((1-r_LM^2)(1-r_SM^2)), `rho < r_LS` holds iff
  `r_LS (1 - d) < c`. That is unconditional for r_LS <= 0 (the script hard-asserts
  it) and for symmetric r_LM ~ r_SM, but it fails legitimately for large positive
  r_LS with strongly ASYMMETRIC r_LM/r_SM (r_LM=0.9, r_SM=0.1, r_LS=0.30 gives
  rho=+0.48). The script warns loudly, reports the violators' asymmetry so the
  algebraic corner can be told from a bug, and escalates above 5%.

**Settled decisions.** Conditioning is spectrum + z only (no WISE, no images).
Noise injection OFF. Train on the whole sample, evaluate stratified by spectype,
with the **QSO arm as an explicit negative control** (CIGALE SFR is known-biased
for QSOs in the same direction as the physical signal, so a matching result there
means we measured the pipeline). Rows missing BOTH SFR and M* are excluded; rows
missing exactly one should be marginalised over.

**Sample, and a gap between the decision and the code.** The rule above is the
decision. The code does not implement it: with `log_flux_p3` dropped,
`JOINT_MARGINAL` is empty, so all three dimensions are REQUIRED and a row with
M* but no SFR is dropped rather than marginalised. Measured 2026-08-07 on
`log_lx` finite and `det_like_0 >= 6`:

| | dr2v2 split | merged split |
|---|---:|---:|
| both SFR and M* present — what the code trains on today | 15,949 (1,407 gal) | 69,308 (6,717 gal) |
| M* present, SFR missing — recoverable by marginalising | 1,910 (240 gal) | 10,411 (1,259 gal) |
| SFR present, M* missing | 0 | 0 |
| **total under the decided rule** | **17,859 (1,647 gal)** | **79,719 (7,976 gal)** |

SFR is never present without M*, so honouring the decision needs exactly one
thing: SFR must be declarable as marginalisable. That is the minimum slice of
A1, and it is worth +1,259 galaxies on the merged sample, +18.7% on the arm the
science claim rests on. **Use merged.**

**What is missing (all model-side, all locally testable now):**
1. **Named joints.** `JOINT_PAIR` is a module constant with no CLI, so Run A and
   Run B cannot declare different joints, and editing it retroactively redefines
   every stored checkpoint. Needs `--joint NAME=dimA,dimB[:marginal=dimC]`
   persisted to `config.json`, plus a legacy state-dict remap verified against
   `mt-v3-lrfix-35416432` before anything is deleted.
2. **`--fixed-combo spectra+z`.** Both validation loops hardcode
   `("spectra","z","wise","image")`, so a spectra+z model would be checkpointed
   and early-stopped on a criterion it was never trained for.
3. **`--select-metric`.** `val_pair_mean` averages a 4-D joint NLL against 1-D
   marginals and gives every head an equal vote. The joint's per-branch stats are
   already exposed (`stats={}` out-param on `multi_target_nll`); the flag is not
   wired. Run A should select on the joint's COMPLETE branch alone -- the pooled
   number mixes two different dimensionalities and drifts with coverage.
4. **Merged source HDF5.** The expansion's spectra are `.npz` shards, not the
   merged source file, so the merged sample cannot be staged yet (plan step 29).
   This is the single thing between us and the 7,976-galaxy sample.

---

## 3. RUN B — the counts / hardness run

**BAND PAIR DECIDED 2026-08-07 by the user: P2/P3.** It is simultaneously the
best-measured pair (median 68% HR-posterior width 0.697 against 0.850 for
P3/P4; 706k + 596k net counts; 3.1% + 4.7% zero-count rows against 43% for P4)
AND the standard convention: eROSITA's P1/P2/P3 are byte-identical to 4XMM's
bands 1/2/3, so our pair IS the standard **HR2 = (R_1.0-2.0 - R_0.5-1.0) /
(R_1.0-2.0 + R_0.5-1.0)** (Webb et al. 2020, A&A 641, A136; Saeedi et al. 2024,
A&A 690, A152 use exactly this on eRASS1). Posterior on latent rates after Park
et al. 2006 (BEHR).

**BUT THE CLAIM MUST BE FRAMED AS CONTINUUM SHAPE, NOT OBSCURATION.** eFEDS AGN
are only ~10% obscured and DESI QSO selection biases us further toward type 1,
so an obscuration headline would be measuring a 10% tail. What HR2 tracks for
this sample is Gamma and the soft excess -- and that is a defensible
optical-to-X-ray channel: DESI spectra give M_BH and lambda_Edd, Gamma
correlates with lambda_Edd (slope ~0.3, Shemmer et al. 2008; Brightman et al.
2013), and soft-excess strength varies by ~5x across lambda_Edd (Chen et al.
2025, A&A 701, A144).

**REDSHIFT ENTANGLEMENT, act on this.** A fixed OBSERVED-frame split probes
N_H ~ 10^22 at z=0 rising to ~10^24 by z=4.75, roughly as (1+z)^2.5. Beyond
z ~ 2 an N_H = 10^22 absorber has redshifted out of eROSITA's band entirely
(Wang et al. 2004, ApJ 612, L109). So a single sample-wide HR2 distribution is
NOT physically interpretable. We hold spec-z for every source: report HR2 in z
bins, with constant-N_H and constant-Gamma tracks overlaid.

**eROSITA PUBLISHES NO HR COLUMN.** Confirmed against Merloni et al. 2024, the
HEASARC ERASS1MAIN listing and the DR2 catalogue paper. So the standing
"prefer catalogue values" preference cannot be satisfied by any HR -- every
option is home-made. What IS catalogue-native is the per-band RATE, which is
what Run B predicts. Frame it that way: we predict catalogue quantities and
derive the standard-convention colour from the joint posterior.

**Objective.** Learn `p(log lambda_band | all modalities)` for the X-ray bands and
validate the implied hardness-ratio posterior against the analytical one. If it
works, the collider analysis is redone as **HR x sSFR**, which is the more
principled and more informative version of Run A's question. Also produces the
updated information-gain-by-modality table.

**Architecture, decided.** The flow predicts a latent RATE, and the training
likelihood is Poisson on the observed counts:

    flow:        p(log lambda_b | inputs)
    likelihood:  N_b ~ Poisson(lambda_b * t_b + B_b)     t = APE_EXP, B = APE_BKG

Counts are exact integers, so **there is nothing to noise-inject** -- all the
uncertainty lives in the observation model, and `N = 0` is ordinary data. This is
Igo et al. 2026 (arXiv:2607.07795, Buchner co-author), so it is citable rather
than invented.

**Metric.** Marginal Poisson log-likelihood of held-out counts. Never needs a
"true" HR. Information gain for the modality table is model count-LL minus
KDE-prior count-LL, which is a reporting choice and not part of training.

**Why this supersedes the DET_LIKE threshold argument.** Gating on detection is
gating on the TARGET, which truncates p(y|x); it also selects on sky position
(median `ML_EXP_P2` 436 s for detected vs 373 s for zero rows). In count space
the question does not arise.

### PENDING DECISION — what exactly is the target, and where does noise live

Raised 2026-08-07, deliberately NOT resolved. "Predict the counts" and "predict
the rate" sound like the same plan and are not, and several distinct questions
hide inside the difference. Settle these before writing the head.

**(a) Counts or latent rate? DECIDED 2026-08-07: latent rate.** Two arguments,
the second decisive. (i) Predicting `N` makes the target discrete and
INSTRUMENTAL -- two sources with identical intrinsic rates but different exposure
have different `N`, so the model would have to learn the exposure map.
(ii) **HR is a function of RATES.** A predicted COUNT distribution carries Poisson
counting noise in its spread, so an HR derived from it is inflated by variance
that is not astrophysical, and recovering the true HR would mean deconvolving the
Poisson back out -- exactly what moving to counts was meant to avoid. With a
joint `lambda` posterior, HR is a deterministic transform of each sampled pair:
exact pushforward, nothing to invert.

So the head predicts a JOINT `p(log lambda_S, log lambda_H | x)`; the loss still
comes entirely from the observed counts through the Poisson.

**(a2) How to evaluate the marginal likelihood, and what it costs.** The
objective is `-log int p(N | lambda) p(lambda | x) dlambda`. The Poisson
FACTORISES across bands; the lambda posterior deliberately does not, so the
integral is genuinely multi-dimensional.
- Quadrature: 48 x 48 = 2,304 flow evaluations per source. Deterministic, but
  quadratic in the number of bands.
- Importance sampling with the flow as its own proposal: draw K lambdas, weight
  each by its Poisson likelihood. K evaluations, not K^2, so it stays linear as
  bands are added.
Sampling is cheaper and is NOT new cost territory: the retired injection scheme
already drew 50 samples per source and the existing quadrature runs 48 nodes.
CAVEAT: the sampled estimator of `log int` is biased low by Jensen (the IWAE
bound), tightening with K. Acceptable for training; for a REPORTED held-out
likelihood use quadrature or a large K so the quoted number is not a bound.
Measure the real cost with AIONFLOW_STUB_ENCODER before committing.

**(b) Where does noise actually enter, and is anything left to inject?** If the
target is the latent rate, the Poisson counting noise lives in the LIKELIHOOD,
not the label, and `N` is an exact integer -- so there is nothing to inject,
which is a real simplification over the log-flux era. The training signal becomes
the marginal likelihood `int p(N | lambda) p(lambda | x) dlambda`, needing a
quadrature over the flow's own density (the existing joint-quadrature machinery
is the model for it). Confirm that is what we want the flow to absorb: genuine
source-to-source scatter at fixed optical data, and nothing instrumental.

**(c) Weak identifiability exactly where we care.** For `N = 0` at low exposure
the likelihood is nearly flat over a wide range of `lambda`, so those sources
barely constrain `p(lambda | x)` and the fit is dominated by the bright end. The
faint censored regime is both the reason we moved to counts AND the regime the
likelihood constrains least. Decide whether that is acceptable, whether it needs
an explicit prior, and how we would detect it going wrong.

**(d) Aperture versus PSF-fitted, and whether HR inherits a band-dependent bias.**
NOTE: this is the eROSITA `APE_RADIUS` (in PIXELS, not arcsec -- the "4 inch"
this line used to say is the eSASS 4"/pixel scale, not the aperture; the aperture
is ~7-10 pix, i.e. ~30-40", set by the ~30" survey PSF), a DIFFERENT aperture
problem from the DESI 1.5" fibre. The fibre one biases SFR/sSFR and is a RUN A control needing
`LS10_shape_r`; this one biases HR between bands and needs `ML_EEF`. They share a
word and nothing else.
`APE_CTS` is photometry in a predetermined circular aperture; `ML_*` is PSF-fitted. The encircled
energy fraction is ENERGY DEPENDENT, so that aperture captures a different PSF
fraction in P2 than in P4. In a hardness ratio the EEF partially cancels but NOT
exactly, and the residual is a systematic in precisely the quantity we are trying
to measure. Quantify before trusting an HR built from aperture counts.
Measured 2026-08-07 on all 1,975,540 rows: **`ML_EEF_n` is a per-band CONSTANT**,
exactly one distinct value each -- 0.89230239 (P1), 0.88694167 (P2), 0.88360250
(P3, and the broad band carries the same number), 0.85624605 (P4). So the P2/P3
EEF ratio is 1.00378 and the P2/P4 ratio 1.03586: the systematic is a fixed
multiplicative offset per band pair, not a per-source nuisance. `APE_RADIUS` by
contrast **does** vary per source (median 7.31 pix in P2, 9.90 in P4 on the
25,454-row sample), so the aperture itself is not fixed across bands either.

**(e) Exposure and background are likelihood terms and must NEVER be model
inputs.** They are per-source and known, and feeding them would let the model
infer how well-measured a source is rather than what it is -- a cousin of the
sigma-conditioning ban, and the same failure mode: feeding the answer. Worth
stating in the head's docstring, because `APE_EXP` and `APE_BKG` will sit in the
same table as the counts and the temptation is structural.

**What is missing:**
1. **Counts columns: CARRIED 2026-08-07, chain re-run still pending.**
   `make_dr2_targets.py` now emits `ape_cts_*`, `ape_bkg_*`, `ape_exp_*`,
   `ape_radius_*`, `ape_pois_*`, `ml_cts_*`, `ml_rate_*`, `ml_exp_*`,
   `ml_eef_*` for the broad band and P1-P4 (44 columns, `ml_exp_1` excluded
   because it is already `ero_exp`), raw: no threshold, no transform. They reach
   the sidecar through its LEFT join on the universe, so no other script
   changes. The on-disk `dr2_targets.csv` does NOT have them yet -- the re-run
   waits on the merged rebuild. Verified on a temp rebuild (25,454 rows): all 44
   columns 100% finite, `ape_exp` strictly positive, `ape_cts` integer and
   never negative, zero-count fractions p1 6.9% / p2 0.7% / p3 1.3% / p4 13.6%,
   and the 52 pre-existing columns bit-identical to the live artifact.

   **The trap, as previously written here, was WRONG. Corrected 2026-08-07**,
   measured on all 1,975,540 rows of `eRASS3_Main_v1.3.fits`:
   - `ML_CTS_UPERR_Pn` **is** `ML_RATE_UPERR_Pn`, bit for bit: 100.0000% of rows
     in P2, P3, P4 and 1,975,537/1,975,540 in P1 (the 3 exceptions are NaN in
     both). That one column is in ct/s and needs `* ML_EXP_Pn`. Its own
     `TUNIT` says `count` and its `TCOMM` says "1-sigma upper counts error", so
     the catalogue's metadata is wrong, not just our reading of it -- which is
     exactly why it is worth a note here.
   - `ML_CTS_ERR_Pn` and `ML_CTS_LOWERR_Pn` are **already in counts**:
     `(ML_CTS_ERR/ML_RATE_ERR)/ML_EXP` has median exactly 1.000000 in every
     band, and exact equality with the rate column occurs on 0 rows for `ERR`.
     `LOWERR` looks equal on 3-42% of rows only because both are exactly 0
     there (the equal-row counts match the zero counts exactly: 448,013 in P1,
     63,967 P2, 94,422 P3, 826,995 P4). Multiplying these by `ML_EXP` would
     inflate an error bar by the exposure, ~340x at the median.
   - **The broad band `_1` is exempt entirely:** 0 rows of exact equality for
     all three, and `ML_CTS_UPERR_1/ML_RATE_UPERR_1` has median 335.94 = the
     median `ML_EXP_1`. All three band-1 error columns are in counts.

   None of these are carried (a Poisson likelihood needs no error bar), so the
   trap does not touch the new columns -- but anything that later reads
   `ML_CTS_*ERR` from the FITS must apply the correction to `UPERR_Pn` only.
2. **A Poisson head.** New likelihood; the current heads are Gaussian-ish flows
   over log-flux. A flow over a continuous density cannot represent the point
   mass at zero directly, so the rate is the latent and the Poisson is the
   observation model.
3. **Which HR to headline is UNRESOLVED.** Measured posterior widths (median
   68%): P2/P3 0.697 (best), P1/P2 0.728, P1/P3 0.744, P2/P4 0.821, P3/P4 0.850.
   P2/P3 wins on measurability. Which is most AGN-diagnostic is a physics
   question the literature agent died before answering. Relaunch it.
4. ~~Presence-aware combo sampling and `encode_tokens` masking.~~ **LANDED
   2026-08-08** (`decisions.md` §12). It was a no-op on dr2v2 -- `has_image` is
   exactly `has_spectrum`, 0 rows differ -- and would have been a third of the
   merged sample's image draws. What REMAINS unresolved is not the machinery but
   the reporting question it exposes: a run on merged conditions different rows
   on different modality sets, so any per-combo number is a statement about a row
   set as well as about inputs. `eval_core`'s `--sample common` is the answer for
   the eval tables; the training curve has no equivalent and its per-bucket
   series should be read with the `[presence]` rates next to it.

---

## 4. Sequencing

Smoke on `gpu_test` -> short LR/hyperparameter run -> **at most TWO long runs**,
one per track, run in parallel. GPU wall-time is the entire fairshare bill.

Both tracks share the trunk, so the CLS-capacity question is still open. It was
parked because overfitting was the binding constraint at n=12,915; the sample is
now 4.5x larger, so that reasoning no longer holds and capacity should not be
trimmed below what is needed.

---

## 5. If Run B fails

Fall back to Run A's Lx x sSFR posterior correlation and characterise the
negative-correlation subset: redshift, spectype, and the line-shape hypothesis --
that quenching galaxies show a second bump blueward of their emission lines,
measurable as line skewness or a two-component fit. Stack and superimpose their
spectra.

---

## 6. Standing constraints

- Never feed per-source measurement errors as model INPUTS. That feeds the answer.
- Prefer catalogue values over home-made computations.
- No em dashes in slide text.
- Never delete anything not produced by this project.
- Present open design decisions and wait for an explicit go before any sbatch.
- `logmstar` MUST be in `--drop-heads`: it is `sidecar: False` and no longer
  staged. The loader now raises rather than training a silent ghost head.
