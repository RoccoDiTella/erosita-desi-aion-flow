# Run plan: objectives, status, and what is missing

**Living document. Last updated 2026-08-07.** This is the handoff page: what the
two planned runs are FOR, what is built, and what still blocks each. Numbers here
were measured on this machine unless marked otherwise.

Companion pages: `DATA.md` §0 (as-built artifact register), `DATA.md` §7 (the
rebuild chain), `rebuild_plan_2026-08-06.md` (the full ordered plan).

---

## 0. BLOCKERS — what stands between here and a launch

Nothing below is a bug. Every known bug is fixed and covered by tests (105
passing, and `AIONFLOW_STUB_ENCODER=1` runs the whole loop on CPU). These are
missing capabilities, in the order they should be built.

### Hard blockers, RUN A (all model-side, all locally testable)

| # | blocker | why it blocks | effort |
|---|---|---|---|
| A1 | **Declarable `JOINT_MARGINAL`.** `JOINT_PAIR`/`JOINT_MARGINAL` are module constants with no CLI. | **CORRECTED 2026-08-07: A1 does NOT block a Run A smoke.** `--drop-heads log_flux_p3` already yields exactly `(logmstar_cigale, log_sfr, log_lx)` with the quadrature branch empty, verified by construction and by a clean 2-epoch stub run. What IS still missing is a way to declare SFR marginalisable, which the row-eligibility decision requires and which is worth 1,259 galaxies on merged (see §2). Full named joints remain a Run B blocker: a band joint is a different joint, and `config.json` records NO joint information at all, so a checkpoint cannot say what it was trained with. | small (marginal) / medium (full) |
| A2 | **`--fixed-combo spectra+z`.** | **Worse than first written:** there is no modality-combo control ANYWHERE in `train-multi`, so training is wrong too, not just the two validation loops that hardcode `("spectra","z","wise","image")`. Without it Run A silently becomes a full-dropout model early-stopped on a criterion it never trained for. A working `--fixed-combo` already exists on the retired single-target `train` parser (`main.py:918`) and can be copied rather than invented. Run B is the opposite and wants all four modalities, so the flag must pin A and leave B alone. | medium |
| A3 | **`--select-metric`.** `val_pair_mean` gives every head an equal vote and pools the joint's complete and quadrature branches, which are densities over DIFFERENT numbers of dimensions. | `best.pt` is chosen on a number that moves with coverage rather than model quality. Per-branch stats are already exposed via the `stats={}` out-param; only the flag is missing. | small |

### Hard blocker, RUN B

| # | blocker | why it blocks | effort |
|---|---|---|---|
| B1 | **Counts columns are not carried.** `dr2_targets.csv` has no `APE_CTS/BKG/EXP`, no `ML_CTS`, no `ML_RATE`, no `ML_EEF`. | The Poisson likelihood is impossible without them. Every ingredient is already in the FITS, so this is a column list plus a ~15 min chain re-run. **Trap:** `ML_CTS_ERR/LOWERR/UPERR` are in ct/s, not counts — multiply by `ML_EXP`. | small |
| B2 | **A Poisson head does not exist.** Current heads are flows over log-flux. | Needs `p(log lambda | inputs)` as the latent with `N ~ Poisson(lambda*t + B)` as the observation model. | medium |
| B3 | **Also needs A1** (a band joint is a different joint). | | |

### Scope limiters, not launch blockers

| | limits what | note |
|---|---|---|
| **Merged source HDF5** (plan step 29) | Sample size only, and it is NOT blocked on any download. | **Measured 2026-08-07: 100% of the merged split's spectra are already on disk** (81,000 of 103,800 from the new shards, the rest from the old source HDF5, zero missing). Two mechanical steps remain: assemble the merged source HDF5, then re-run `build_manifest` twice — `manifest_merged.csv` currently has `split` 100% NaN and `source_row >= 0` on only 24,058 of 129,486 rows, and the stager filters on both, so merged staging would emit nothing today. **Image cutouts do not gate this**: `require_cutout` is off by default and image-less rows stage with a zero image, and Run A is spectra+z, so the cutout fetch (~5 more days) blocks nothing. Payoff is 6,717 galaxies against 1,407. |
| **eval path never run on a real model** | Reporting, not training. | Both eval scripts are fixed and tested, but no evaluation has ever executed on a DR2 checkpoint. Every performance number we hold describes a different model. |
| **Presence-aware combo sampling + `encode_tokens` masking** | The merged sample only. | Measured no-ops on the current sample: `has_image` is EXACTLY `has_spectrum`, 0 rows differ. Real once the expansion is in. |
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
result**, not work sitting in front of us.

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
NOTE: this is the eROSITA 4" `APE_RADIUS`, a DIFFERENT aperture problem from the
DESI 1.5" fibre. The fibre one biases SFR/sSFR and is a RUN A control needing
`LS10_shape_r`; this one biases HR between bands and needs `ML_EEF`. They share a
word and nothing else.
`APE_CTS` is photometry in a FIXED 4" radius; `ML_*` is PSF-fitted. The encircled
energy fraction is ENERGY DEPENDENT (`ML_EEF_n` is in the catalogue), so a 4"
aperture captures a different PSF fraction in P2 than in P4. In a hardness ratio
the EEF partially cancels but NOT exactly, and the residual is a systematic in
precisely the quantity we are trying to measure. Quantify before trusting an HR
built from aperture counts.

**(e) Exposure and background are likelihood terms and must NEVER be model
inputs.** They are per-source and known, and feeding them would let the model
infer how well-measured a source is rather than what it is -- a cousin of the
sigma-conditioning ban, and the same failure mode: feeding the answer. Worth
stating in the head's docstring, because `APE_EXP` and `APE_BKG` will sit in the
same table as the counts and the temptation is structural.

**What is missing:**
1. **Counts columns are not carried.** `dr2_targets.csv` has no `APE_*`, no
   `ML_CTS`, no `ML_RATE`, no `ML_EEF`. Every ingredient is in the FITS; this is
   a column list in `make_dr2_targets.py`, then a chain re-run (~15 min).
   **Trap:** `ML_CTS_ERR/LOWERR/UPERR` are numerically identical to
   `ML_RATE_*` -- they are in ct/s, not counts. Multiply by `ML_EXP`.
2. **A Poisson head.** New likelihood; the current heads are Gaussian-ish flows
   over log-flux. A flow over a continuous density cannot represent the point
   mass at zero directly, so the rate is the latent and the Poisson is the
   observation model.
3. **Which HR to headline is UNRESOLVED.** Measured posterior widths (median
   68%): P2/P3 0.697 (best), P1/P2 0.728, P1/P3 0.744, P2/P4 0.821, P3/P4 0.850.
   P2/P3 wins on measurability. Which is most AGN-diagnostic is a physics
   question the literature agent died before answering. Relaunch it.
4. Presence-aware combo sampling (`ComboSampler.default()` can draw `('image',)`
   for an image-less source) and `encode_tokens` masking. **Both are no-ops on
   the current sample** -- `has_image` is exactly `has_spectrum`, 0 rows differ --
   and become real on the merged sample.

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
