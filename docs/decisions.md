# Decisions log — data, targets, errors, architecture, runs

Living record of every load-bearing decision, with exact formulas **verified
against the shipped data** (each formula below was recomputed from the source
catalogs and matches the sidecar to ≤1e-12 unless noted). Maintained under the
same rule as `pipeline.md`: any change to matching, targets, errors, or
architecture updates this file in the same commit.

---

## 1. Crossmatch and NWAY cleaning

**Original (paper) matching:** nearest LS10 counterpart within 5″ of the eRASS1
X-ray position → DESI DR1 spectrum. Known failure mode: chance alignments and
wrong counterparts.

**Cleaning (this branch):** join the Salvato et al. 2025 NWAY Bayesian
counterpart catalog (A&A 704 A344, VizieR `J/A+A/704/A344/dr1ls10`) on
`IAUName`, compare NWAY's chosen counterpart with ours per `targetid`.
**Operational same-object test = spec-z agreement** (our `ls_id` vs their
`LS10objid` packing could not be reconciled reliably): **plain |z−zsp| < 0.01**
(NOT (1+z)-normalized; verified to reproduce the 1,017 exactly), sitting in the
valley of a strongly bimodal distribution (88.7% at dz<1e-4, ~3% at dz>0.1).
Caveat: z-agreement is an identity *proxy* — distinct objects at equal z (pairs,
group members) would pass as `correct`. Classes in `match_quality.csv`
(30,441 targetids):

| class | n | actual rule | kept? |
|---|---|---|---|
| `correct` | 26,632 (87.5%) | NWAY counterpart has spec-z and dz < 0.01 | ✅ |
| `spurious` | 1,514 | NWAY assigns NO counterpart (below reliability cut; consistent with spurious detection) | ❌ |
| `ambiguous` | 1,033 | NWAY counterpart exists but has no spec-z — unverifiable | ❌ |
| `wrong` | 1,017 | NWAY counterpart has spec-z and dz ≥ 0.01 — different object | ❌ |
| `not_in_NWAY` | 245 | X-ray source absent from the NWAY DR1 sample | ❌ |

**`keep = (class == correct)` only** — conservative by design.

**Corroboration from the full Salvato+2025 read (2026-07-22):** their spec-z
compilation *includes DESI DR1* and drops sources with literature redshifts
disagreeing by >0.1 — so an NWAY counterpart without zsp (our `ambiguous`) is
either not our DESI object or has conflicting redshifts; dropping it is right.
Their own LS10 counterparts average ~85% (DET_LIKE≥6) to ~91% (≥8)
completeness/purity at the optimal p_any threshold — our Δz-confirmed keep class
is spectroscopically verified, i.e. stricter. §3.1.4: low p_any + DET_LIKE_0<7
suggests the X-ray source itself is spurious (our `spurious` reading).
**DET_LIKE=6 is the eRASS1 Main-catalog inclusion threshold with P(spurious)≈14%
(4% at ≥8; Seppi+2022)** — "detected" at 6 is catalog-grade, not certain.
Their catalogs carry per-source p_any + purity[6,7,8] if a tunable cut is ever
preferred over our binary keep. Paper archived at
`../summer2026/predicting_xray/papers/salvato2025_erass1_counterparts.pdf`.

**Why we trust the cut** (three independent validations):
1. Δz between our DESI z and NWAY's zsp is bimodal — mismatches form a distinct cloud.
2. The published paper model is ~2× worse on the rejected rows (R² 0.29 vs 0.57); dropping them lifts test R² 0.549 → 0.567 *without retraining*.
3. Every physically impossible staged value traced back is a reject: e.g. the log_lx = 48.07 row is a z = 4.706 object with `keep=False`; the logmstar < 5 rows are `keep=False`.

**Application:** the staged superset keeps the paper layout (32,092 manifest rows
incl. 3,389 duplicate targetids; staged 30,762 after 1,330 missing-cutout drops).
The cleaned configuration is a **runtime view** (`clean_split.csv`): keep-filter →
dedup (first occurrence) → **re-split after cleaning** (targetid-grouped 80/10/10,
seed 42 → 20,160/2,520/2,520 of 25,200). One copy of the data serves both;
view ≡ the previously materialized version (verified targetid-exact).

## 2. Targets — exact definitions

| target | definition | source columns |
|---|---|---|
| `log_ml_flux_1` | log10(ML_FLUX_1) — broad band 0.2–2.3 keV | eRASS1 `ML_FLUX_1` |
| `log_lx` | log10(4π D_L(z)² · ML_FLUX_1), Planck18 D_L | + DESI `z` |
| `hr32_u` | arctanh-space hardness, see below | eRASS1 `ML_RATE_P2/P3` |
| `logmstar` | stellar mass, h=1.0 Chabrier IMF — this is **FastSpecFit's** LOGMSTAR, reaching us via the `agngal` VAC | DESI VAC `logmstar` |
| `log_sfr` | log10 SFR averaged over 10 Myr, Chabrier IMF — **CIGALE**, a different SED fit than logmstar (deliberately; see 2026-07-28) | DESI DR1 CIGALE VAC `LOGSFR` |

**HR32, specifically** (all verified to ≤2e-15 on 30,375 rows):
- Bands: **P2 = 0.6–2.3 keV (soft), P3 = 2.3–5.0 keV (hard)** count rates.
- `HR32 = (R_P3 − R_P2) / (R_P3 + R_P2)` from `ML_RATE_P2`, `ML_RATE_P3`.
- Propagated error: `σ_HR = 2/(R2+R3)² · √(R3²σ2² + R2²σ3²)` from `ML_RATE_ERR_P2/P3`.
- Model space: `u = arctanh(clip(HR, ±0.999))` → |u| ≤ 3.8002, so the bounded
  ratio becomes unbounded and flow-friendly; `σ_u = σ_HR / (1 − HR_clip²)`.
  Report back via tanh. `hr32_ok` flags rows with sane rates.
- Caveat (open): low-count rows have enormous σ_u (median 0.51 but max ~8×10³)
  and pile at the clip; an S/N gate is planned before the HR paper claim.
  Sub-band non-detections are really *upper limits* — censored-likelihood
  treatment deferred. (Citation fixed 2026-07-22: 2022A&A...661A...3S is
  Salvato+2022, the eFEDS COUNTERPARTS paper — not an upper-limit method. For
  upper limits use Tubín-Arenas+2024, the eROSITA-DE upper-limit database;
  detection-threshold characterization is Seppi+2022.)

**logmstar sentinels:** `logmstar ≤ 2` (exactly 0.0 in practice, 4 rows) are
failed fits → NaN'd in the sidecar. Loaders exclude non-finite-target rows.

## 3. Measurement errors — construction and use

**Construction (`targets_extra.csv`):**
- Flux/Lx (asymmetric, from eRASS1 LOWERR/UPERR; verified ≤1e-16):
  `sig_lo = −log10(1 − LOWERR/F)` **capped at 1.0 dex** (cap active on 1 row),
  `sig_hi = log10(1 + UPERR/F)`. Medians: 0.188 / 0.161 dex.
  The same σ applies to `log_lx` (the distance modulus is z-deterministic).
- HR: symmetric `σ_u` as above.
- logmstar: **no per-object σ exists in the VAC** (external fetch non-viable), so
  a spectype floor: **0.2 dex (GALAXY), 0.3 dex (QSO/STAR)** — QSO masses are
  AGN-contaminated. Honest framing: for logmstar, "convolve" deconvolves a
  class-level noise floor (uncertainty-aware regularization), not per-object
  measurement error. `spectype` is carried for per-class eval splits.

**Use in training (`--error-mode`):**
- Kernel: **split-normal** (two-piece Gaussian; σ_lo below the mode, σ_hi above)
  — matches the asymmetric X-ray errors; reduces to a Gaussian when σ_lo = σ_hi.
- `convolve` (default, principled): the flow models the latent (deconvolved)
  value t; the likelihood of observing y is
  `log p(y|x) = log ∫ p_flow(t|x) · K(y|t; σ_lo, σ_hi) dt`,
  computed by fixed Gauss–Legendre-style quadrature (41 nodes spanning ±5σ),
  vectorized over nodes. Validated analytically: a N(0,1) flow convolved with a
  Gaussian kernel recovers N(0, 1+σ²).
- `inject`: sample ε ~ split-normal per step, add to the standardized target
  (documented as broadening, not deconvolution). **Draws truncated at 1.5σ per
  side** (inverse-CDF truncated half-normals; side odds preserved) — keeps
  single draws out of the far tail (huge-σ HR rows); slightly narrows the
  effective kernel. Multi-draw: `--inject-samples 8`.
- `none`: paper behavior; σ columns ignored.
- σ enters standardized space as **σ/std** of the target standardizer.
- **Eval consistency rule:** a convolve-trained checkpoint is *scored* through
  the same kernel (`log_prob_convolved`) — scoring the deconvolved density at
  noisy y against a KDE prior fit on noisy y understates IG structurally.
  `error_mode` is read from the checkpoint config, so old checkpoints keep old
  behavior. (This asymmetry was a bug we introduced and caught by audit.)

## 4. Architecture decisions

- **Frozen AION-base encoder** (`polymathic-ai/aion-base`): per-modality codecs →
  **one joint transformer forward per modality combo** (tokens cross-attend
  across modalities; consequence: cached tokens would be combo-dependent).
  Raw modalities staged; encoding happens per batch at train time.
- **Heads:** paper = q4/l2 (4 queries, 2 layers, 8 heads, MLP 512×512 → 256;
  ~16.8M). **V1 minimal = 1 query, 1 layer, hidden 128 (~7.8M)** — same
  interface, defaults in code remain the paper values so the baseline is always
  runnable. V2 (planned) replaces AION with a small self-trained encoder.
- **Flow:** conditional NSF (zuko), 1-D target, standardized; KDE (Scott) prior
  on train targets anchors IG.
- **Training mix:** per batch one modality combo — 25% singles / 25% pairs /
  25% triples / 25% all-inputs. Checkpoint criterion: all-inputs val NLL.
- **Per-target runs** (no multi-target head): flux, Lx, logmstar, HR trained
  separately — cheaper to reason about, and error models differ per target.
- **Leakage rules:** split by unique targetid; standardizer + prior fit on the
  training split only; re-split happens *after* cleaning.

## 5. Data-engineering decisions

- **One canonical staged copy** (`staged_paper`, paper layout incl. dupes) +
  cleaned configuration as a **row-selection view** — no second materialization.
- **Validation gate before GPU time** (`scripts/validate_staged.py`, wired into
  `train.sbatch`): schema, alignment, dedup/leakage/fractions, clean filter,
  σ > 0 + coverage, physical ranges (fractional, unit-slip detector),
  derived-column consistency (recompute log-flux/log-Lx row-wise), fits
  coverage, dataloader 10-tuple contract + real batch_nll forward.
- Loaders exclude non-finite targets (secondary targets are partially measured).
- Idempotency by count, not existence (the fits_pool lesson).

**OVERNIGHT multi-target V3b launched (job 34994658, 2026-07-25 ~01:30):**
`train-multi --bucketed` — 8 heads, bs 896 / chunk 224, **inject-samples 50
broadcast** (user choice; smoke A/B: k=50 improved 1-epoch val 9.95→9.75 at
IDENTICAL wall/VRAM — draws are free under one conditioner pass), lr 1e-4
constant, adapter wd 1e-1, availability-weighted pair-mean criterion, 40 ep /
patience 6, 8h limit. Gated on 4 smokes: bucketing verified, chunk cap added
(heavy bucket = 46% of mix), validation batch capped (MIG OOM), k=50 verified.
Broadcast injection also retrofitted to the single-target path (old loop
re-ran the conditioner per draw).

**MULTI-TARGET V3B RESULTS (job 34994658 + eval 35017708, 2026-07-25) — the
project's best numbers, from ONE 3-hour run:** early stop at ep 25 (best 19),
46.5 samp/s bucketed, 28.3 GB. Test all-inputs vs the single-target baselines:
**flux 0.603 (vs 0.572)**, **Lx 0.917 (vs 0.905)**, **logM★ 0.731 (vs
0.648)**, P1 0.374 (0.320), P2 0.415 (0.380), P3 0.415 (0.378); only P4 loses
(−0.06 vs 0.071 — noise-bound). Six of seven heads BEAT their specialized
runs: multi-task transfer is a broad win, and flux 0.603 is the best flux
number to date (published 0.549 → clean single 0.572 → multi 0.603). IG:
mstar 1.36 (vs 1.09), Lx 1.19 (vs 1.17); flux IG 0.282 slightly under the
single-target 0.323 (sharpness traded for point accuracy).
**HR from the JOINT (P2,P3) posterior: corr +0.238 on the hr32_ok subset vs
+0.10 for independent bands — the correlated posterior more than doubles the
hardness signal**, per-source width 0.508 vs 0.551. Still corr²≈0.06 (eRASS1
noise floor stands), but the joint-flow mechanism works exactly as designed.
Outputs: `outputs/mt-v3b-8head-34994658/eval/` (multi_test_metrics.csv —
15 combos × 8 heads; hr_joint_posteriors.csv).

**HR32 AS AN IMPLIED TARGET (job 35049997, 2026-07-25) — a real IG for a
target that was never trained.** Because the per-band ECFs are exactly
constant, HR = tanh(d·ln10/2) with d = logF3 − logF2 + (C2−C3): hardness is a
monotone function of the log-flux DIFFERENCE alone, so p(HR|x) is the joint
(P2,P3) posterior marginalized along a shear (unit Jacobian) times an analytic
transform factor. Computed by line-integral quadrature (96 nodes, broadcast
under one conditioner pass); algebra + quadrature validated against an
analytic Gaussian to 1e-14. Strictly preferable to sample+KDE (no bandwidth
bias, no MC noise) — user's call.
Results: all measured (n=2,169) **IG +0.090 nats (1.09×), corr +0.166**;
well-measured hr32_ok (n=415) **IG +0.121 (1.13×), corr +0.249**, median 68%
half-width 0.408. R² stays ≈0 (posterior is under-dispersed vs noisy measured
values), so the honest claim is *information*, not point accuracy. Note the IG
magnitude is comparable to the direct-HR model's flat 1.13× — but THAT was
shape-only (flat across all 15 combos, corr≈0); here corr +0.25 is genuine
per-source signal. Ranking of evidence: correlation > IG for this target.
Runtime 2m50s. Output: `eval/hr_implied_target.csv`.

**V2 TRAINING RUN (accumulated buckets, P4 dropped; job 35073203, 20 ep,
2.41 GPU-h) — diagnostics, not a performance win.** 7-head val sum 7.263 vs the
per-step run's 7.017, BUT still descending at epoch 20 (last-5 slope
−0.034/epoch), so the gap is undertraining: accumulation gives ~22 steps/epoch
instead of ~114. Per-epoch wall-time identical (433 s), so epochs are the fair
unit and ~27 epochs would match. Verdict: accumulation is roughly wall-time
NEUTRAL and buys interpretability, not accuracy.
**What the diagnostics establish:**
- Per-head convergence is wildly uneven: P1/P2 by epoch 5, Lx/P3 by 10, flux
  and the joint head by 13, **logM★ still improving at 20** (−0.031/epoch) and
  is the only reason the total keeps falling. Per-head schedules or freezing
  are justified.
- **Overfitting is real and linear from epoch 1**: val−probe gap (identical
  protocol) grows 0.009 → 0.062 nats. First honest generalization measurement.
- **Spikiness fully explained**: per-bucket decomposition of log Lx separates
  into four smooth curves, z/W 0.70 > image 0.60 > spectra 0.37 > all 0.33.
  The old step-level "noise" was this 0.35-nat spread aliased by logging.
- **No head dominates the shared trunk**: influence shares stay 0.10–0.20 all
  run, so the EMA weighting is NOT misallocating despite loss weights spanning
  4×. The earlier concern is measured and dismissed.
- **Adapters move 10–20× more than CLS tokens and the shared MLP** (relative
  movement 0.06 vs 0.005 vs 0.003 at step 450, still decaying) — evidence for
  a lower adapter LR or stronger decay in the next regime.
Figure: `docs/figures/fig_v3b_training.png` (docs/make_training_diagnostics.py).

**V3 RUN: accumulation + separated adapter LR (job 35416432, 30 ep, early stop
at 22, 4.05 h) — closes the accumulation question.** Config: accumulated
buckets, P4 dropped, lr 3e-4 for flows/MLP/CLS with **adapter-lr 1e-4** and
beta1 0.95. Test all-inputs vs the per-step multi-target run: **flux 0.604
(was 0.603), Lx 0.919 (0.917), logM★ 0.744 (0.731)**; bands ~0.010 lower
(P1 0.364, P2 0.404, P3 0.406). Flux IG 0.300 (was 0.282). HR implied:
IG +0.115 / 1.12x on hr32_ok, corr +0.173 (was +0.249 — the correlation is
noisier than the IG and moved against us).
**Mechanism confirmed:** the flows had been frozen by a shared LR (update/weight
1.2e-4, ~10x below the healthy band) because they are standard-init (|w|~40)
while the adapters are zero-init (|w|~3); separating the LR moved them to
7.5e-4, and logM★ — the head that had been furthest from converged — gained
most. **Verdict: accumulation is wall-time neutral and now matches per-step;
it buys interpretability, not accuracy.** New binding constraint is
overfitting: the train-probe gap reaches 0.157 nats (3x the previous run), so
the next lever is regularization or a shorter schedule, not more LR.

**wandb retention policy (2026-07-25, applied):** delete failed/crashed/
superseded runs and plumbing smokes whose numbers live in this registry
(17 deleted); KEEP every finished run with results, the calibration-grid runs
(decision evidence), and v3-cls-34356743 (state "failed" from its eval OOM but
holds the real V3a curves). Project now: 19 runs, all meaningful.

## 6. Run registry and performance

All runs: wandb project `erosita-desi-aion-flow`; outputs under
`$AIONFLOW_ROOT/outputs/<run-id>/` (netscratch — copy keepers to holylabs).

| run | config | data | epochs | all-inputs R² | exp(IG) | status |
|---|---|---|---|---|---|---|
| paper (published) | q4/l2, none | noisy, paper split (n_test 3,054) | 50 | **0.549** (best combo 0.554) | 1.40 | reference |
| smoke v1 | V1, convolve | clean view, 667 rows | 2 | 0.29 (plumbing only) | — | done |
| smoke repro | q4/l2, none | paper subset, 667 rows | 2 | 0.02 (plumbing only) | — | done |
| `v1-clean-log_ml_flux_1-34083239` | V1, convolve | clean view (25,200) | 15 | **0.569** (best combo S+z+I: 0.572) | 1.23 (IG +0.205) | ✅ done (1:34 wall) |
| `paperhead-clean-log_ml_flux_1-34089921` | **q4/l2**, convolve | clean view | 15 | 0.555 (best combo S+I: 0.556) | **1.24 (IG +0.218)** | ✅ done (1:43 wall) |
| `v1-clean-inject-log_ml_flux_1-34171110` | V_simple, **inject(8), 1.5σ trunc** | clean view | 15 | **0.572** (plain-LL eval; best combo = all-inputs) | 1.38 (IG +0.323, plain LL) | ✅ done |
| `paperhead-clean-inject-log_ml_flux_1-34171111` | q4/l2, **inject(8), 1.5σ trunc** | clean view | 15 | 0.565 (best combo S+W+I: 0.568) | 1.36 (IG +0.306, plain LL) | ✅ done |
| `v1-clean-inject-log_lx-34185556` | V_simple, inject(8) | clean view | 15 | **0.905** (z-only 0.835, S-only 0.884) | 3.23 (IG +1.172) | ✅ done |
| `v1-clean-inject-hr32_u-34185558` | V_simple, inject(8), **σ_u≤1.0 gate** | clean view (n_test 2,121) | 15 | ≈0.00 (no point signal, as expected) | 1.13–1.14 flat across combos (IG +0.13) | ✅ done |
| `v1-clean-none-logmstar-34185562` | V_simple, none (no real σ) | clean view | 15 | **0.648** (S-only 0.383, S+W 0.561 — WISE is the big adder) | 2.98 (IG +1.090) | ✅ done |
| `v1-spec-mask-log_ml_flux_1-34193770` | V_simple, none, spectra-only + token-mask augment | clean view | 12 | spectra-only 0.529 (vs 0.541 unmasked full model → surrogate faithful, augment costs ~0.01) | 1.34 | ✅ done (Shapley surrogate) |
| shapley sweeps 34193771 | 2 full + 4 line(Owen) sweeps | test view | | 7 lines Σφ 0.007 nats vs continuum 0.070; MgII top line | | ✅ done (33 min) |
| `band-p1-34629507` | V_simple, inject(8), σ≤1.0 | clean view (n_test 1,976) | 15 | 0.320 | 1.15 | ✅ done |
| `band-p2-34629508` | V_simple, inject(8), σ≤1.0 | clean view (n_test 2,372) | 15 | **0.380** | 1.17 | ✅ done |
| `band-p3-34629519` | V_simple, inject(8), σ≤1.0 | clean view (n_test 2,310) | 15 | 0.378 | 1.14 | ✅ done |
| `band-p4-34629535` | V_simple, inject(8), σ≤1.0 | clean view (n_test 1,176) | 15 | 0.071 (no signal → dropped from V3) | 0.94 | ✅ done |
| `v3-cls-34356743` | **V3a**: LoRA r8 all blocks + trainable CLS, bs 128 + ckpt | clean view | 15 | 0.575 flux (tie with frozen V_simple) | 1.32 | ✅ done, not pursued |
| `multi-35073203` | **V3b** read-only CLS, 7 heads, accumulated buckets, shared LR | clean view | 20 | flux 0.603 / Lx 0.917 / logM★ 0.731 | — | ✅ done (diagnostics run) |
| `multi-35416432` | **V3b** + separated adapter LR (1e-4), β₁ 0.95 | clean view | 30 (early stop 22, 4.05 h) | **flux 0.604 · Lx 0.919 · logM★ 0.744 · P1 0.364 · P2 0.404 · P3 0.406** | 1.35 / 3.28 / 3.92 | ✅ **current best** |
| shapley 34345374 | 13-line catalog, drop-mode | test view | | Hα 64.0 > Hβ 36.8 > OIII 27.6 mnats | | ✅ done (1h22) |
| shapley 34672894 | v4, merged Hβ+OIII player, guard 10 | test view | | merge 48.5; guard 10 caused dilation artifacts | | ⚠️ superseded |
| shapley 35483424 | 12 redshift bins, pinned guard | test view | | — | | ⏳ to harvest |
| paper reproduction | q4/l2, none | noisy, paper split | 50 | *deferred* | | planned |

**Queue decision (2026-07-21):** compare **V1 vs paper head, everything else
identical** (clean view, convolve, flux, 15 epochs) — the head A/B informs what
to try next. Per-target runs resume after. Every completed run gets the standard
**evaluation packet** (`scripts/make_run_packet.py`, ported from the RunPod
packet-v5): diagnostics PDF (scatter grid + IG histograms + calibration),
upset-style combo figure, per-spectype and per-redshift slices.

Comparison caveats: the cleaned-view test set (2,520 cleaned rows) ≠ the paper
test set (3,054 noisy rows) — deltas vs the paper are indicative, not
row-comparable (the provenance report quantifies overlap). HR reports
IG as primary (R² is not meaningful near the clip). Convolved IG is
systematically smaller than plain IG for the same model (the predictive
includes measurement noise), so IG is only comparable within an error mode.

**First result (2026-07-21): V1 + clean + convolve reaches all-inputs R² 0.569
in 15 epochs** — above the published 0.549 (noisy test) and at the level of the
paper-model-on-clean anchor 0.567, with 2× fewer head parameters and 3× fewer
epochs. Modality ordering matches the paper qualitatively (z < wise < image <
spectra < multi-modal); spectra-only 0.512 (paper 0.480), z-only 0.199 (paper
0.140). z adds ~nothing on top of S+I (0.572 → 0.569 all-inputs), echoing the
paper's z-drop quirk.

**Head A/B verdict (2026-07-21):** on identical clean data + convolve, V1 vs
paper q4/l2: V1 better point prediction (R² 0.569 vs 0.555), paper head better
density sharpness (IG 0.218 vs 0.205 nats); both deltas ≈1σ of test-set noise.
**Decision: the V1 head is not the bottleneck → use V1 for the per-target sweep**
(2× cheaper head; paper head kept for the eventual paper-config reproduction).
Kernel-misspecification note: observation-space eval is proper (wrong σ cannot
inflate it), but the *latent* decomposition is not certified by it — the
σ-stratified calibration panel must be flat before any intrinsic-scatter claim.

**Inject A/B verdict (2026-07-22, jobs 34171110/34171111):** truncated inject(8)
on clean flux, plain-LL test eval. V_simple all-inputs R² **0.572** / exp(IG)
1.38, V_PAI 0.565 / 1.36 — V_simple wins both point AND density metrics this
time (under convolve the density edge had gone to V_PAI). Same ~1σ scale, but
now consistent across both metrics. Note: V_PAI led on *val* NLL throughout
training (1.031 vs 1.035) yet lost on test plain LL — the val criterion under
inject scores noise-perturbed draws, so small val edges do not transfer.
Inject R² matches convolve (0.572 vs 0.569, same model class) → operating-mode
choice costs nothing in point metrics, as expected. V_simple + inject stands as
the per-target sweep configuration (already in flight).

**Per-target results (2026-07-22, jobs 34185556/34185558/34185562):**
- **log Lx 0.905 / exp(IG) 3.23** all-inputs. Decomposition: z-only 0.835 (the
  D_L(z)² term), spectra-only 0.884 (spectrum encodes z implicitly + flux info);
  the increment over z-only (0.835 → 0.905) is the part the paper's flux R²
  speaks to. RMSE 0.229 dex.
- **logM★ 0.648 / exp(IG) 2.98** all-inputs, no error model. Modality story
  differs from flux: spectra-only just 0.383, **WISE is the big adder**
  (S+W 0.561; W1/W2 trace stellar mass; DESI fiber spectra lose aperture flux).
  Caveat: X-ray-selected sample is AGN-rich and the VAC photometric mass is
  cleanest for GALAXY spectypes — slice by spectype before claiming anything.
- **HR (hr32_u, gate σ_u ≤ 1.0): R² ≈ 0 in every combo** — no point-level
  predictability at eRASS1 depth, matching the honesty expectation. exp(IG)
  1.13–1.14, *flat across all 15 combos including singles* → the flow learned a
  sharper-than-KDE population density (shape gain), not source-level
  discrimination. Report HR as "no evidence of per-source information" for now.
- Flux Shapley surrogate (34193770): spectra-only 0.529 with mask augmentation
  vs 0.541 unmasked (full model) — augmentation costs ~0.012 R², surrogate is
  faithful enough to attribute.

**Line/continuum Shapley findings (2026-07-22, job 34193771, 33 min):** total
attributed (unmasked vs fully-median-filtered spectrum) 0.077 nats of the
0.291-nat spectra IG → **~74% of the spectral flux information survives 80 Å
median smoothing** (broad SED/colors); fine structure carries ~26%, of which the
7 named lines are ~10% (Σφ 0.0073 nats). Line ranking: **MgII 0.0027±0.0005**,
then OIII 0.0013, Hβ 0.0011, NeV 0.0010, OII 0.0008; Hα and HeII consistent
with zero (Hα null is z<0.5-specific — coverage). Hottest continuum bins:
1098–1213 Å (Lyα region, z>1.7 only, 0.020±0.010), 5970–6596 Å, 2205–2691 Å
(UV FeII / small-blue-bump territory). Caveat as designed: line availability is
a z-window, so per-line φ conditions on coverage.

**Modality Shapley (exact, 16 coalitions from the test tables — no extra
compute):** share of all-inputs IG — flux: S 44 / I 28 / W 21 / z 8%.
Lx: S 46 / z 34 / I 17 / W 2% (distance info split between S and z — the
spectrum carries z). M★: S 42 / I 24 / W 22 / z 12%. HR: flat 23–26%×4
(population-shape gain evenly credited = no modality-specific signal).

**Pairwise Shapley interaction index (exact, same 16 coalitions; negative =
redundant, positive = synergistic; IG nats):** Lx spectra+z **−0.679** (the
spectrum-carries-z redundancy, quantified) and spectra+image −0.317, z+image
−0.166 (image also proxies distance). M★ spectra+wise **+0.154** — the only
large synergy: per-aperture SED × total NIR luminosity jointly pin the mass —
plus wise+image +0.097; but spectra+z −0.232. Flux: mild redundancy everywhere
(all pairs negative, max spectra+image −0.118) — every input reads the same
source brightness. HR: uniformly −0.03…−0.04, four redundant copies of the
same population-shape gain. Figure: `docs/figures/fig_modality_interactions.png`.
Relation to the naive synergy IG(A∪B)−IG(A)−IG(B): that is the empty-background
Harsanyi dividend m(AB); the index averages the same contrast over all
backgrounds, I_AB = m(AB) + ½[m(ABC)+m(ABD)] + ⅓m(ABCD).

**Full Möbius/Harsanyi decomposition (exact):** net dividend by interaction
order — flux +0.81/−0.84/+0.42/−0.07, Lx +2.65/−2.01/+0.63/−0.09, HR
+0.48/−0.71/+0.48/−0.13: the alternating, damping cascade of ONE shared signal
read four ways (HR is the pure limit: every nonempty coalition ≈ 0.12 nats).
**M★ breaks the pattern: order-2 net is POSITIVE (+0.67)** — the only target
where inputs genuinely combine. Figure `docs/figures/fig_interaction_orders.png`.

**Line-Shapley methodology audit (2026-07-22, user-prompted) — three real
defects found in v1, all fixed for the v2 re-run (commit 26219e5):**
1. *Replace-mode leakage:* the 101-px (~81 Å observed) median filter fails to
   erase BROAD lines at low z (broad Hα FWHM 5–10k km/s = 140–290 Å observed at
   z≈0.3), so "removed" Hα left a smoothed line bump → φ(Hα) biased low. Fix:
   **mask_mode=drop, pure removal (user's design)** — tokens dropped via AION's
   native `embed_inputs` input mask (partial input = the pretraining task);
   dropped positions become group id −1 and the head skips them via
   key_padding_mask (pad-invariance unit-tested). NOTHING is replaced or
   imputed in pixel space (an earlier sanitize-fill variant was rejected: the
   fill is synthetic and coalition-dependent, contaminating every marginal).
2. *Codec-grid misalignment:* token wavelengths assumed the DESI 3600 Å grid but
   the codec latent grid starts at 3500 Å (aion/codecs/spectrum.py) — 100 Å =
   3.9 tokens; the integer probe absorbed 3, leaving all masks ~23 Å blueshifted
   (cores still masked; red wing leaked). Fixed: geometry on the codec grid,
   dual-spike median-offset probe.
3. *Codec receptive field:* the spectrum codec encoder is a ConvNeXt (k7
   depthwise, 4 scales) — theoretical RF ≈ ±26 tokens, so KEPT neighbour codes
   see the line. Fix (user's design): `scripts/codec_leakage_probe.py` injects
   synthetic lines and measures the EFFECTIVE leak radius empirically; line
   windows are then dropped with a **guard band** of that radius
   (`--line-guard-tokens`, auto-set from the probe's RECOMMENDED_GUARD output
   in the launch chain; default 1). Guard applies to line players only —
   continuum bins are 10–20 tokens wide, boundary blur is fractionally small.
   Cost: guard-band continuum info is attributed to the line (bias up, bounded
   by 2·guard·(per-token continuum density) ≈ 0.6 mnats/side at v1 densities).
   Broad-line windows widened ±80→±120 Å (BLR wings).
Fairness note (per-token view of v1): lines 0.215 vs continuum 0.293
mnats/token — the continuum's 10× total dominance is mostly token count
(~238 vs ~34 tokens); OIII has the highest per-token density (0.34).
v2 sweeps add: line-PAIR Shapley interactions (Owen base + 29 flip configs =
all 21 pairs + bonus single marginals; estimator verified against an analytic
stub) and a direct coalition summary (full / lines-only / continuum-only /
norm-only). Prevalence accounting (user request): shapley_table now carries
`phi_per_token` (window-size fairness) and `availability_frac` +
`phi_population` = φ·P(available) (population-level importance vs the
conditional φ). Hα caveat stands regardless: available only at z<0.5 (33% of
sources), so φ(Hα) is conditioned on that subpopulation and its SE (±0.5 mnats
in v1) cannot separate 0 from Hβ-level.

**Both-detected bookkeeping:** slide's 30% = both bands DET_LIKE≥6 (7,984 —
verified exact); the sidecar `hr32_ok` 16.9% = both rates > 2×err (stricter,
strict subset). Slide is correct. HR noise-floor on the DET_LIKE≥6 subset is
WORSE than on hr32_ok (Var 0.061 vs E[σ²] 0.095, ceiling −0.55; median σ_u
0.31): detection likelihood ≠ rate precision. HR verdict unchanged.

**HR verdict (2026-07-22): not learnable at eRASS1 depth — do NOT re-run on the
detected-only subset.** Noise-floor analysis (clean sample, hr32_u/σ_u
sidecars): at the σ_u≤1.0 gate, Var(u)=0.16 < E[σ_u²]=0.25 → the observed HR
spread is entirely explainable by measurement noise (R² ceiling negative). The
both-detected subset (`hr32_ok`, 4,485 = 16.9% of clean, 87% QSO) is *also*
ceiling-negative (Var 0.047 vs noise 0.059). Ceiling only turns positive at
σ_u≤0.3 (14% kept, ceiling 0.07) and reaches ~0.4 only at σ_u≤0.2 (n=1,290 —
too few, and corr(σ_u,|u|)=0.65 means tight gates also truncate the signal:
extreme HRs are exactly the low-count sources). Measured R²≈0 + flat IG 1.13
is therefore the *expected* outcome, not a modeling failure. Paths that could
work later: eRASS:4 depth, stacked/binned HR, or the deferred censored
band-rate likelihood (predict P2/P3 rates with upper limits, not their ratio).

**Error-treatment decisions (2026-07-21, post-calibration-check):**
- **σ-conditioning is ruled out** (p(y|x,σ)): σ is metadata of the measurement
  being predicted — unavailable at deployment and it leaks the target
  (σ ∝ 1/√counts ∝ flux). Error info may enter the LOSS, never the conditioning
  set; convolve/inject keep the model σ-free at inference by design.
- σ-stratified predictive coverage (both heads, identical pattern): mid/high-σ
  over-coverage is partly an artifact (the 15-epoch flow barely deconvolves, and
  the widened-interval diagnostic then double-counts noise); the surviving real
  signal is **low-σ under-coverage 0.56±0.02 vs 0.68** — overconfidence on
  precise sources, consistent with an epoch-mismatch AGN-variability floor
  missing from the kernel. Not a blocker for point metrics or model comparisons.
- Queued fixes, in order: residual regression Var(y−p50)=a·σ²+b (free, decisive
  on kernel scale + floor); one learned noise-floor scalar in the kernel (still
  σ-free at inference); reassess after a 50-epoch run (does the latent sharpen?).
- **OPERATING MODE (2026-07-21, supersedes the line below for training): `--error-mode inject` with multi-draw sampling (`--inject-samples 8`)** — per step, k noise draws from each source's split normal, trained point-wise (mean log-prob; AION context shared across draws so k is ~free). Accepts the documented broadening bias (~E[σ²]≈0.03 variance inflation) in exchange for: observation-space simplicity, no kernel at eval, per-source errors still shaping the fit. Convolve remains the optional latent tool; the p(t|x) deployment question is out with a research agent.
- **MODELING DECISION (2026-07-21): the primary estimand is p(y|x)** — plain
  likelihood at observed values, `--error-mode none`, no kernel anywhere. One
  space, fully checkable against held-out data, IG directly comparable to the
  paper's. K is per-datum and *fixed* (never estimated), so the convolution had
  no internal redundancy — but the latent p(t|x) interpretation leans on
  trusting K, and we choose not to. Convolve stays in the repo as the optional
  latent-analysis tool for future intrinsic-scatter work. Canonical flux run of
  the new config: `v1-clean-none-log_ml_flux_1-34131280`.

## 7. Buchner feedback checklist (Slack, 2026-07-21)

1. **Text**: state clearly the counterpart catalog is an optical-to-optical
   crossmatch (their eRASS1–LS10 catalog to ONIR catalogs); naive X-ray-to-
   optical matching is bad due to X-ray positional errors. (We use their NWAY
   counterparts for cleaning — wording must reflect it.)
2. **Validation test**: predict X-ray flux for NON-detected DESI sources —
   predictions should fall below the eROSITA detection limit. Per-position
   limits: Tubín-Arenas+2024 upper-limit database. (Needs a DESI sample
   without eRASS1 detections — new data pull; natural companion to the
   deferred censored band-rate likelihood.)
3. **Text**: more X-ray detail; state the band explicitly (0.2–2.3 keV
   ML_FLUX_1, correct guess).
4. **Table 1**: add a header row "X-ray flux prediction" above R²/IG/exp(IG).
5. **Figure**: rest-frame SED from X-ray to IR per source using predictions +
   z; color-code by X-ray flux, median-bin to denoise. Model-interpretation
   plot.
6. **Cite** Salvato+2022 (eFEDS) as related work: its §3 ML prior (built by
   J. Wolf) predicts P(X-ray emitter) from ONIR photometry (RF on griz+WISE,
   trained on ~40k secure emitters; the NWAY prior). Comparison idea: our
   P(detectable) = ∫ p(flux|x) above the local limit vs their classifier.

**Line-Shapley v2 RESULTS (2026-07-22, drop-mode + aligned grid + wide windows
+ guard 2, job 34298428): the v1 line ranking was an artifact of the masking
bugs — corrected attribution REVERSES it.** Hα is the top line (φ = 68 ± 3
mnats, was 0.2 in v1), then Hβ 41 ± 2, OIII 27 ± 1; NeV 5.5, OII 6.4, HeII
3.2; **MgII slightly NEGATIVE (−4 ± 1 mnats, was v1's top line)**. Lines total
Σφ ≈ 0.147 nats, ~20× v1. Per-token density: OIII 7.9 > Hα 4.5 > Hβ 2.7
mnats/token. Availability-weighted (population view): Hβ 26 > Hα 18 > OIII 14.
**Coalition test (community "lines carry the lion's share" hypothesis): full
−0.980, continuum-only −0.992, lines-only −1.003, norm-only −1.265 nats mean
LL. Of the 0.284-nat shape gain over norm-only, continuum-only retains 96%,
lines-only retains 92% — the information is largely REDUNDANT: either channel
alone nearly suffices; neither holds a lion's share.** Corrected leakage
calibration (same-encoding double difference; norm channel cancels in
marginals): residual beyond-guard leak at guard 2 = 13 (narrow) / 21 (broad)
mnats — small-line φ (NeV/OII/HeII) sits WITHIN this systematic; Hα/Hβ/OIII
are well above it, and the bias direction (line visible when "absent") makes
their φ conservative (biased low). V3 CLS run: OOM at bs 256 in full run
(smoke peak 51 GB but combo-mix + fragmentation exceeded 80 GB) → relaunched
bs 192 + expandable_segments (34327106). Per-band ceilings (sidecar
targets_bands.csv): P1 0.06, P2 0.34, P3 0.23, P4 −1.09; runs 34321151-4.

**2026-07-23 failure post-mortem + recovery:** band runs 34349303-31 died of
plain SLURM TIMEOUT (3h wrap limit; band epochs ≈ 20.5 min → need ~5.5h) —
relaunched identically at 7h as 34629507/08/19/35. v3-cls 34319596/27106/49334
were bs-256/192 OOM iterations; **34356743 trained all 15 epochs (bs 128 +
grad checkpointing) and only its in-process final eval OOM'd.** Root cause
found and fixed (0df4d31): the CLS branch forced `torch.enable_grad()`, which
OVERRIDES eval-time `@torch.no_grad()` → every eval forward built and retained
autograd graphs through the unfrozen encoder (77 GiB accumulated). Encoder now
respects the ambient grad mode. Fresh eval on the saved best.pt: 34632659.

**13-line Shapley results (job 34345374, 1h22m — ran 2026-07-22 evening,
harvested 2026-07-23):** extended catalog (adds Hγ, Hδ, NeIII, Lyα, CIV,
CIII). **Balmer dominance sharpens: Hα 64.0±3.5 > Hβ 36.8 > OIII 27.6 >
Hδ 26.9 > Hγ 18.0 mnats — the four Balmer lines carry ~65% of all line
information.** Then Lyα 15.3±3.7 (11.6% availability, z>1.96), CIII 13.7,
NeIII 10.1, CIV 6.0, OII 5.4, NeV 4.8, HeII ≈0; MgII stays negative
(−4.1±1.2). Per-token density: OIII 8.0 > Hα 4.2 > Hβ 2.4. Population view
(φ·availability): Hβ 23.2 > Hδ 21.7 > Hα 17.4 — Hβ/Hδ overtake Hα on
coverage. Pair interactions: Balmer×Balmer and Balmer×OIII strongly REDUNDANT
(Hβ×Hα −37.7±4.6, OIII×Hα −31.6, Hα×Hδ −27.4 mnats) = one shared
ionization/luminosity signal; only MgII×Lyα weakly synergistic (+17.1±8.0,
2σ). **Coalition summary shifts vs the 7-line v2: lines_only −0.971 now BEATS
full −0.980 (103% of the 0.284-nat shape gain; continuum_only retains 77%,
down from 96%)** — with 13 windows + guards the line set covers enough
spectrum to carry essentially all shape information; the residual continuum
adds nothing (mildly negative marginal, MgII-like). The "redundant channels"
verdict from v2 softens to "line windows suffice; continuum-minus-line-regions
is the weaker channel."

**Band-rate results (reruns 34629507/08/19/35, 2026-07-23; V_simple inject,
gate σ≤1.0):** P1 (0.2–0.6 keV) all-inputs R² **0.320** / exp(IG) 1.15,
**P2 (0.6–2.3) 0.380 / 1.17**, P3 (2.3–5.0) **0.378 / 1.14**,
P4 (5.0–8.0) 0.071 / 0.94. **P1 and P3 land far ABOVE their
predicted noise-floor ceilings (0.06 / 0.23)** → the eRASS1 per-band σ's are
systematically overestimated for faint bands, so the ceiling table is a lower
bound on achievable R², not an upper bound (same error-inflation family as the
HR σ blow-up, opposite consequence). P3's real hard-band predictability
strengthens the censored band-RATE approach over the HR ratio. P4 ≈ its
ceiling (5% detected — genuinely empty).

**PINNED QUESTION (user, 2026-07-23): are we doing a disservice by training
with the catalog lolim/hilim?** The ceiling overshoot implies the eRASS1
LOWERR/UPERR-derived σ's are inflated for faint bands/sources — and inject
training smears targets by exactly these σ's, so we inject MORE noise than the
measurements actually carry. The accepted broadening bias is then larger than
the documented E[σ²]≈0.03, worst where σ is most inflated. Candidate probes
when picked up: (a) the already-queued residual regression
Var(y−p50)=a·σ²+b — a<1 directly MEASURES the σ inflation factor; (b) an
error-mode none vs inject A/B on one band target (P1, most inflated) to price
the damage; (c) a global σ-shrinkage calibration factor in the kernel (still
σ-free at inference). Not scheduled — pinned.

**Why band predictions carry no HR information (variance decomposition,
2026-07-23):** measured logF_P2/logF_P3 are only 0.56 correlated with ~0.40 dex
spread each; the DIFFERENCE (all HR lives there) has 0.374 dex measured spread
of which ~0.35 is independent per-band measurement noise → intrinsic hardness
signal ≈ 0.13 dex, per-source S/N ≈ 0.4 on both target and model side. The
per-band R² 0.38 is almost entirely the SHARED brightness component, which
cancels exactly in HR; our independent band models' residuals are only 0.24
correlated (residual on the diff 0.38 dex ≈ the whole measured diff spread;
corr(predicted diff, true diff) = 0.099). Composed-HR point estimates: corr²
≤ 0.016 on every quality subset (raw R² negative from under-dispersion) — but
the +0.11 corr at n=1,209 is a ~4σ whisper the direct HR model never showed.
The joint P2×P3 flow exists to amplify exactly that (correlated residuals
cancel the shared error in the ratio).

**Multi-target V3b — final spec (user-designed, commits cdc28be + bucketing):**
8 heads = 7 scalar 1-D flows (flux, Lx, M★, P1–P4) + one JOINT 2-D flow over
(P2,P3); HR never a direct target. Per-target CLS vectors; SHARED FULL-RANK
per-block Q/V read adapters (deltas on the projection maps, zero-init; capacity
control = weight decay, not rank — compute is ~free and activation memory is
rank-independent); shared 768→512→256 MLP; sharing ends at the 256-d
conditioning vectors. Losses: per-source availability masks (missing bands
contribute nothing), split-normal injection where σ exists (independent per
band — Poisson), detached-EMA per-head loss normalization. All unit-tested
(60), incl. reader independence (K stacked CLS ≡ K singles).

**Calibration grid (jobs 34706362-89, 2 epochs each, 2026-07-24):**
bs 224: 53.2 samp/s, 27.6 GB, val-sum 9.96 | bs 448: 44.3, 53.3, 10.62 |
bs 896: OOM (frozen attention transient 29 GB + 50 GB checkpoint saves) |
wd 1e-1: 10.41 (≥ wd 1e-2 — stronger adapter decay costs nothing early) |
lr 3e-4 cosine: 10.74, destabilizes the easy heads (Lx 0.82 vs 0.65).
**Decisions: bs 224 (beats 448 on BOTH throughput and per-epoch convergence —
attention is bandwidth-bound at these lengths and more steps win below B_crit),
lr 1e-4 constant, adapter wd 1e-1. Overnight config: train-multi, 30 epochs,
~3.8 h — STAGED, awaiting explicit go.**

**Length-bucketed combo packing (user design, implemented + tested,
`--bucketed`):** combos grouped by padded token LENGTH, not identity — four
buckets (tiny z/W/z+W @ 4 tokens; spectra-ish @ ~281; image-ish @ ~510; heavy
@ ~790), each encoding its combos' modality UNION once with per-source
native-mask dropout; padding waste ≤1.5% everywhere. One forward AND one
optimizer step per bucket: with a large loader batch (~896) each bucket lands
near the calibrated 224 while the step count stays high. Codec batching
improves ~4× per bucket (saving ~1–3% train wall); the big codec win remains
eval-side reuse (15-combo sweep: tokenize once, 15×→1 — queued).

**Composed hardness from band predictions (2026-07-23, user request):** the
eRASS1 flux/rate ratio is EXACTLY constant per band (global ECFs: log F−log R =
−12.133/−12.006 for P2/P3, scatter 0.0000), so HR32 from predicted band fluxes
is exact arithmetic. On the P2∩P3 test intersection (2,169): **the population
HR32 distribution is recovered** — predicted p10–p90 spread (half-width 0.14)
matches the noise-subtracted measured width (~0.16; detected-subset measured
0.29 minus median σ_HR 0.24 in quadrature). **Per-source HR skill is null**
(corr 0.06 on hr32_ok) — independently confirming the direct-HR verdict via a
completely different route. Per-source composed-HR posteriors are wide (±0.55)
because the two band posteriors were modeled independently; a JOINT P2×P3
posterior would cancel the shared flux error in the ratio — a concrete
motivation for the multi-target correlated-flow extension. Slide added
(band-performance table + HR quantile table).

**Shapley v4, merged Hbeta+OIII player (job 34672894):** merge behaves as
predicted — merged φ 48.5±2.2 mnats ≈ v3's separate sum plus their pair
interaction; **Hα robustly top (81±4)**. CAVEAT: the sweep auto-adopted
**guard=10** from the codec-leakage probe (v3 used ~2). ±10 tokens ≈ ±250 Å
dilation explains the anomalies (NeIII availability 83→8% — swallowed by
Hδ+guard; HeII φ ×20 — its guard claims Hβ's blue wing). Ranking trustworthy;
absolute line/continuum split NOT comparable to v3; "lines carry everything"
partly window dilation. Open: pin guard=2 and re-run (~30 min) or audit why
the probe recommends 10.

**V3b read-only CLS (designed with user, commit 8f6b5d8; implemented +
unit-tested, NOT yet run):** CLS outside the sequence — no token attends to
it, data stream bit-identical to the frozen encoder (no_grad); per block the
CLS cross-reads with the block's own frozen attention + frozen MLP; trainable
= CLS vector + per-block zero-init low-rank deltas on the CLS query and
consumed values. Kills both measured V3a failure modes by construction; bs 448
fits; per-block K/V saves cap it near bs ~900. Design points still OPEN with
the user: Q-only vs Q+V adapters, keep/skip the frozen MLP between reads,
per-block vs shared adapters, single vs per-task CLS bank.

**Batching/efficiency design (2026-07-23, user-driven; supersedes any
"maximize VRAM" phrasing):** the objective is information learned per
GPU-hour; per the FASRC TRES scheme (A100 = 836.5 CPU-core-equiv) GPU
wall-time is the entire fairshare cost. Converged design: **combo-pure
forwards** (uniform token count per forward — mixing lengths in one tensor
pays the max length for every source; true packing needs block-diagonal masks
the frozen backbone lacks), **one optimizer step per forward** (same as the
current per-batch-combo protocol), per-combo batch size = **throughput knee,
capped by critical batch size** (below knee: hardware idles; between: 1:1
regime, free choice; above B_crit: redundant data per update). Cheap combos
are optimization-ruled (their forwards are wall-time-free anyway). VRAM
filling is NOT a goal — FLOPs saturation is; empty VRAM = OOM insurance.
Historical runs never varied batch size (all bs 448; V3a only 128), so
knees are being MEASURED: `scripts/throughput_probe.py` (timed steps, no
epochs; job 34686753, manual review — no auto-chaining per user rule).
Infrastructure landed (1e7baf3): `--stratified-combos` per-source assignment
via the native input mask, `oom_resilient_step` (CUDA OOM is catchable →
half-batch gradient accumulation, mathematically identical update),
`vram/peak_gb` + OOM-fallback logging.

**V3-CLS verdict (2026-07-23, eval of 34356743 best.pt): all-inputs R²
0.575 / exp(IG) 1.32 (plain-LL eval) — a statistical TIE with frozen-encoder
V_simple (0.572 / 1.38) on points and WORSE on density.** LoRA(r8, all
blocks/modules) + trainable CLS at ~2.2× the wall time and ~8× the memory of
V_simple buys +0.003 R². Notably single-modality combos DEGRADE (spectra-only
0.476 vs 0.541) while multi-modal catches up — the fine-tune specializes the
encoder toward joint context at the cost of marginal encodings. Read: the
frozen AION representation is not the bottleneck at this sample size; caveats
(15 epochs is short for a fine-tune, one warmup epoch, single LR config) noted
but not compelling enough to iterate further now. V_simple stays the workhorse.

**Deck restructure (2026-07-27).** `docs/slides.pdf` is 21 slides, built by
`docs/build_deck.sh`. Order: framing (title, architecture, Buchner, NWAY,
errors, band coverage) -> validation loss -> the per-target tables in the
paper's format (V_PAI and V_simple first, then the six heads and implied HR)
-> V3b summary -> modality Shapley -> spectral Shapley. Removed: every
text-only slide, the "against the paper architecture" slide, the P2xP3 joint
table (the joint head is only a means to HR, its NLL is not comparable to the
scalar heads), and the bullet annotations on the Shapley slides.
`fig_loss_curves.png` gives each head its own y-axis: on a shared axis all
seven curves collapse into one band. Per-head reading at 30 epochs: P1 never
learns (oscillates in a 0.07-nat band), P2 and P3 are still descending, flux
and Lx are flat from ~epoch 15.

**Defaults aligned with the estimand (2026-07-27).** `--error-mode` had still
defaulted to `convolve` in `main.py` and to `ERROR_MODE=convolve` in both
`sbatch/train.sbatch` and `sbatch/train_smoke.sbatch`, six days after convolve
was demoted to a latent-analysis tool. Every run in the registry passed the mode
explicitly, so no result is affected, but a launch that forgot the variable would
silently have trained through the kernel. All three now default to **`none`**,
the primary estimand p(y|x) and the mode eval already scores in; `inject` (the
adopted training mode, with `--inject-samples 8`) stays an explicit opt-in, which
is right since it needs per-source sigmas that some targets lack (logM* has none).

**SFR added as a 9th head, from CIGALE not FastSpecFit (2026-07-28, user
request: "what we should be predicting is star formation rate").**

*Why SFR is not a relabel of stellar mass.* Measured on the only subset where an
SFR can be derived independently of any stellar-mass fit (BPT star-forming,
Balmer-decrement-corrected Halpha, Kennicutt & Evans 2012, n=536): corr(logM*,
logSFR) = +0.543, so **logM* explains 29.5% of logSFR variance and 70.5% is
orthogonal to mass**. Fitted slope 0.476 (shallower than the canonical main
sequence, expected for X-ray-selected AGN hosts); log sSFR spread 0.906 dex.
Treat as an order of magnitude, not a main-sequence measurement.

*Why not FastSpecFit.* Our `logmstar` IS FastSpecFit's LOGMSTAR: it arrives via
the DESI DR1 `agngal` VAC, whose column description is verbatim FastSpecFit's
("Logarithmic stellar mass (h=1.0, Chabrier+2003 initial mass function)"), and
whose `z` is "Redshift from the FastSpecFit catalog". FastSpecFit's SFR comes out
of the SAME stellar-continuum fit, so it shares that fit's priors and
degeneracies -- an SFR head trained on it could score well by re-predicting mass.
FastSpecFit is also 79 GB (43 GB main-dark + 27 GB main-bright to cover our 94%).

*What we used.* The DR1 **CIGALE** VAC (Siudek+2024, `IronPhysProp_v1.2.fits`,
7.3 GB, 17,149,172 rows), fetched from the public S3 bucket `desidata` (the
documented `data.desi.lbl.gov/public/` root 404s from here; Astro Data Lab's
`desi_dr1` schema hosts `agngal` and `emfit` but NOT fastspecfit). CIGALE is an
independent SED fit, reports `LOGSFR`/`LOGSFR_ERR` already in log space (no
linear->log propagation, no SFR<=0 censoring), and fits an explicit AGN component
(`AGNFRAC`) -- which matters at 87% QSO. 26,506 of our 26,632 match (99.5%).

*Join verified before trusting anything*: CIGALE Z agrees with our z to <1e-3 on
**99.44%** of rows and spectype agrees on 99.88%, so the row match is right and
any disagreement below is astrophysical, not a bad join.

*Cuts (`scripts/make_sfr_sidecar.py`).* Failed fits are written as EXACTLY zero
in both LOGM and LOGSFR **and carry a small LOGSFR_ERR (0.02)**, so an error gate
alone admits them as spurious "log SFR = 0" labels -- 524 of ours. Dropped
explicitly, with the -99 sentinels (25). `FLAG_MASSPDF`/`FLAG_SFRPDF` are NOT
booleans: they are best-fit/Bayesian ratios running to ~1e11, and Siudek+2024
recommend keeping 1/5 < ratio < 5 (drops 5,943). Plus err > 1.0 dex (2,268).
**Survivors: 18,295 of 26,632 (68.7%)** -- GALAXY 1,911/3,462, QSO 16,384/23,044.
log SFR p5/50/95 = 0.88/2.04/3.20, median error 0.235 dex.

*The honest caveat.* Cross-checking the two independent stellar masses on the
surviving rows: corr(CIGALE, FastSpecFit) = **+0.764 for GALAXY but only +0.297
for QSO** (scatter 0.403 vs 0.857 dex). Against the independent Halpha SFR,
CIGALE gives corr +0.396 (+0.487 at AGNFRAC<0.3) with ~1-2 dex scatter; against
PROVABGS (n=114) corr +0.428. So the SFR label is sound for the galaxy minority
and weak for the QSO majority. It is a masked head, `cigale_spectype` and
`cigale_agnfrac` ride along in the sidecar, and results MUST be reported
per-spectype rather than as a single number.

*Guard against the degeneracy.* `scripts/eval_multitarget.py` now scores the SFR
head against two mass-only baselines on the same test rows -- the main sequence
fitted on train and applied through (a) the TRUE logmstar, the ceiling for any
mass-only predictor, and (b) the model's PREDICTED logmstar. The head must beat
both, or it is re-predicting stellar mass. Written to `sfr_vs_mass_baseline.csv`.

*Plumbing.* `log_sfr` is APPENDED to `MULTI_TARGETS` (never inserted -- the flows
are positionally indexed, so inserting renumbers every checkpoint). Sidecar
columns merge into `targets_sidecar.csv` (26,632 rows, 22 cols, 9.0 MB), which
replaces `targets_bands.csv` as `--extra-targets-csv`; the loader needed no
change. Old checkpoints stay loadable via the new
`configure_heads_from_config()`, which derives the head set from the
checkpoint's stored `heads` list rather than `drop_heads` alone. 65 tests green.

*Rejected: PROVABGS.* Held locally with full 100-sample MCMC posteriors, but it
overlaps our sample by only 114 sources (0.43%) -- it is BGS (bright, low-z) and
we are QSO-dominated at median z=0.86. Kept as a calibration set only.

**V4 with the SFR head: run `mt-v4-sfr-35828655` (2026-07-28/29).** 40 epochs,
5 h 30 m on one A100, best epoch 34, 28.4 GB, `sbatch/train_multi.sbatch`.
Eval `36028553` via the new `sbatch/eval_multi.sbatch`.

Test set, all inputs (n = 2,520; SFR 1,714; HR 2,169):

| head | R2 | IG (nats) | lines-only R2 |
|---|---|---|---|
| log Lx | 0.921 | 1.201 | 0.689 |
| **log SFR** | **0.815** | **0.939** | 0.429 |
| log M* | 0.765 | 1.353 | — |
| log flux | 0.614 | 0.317 | 0.482 |
| P3 | 0.418 | 0.182 | — |
| P2 | 0.390 | 0.215 | — |
| P1 | 0.366 | 0.150 | — |
| HR32 implied | 0.015 | 0.084 (0.108 on hr32_ok) | — |

**The SFR head is not re-predicting stellar mass.** The main sequence fitted on
train is nearly flat in this sample (logSFR = 0.076*logM* + 1.233), so a mass-only
predictor reaches R2 = 0.001 with the TRUE mass and 0.008 with the model's
predicted mass, against the head's 0.815. Verdict recorded in
`eval/sfr_vs_mass_baseline.csv`. Note this flatness is specific to an X-ray
selected, 87% QSO sample -- on the BPT star-forming subset with an independent
Halpha SFR, log M* explains ~30% of log SFR variance.

**Adding the ninth head was free.** Best-epoch validation NLL against V3
(35416432) on the seven shared heads: flux 0.975 vs 0.974, Lx 0.169 vs 0.166,
M* 0.006 vs 0.018, P1 1.247 vs 1.244, P2 1.159 vs 1.158, P3 1.203 vs 1.202,
joint 2.309 vs 2.300 -- summed +0.006 nats, indistinguishable. Epoch time +1.8%,
GPU memory unchanged.

**Trajectories say the run is far too long for every head except SFR.** 90% of
all validation improvement arrives by epoch 10, 94% by epoch 20, while the total
train-probe/validation gap grows +0.264 -> +1.301 nats. Validation gain AFTER
epoch 10: log SFR +0.212, log M* +0.163, everything X-ray <= +0.022 (P3 exactly
0.000). log SFR also has the second-smallest gap (+0.085) and was still
descending at epoch 40 -- it is the only head not yet saturated. Heads peak at
wildly different epochs (P1 5, P3 10, P2 22, joint 23, flux 26, Lx 29, M* 36,
SFR 39), so the single global early stop at 34 is wrong for nearly all of them:
**per-head checkpoint selection would gain 0.144 nats for free, no retraining.**
This turns the deferred per-head LR-schedule item into a concrete one.

**Modality UpSet, first real read.** log flux is spectra-dominated (spectra alone
0.285 of 0.317 all-in) and z adds nothing on top of it (spectra+z 0.286). log Lx
is the opposite: z alone reaches 0.94 of 1.20 because Lx is mostly distance, and
spectra+z is heavily redundant since the spectrum carries z. **log SFR is the
only target with no dominant single modality** -- spectra 0.59, image 0.29, z
0.25, WISE 0.22 of 0.94 all-in -- and its combinations stack far more additively.
That partly answers the label-leakage worry: CIGALE fits SFR from grz+WISE
photometry, so if the head were merely reproducing that fit, WISE and image
should dominate. They do not; the spectrum does.

**Deck:** `docs/build_deck.sh` now regenerates every narrative figure, and
`make_results_figure.py` reads HR straight from `hr_implied_target.csv` instead
of the old hand-transcribed `--hr-r2/--hr-ig` flags.

**Target set reworked: CIGALE for mass and SFR, catalogued M_BH, sSFR implied
(2026-07-29).** `scripts/make_sfr_sidecar.py` is now
`scripts/make_targets_sidecar.py` and builds all of it.

*Stellar mass moves to CIGALE.* Our `logmstar` was FastSpecFit's, inherited from
the paper rather than chosen. **FastSpecFit has no AGN component** -- it fits a
stellar continuum plus emission lines and is built for galaxies -- so on a sample
that is 87% QSO the accretion-disk continuum is attributed to stars, biasing the
mass in a way the fit never models. CIGALE fits the AGN explicitly (that is where
AGNFRAC comes from). The two masses agree at corr **+0.76 for GALAXY but only
+0.30 for QSO**, which is what that bias looks like. CIGALE also publishes
`LOGM_ERR`, replacing the fabricated 0.2/0.3 dex spectype floor. Coverage 20,563
(77.2%). The FastSpecFit head is kept in the code and dropped by default
(`DROP_HEADS="log_flux_p4 logmstar"`), so the comparison stays available.

*So mass and SFR now come from ONE fit*, which is what makes sSFR well defined.
This was the user's question and it had no good answer before: SFR was chosen
from CIGALE deliberately, mass was FastSpecFit's by inheritance.

*sSFR is NOT trained.* It is an exact function of log SFR and log M*, so a head
on it adds no label information -- only a different projection of the same data,
and one whose posterior could contradict their difference. It will be implied
from an (M*, SFR) joint by the same shear-marginalisation already validated for
HR, which yields the error correlation rather than assuming it. Carried in the
sidecar as `ref_log_ssfr` purely to validate that implied version later. Note
sigma(sSFR) genuinely needs Cov(LOGM, LOGSFR), which CIGALE does not publish, and
the sign is not knowable a priori -- overall normalisation correlates the two,
the young/old split anti-correlates them.

*Black-hole mass, catalogued.* The DR1 **qmassiron** VAC
(`VAC_BHmass_338_v1.7.fits`, **109 MB**, 490,648 quasars at z<1.6, MgII) -- not
the 70 GB FastSpecFit pull that was assumed necessary. There is no BH mass in
`agnqso`, which is a 36-column classification summary. Two heads:
**`log_mbh_pan25`** (primary, iron-corrected: FeII blends with MgII and inflates
FWHM, and M_BH goes as FWHM^2) and **`log_mbh_vo09`** (the classic estimator).
Coverage 9,813 / 9,806 (36.8%) after an err<1 dex gate, the lowest of any head.

The other three calibrations ride as sidecar columns, not heads: VO09, SHEN11,
LE20 and YU23 intercorrelate at **0.99+**, and LE20 vs VO09 is exactly **1.0000**
-- a pure rescaling. Only PAN25 differs (r ~ 0.91), i.e. the iron correction is
the only thing that changes the answer. Median spread across all five is **0.375
dex** against a median published error of 0.330, so the choice of calibration
matters about as much as the measurement does.

*Head set:* 9 scalar + the P2xP3 joint = 10, with `log_flux_p4` and `logmstar`
dropped by default. Sidecar `targets_sidecar.csv` is 39 columns / 12 MB.

**CORRECTION: the "flat main sequence" was a cross-catalogue artifact
(2026-07-29).** The 2026-07-28 SFR entry above records the star-forming main
sequence as nearly flat on this sample — `logSFR = 0.076*logM* + 1.233`, R² 0.001
— and attributes it to AGN-compromised CIGALE fits. **That is wrong.** It
compared **FastSpecFit's** `logmstar` against **CIGALE's** `LOGSFR`: two
different SED fits. With both taken from the same CIGALE fit the relation is
normal: **slope +0.745, corr +0.562, R² 0.316, scatter 0.588 dex** on 17,294
sources, inside the literature 0.7–1.0 range.

It also resolves a disagreement that should have been caught earlier: the
independent Halpha-based measurement (R² 0.295 on 536 BPT star-forming sources)
never matched the 0.007 cross-catalogue number, and it was the cross-catalogue
number that was broken.

Consequences:
- **`eval/sfr_vs_mass_baseline.csv` from run 35828655 is inflated.** It reported
  the SFR head at R² 0.815 against a mass-only baseline of 0.008, using
  FastSpecFit mass. Against `logmstar_cigale` a mass-only predictor reaches
  ~0.32, so the margin is ~0.50, not ~0.81. The verdict (the head is measuring
  SFR, not mass) survives; the number does not. The check must be recomputed
  against the CIGALE mass.
- The earlier estimate of R²(sSFR) ~ 0.79 assumed zero covariance and is void.
  Measured sd(log sSFR) = 0.604 dex, NARROWER than sd(log SFR) = 0.711, because
  the correlation suppresses it.
- This is the strongest argument for the CIGALE switch: mixing fits did not just
  make sSFR ill-defined, it destroyed a real physical relation.

## 8. Objective: per-head overfitting control on a shared body (2026-08-03)

**The objective, stated once.** Recover each head's own best stopping point
WITHOUT giving up the property the whole project rests on: **one body, one
checkpoint, one forward pass, all targets.** Any fix that ends in N bodies has
quietly rebuilt the specialized baselines we claim to beat.

**The evidence** (run v4, 40 epochs, `docs/figures/v4_epoch_history.json`, drawn
in `fig_overfit.png`). Per-head best validation epochs span **5 to 39**: P1 peaks
at 5 and then trains 35 further epochs; log SFR is still improving at 39. The
single kept checkpoint is epoch 34. Scoring every head there instead of at its
own optimum costs **0.1451 nats** total, worst for log M* (0.031) and the P2xP3
joint (0.027). Train-probe minus validation confirms it is real overfitting and
not scale: the gap widens with epoch for 7 of 8 heads (log M* 0.389 -> 0.486).

*Read the loss curves as (val - own best).* The default per-panel view uses
independent y-scales, so a head falling 1.6 nats looks smooth while one that
barely moves looks violently noisy, with identical epoch-to-epoch scatter. That
appearance is a plotting artifact, not a property of the heads.

**Two failure modes, which need different fixes.**
- **(A) A converged head keeps carving the shared trunk.** Its gradient goes on
  tuning CLS tokens, read adapters and the shared MLP to its own idiosyncrasies
  past the point where that helps it. This damages *every* head.
- **(B) A converged head is dragged by the others.** Even a frozen flow degrades
  because the body underneath it keeps moving.

Loss down-weighting fixes (A) only. Per-head checkpoints would fix (B) but need
the body stored and re-run per head, which is exactly the outcome ruled out
above. Hence:

**Decision: two phases, one body.**
1. **Joint phase.** Train body + all heads. A head whose val NLL fails to improve
   by more than `delta` for `patience` epochs has its loss weight decayed
   (multiplied by `gamma`, floored at 0). This exists ONLY to stop it from
   dragging the trunk, so it can be aggressive. Keep one body checkpoint chosen
   by the global summed-val criterion, which is the right criterion for the one
   thing that is genuinely shared.
2. **Refit phase.** Freeze that body. Cache CLS embeddings for train and val
   once. Refit each flow independently on the cached embeddings, each with its
   own early stopping. Cheap: the flows are small and the expensive frozen-AION
   forward happens once, not once per epoch per head.

This yields exact per-head early stopping, fixes (B) as well (the body does not
move in phase 2), leaves inference at one forward pass, and makes phase-1
down-weighting safe to push to zero since every flow is refit afterwards anyway.
It is also the same move the project already makes one level up (freeze AION,
fit flows on its embeddings). Standard practice: frozen-backbone per-task head
refit; plateau down-weighting is in the GradNorm / uncertainty-weighting family.

**Success criterion:** recover most of the 0.1451 nats with a single body, and
verify no head is worse than it was under the shared checkpoint.

**Not decided yet:** `delta`, `patience`, `gamma`; whether a down-weighted head
may be re-promoted if val improves again; and whether phase 2 should re-tune
flow width or regularization per head, or only the stopping epoch.

## 9. Phase 1 run plan: the joint head alone (2026-08-03)

**Shape of the experiment.** Train ONE head, the 4-D joint over
(logmstar_cigale, log_sfr, log_lx, log_flux_p3), and nothing else. Then freeze
the body and fit every marginal plus the P2xP3 joint on the frozen
representation. Phase 1 has a single loss, so there is no weighting problem to
solve: the whole loss-weight / per-head-LR / plateau-detection question is
answered by not having more than one head.

**Data.** 17,294 of 25,200 clean rows (68.6%) carry M*, SFR and Lx. Of those,
1,364 lack P3 and are marginalised by the 48-node quadrature. The remaining
7,906 rows (31.4%) do not enter phase 1 at all, and they are not a random
sample: they are the sources CIGALE failed to fit. **If the joint looks
data-starved, this is the first suspect**, and the fix is to generalise the
quadrature so a missing M* or SFR is integrated out per row exactly as P3 is
(rows missing one dimension cost the same 48 nodes we already pay).

**Trainable parts.** CLS token; per-block Q/V read adapters; shared MLP
768->512->256; the 4-D NSF flow. All of AION stays frozen.

### Setting the learning rates by measurement, not by guess
`group_metrics` already logs `move/<group>` = |w_t - w_{t-k}| / |w_t|, the
update-to-weight ratio, for cls_tokens, adapters_low/mid/high (split by encoder
depth), shared_mlp and each flow. A healthy group sits near 1e-3 per step;
groups an order of magnitude off are the ones that need their own LR. This is
the evidence that produced the current split in the first place: zero-init
adapters (|w| ~3) moved ~30x faster than standard-init flows (|w| ~40) on a
shared LR, and left the flows near their initialisation for an entire run.

Procedure: run ~300 steps, read `move/` per group, and set each group's LR to
bring them into one band. Measure AFTER warmup, since `move` is meaningless
while |w| is still ~0 for the zero-init adapters. The adapter depth split
exists so that "do deep and shallow adapters want different LRs" is answered
with data rather than assumed.

Starting point (v4 values, known to train): LR 3e-4 (MLP + CLS),
ADAPTER_LR 1e-4, ADAPTER_WD 1e-1, WEIGHT_DECAY 1e-4, batch 896. HEAD_LR is the
knob expected to move first: those defaults were tuned for 1-D flows and the
joint is a 4-D density.

### Three loop changes for this run (corrected 2026-08-03)
1. **LR warmup.** Missing from `train-multi`. (It exists only in the
   single-target path in main.py, and there it is an epoch-level adapter freeze,
   not a step-level ramp.) Added as `--warmup-steps`, composed with the cosine
   in ONE `LambdaLR`: LambdaLR scales each group's own initial LR, so the
   adapter/flow/trunk ratios survive the schedule, which chaining two schedulers
   would not guarantee.
2. **Gradient clipping ALREADY EXISTED**, hardcoded at 5.0. An earlier note here
   claimed it was absent; that was wrong. Now `--grad-clip`, default 5.0 to
   preserve behaviour, worth tightening if the quadrature logsumexp spikes.
3. **`--lr-schedule` is hardcoded to `constant` in the sbatch** (not merely
   defaulted). Now driven by `$LR_SCHEDULE`, cosine for this run.

**Also pin the EMA to 1.** With a single head, `weight = 1/EMA` becomes a
time-varying global scale on the only loss, which is silently a second
learning-rate schedule on top of the cosine.

### Overfitting
v4's common-mode gap grew 0.016 -> 0.121 over 40 epochs, and phase 1 has FEWER
rows (17,294), so the risk is higher, not lower. Controls in place: weight decay
1e-4, adapter weight decay 1e-1, error injection acting as augmentation, early
stopping on the single joint val NLL.

The useful realisation: **phase 1 is judged on the BODY, not on the flow**,
because the flow is refit in phase 2 regardless. So snapshot the body every ~4
epochs and let phase 2 choose among snapshots by post-refit validation. That is
the correct criterion and it is not the same epoch as min joint val NLL.

### Go / no-go gates
- **Before GPU:** unit tests green; a 2-batch forward/backward shows a non-zero
  `gnorm/` for every trainable group (a zero means a detached path) and
  `requires_grad=False` everywhere in AION.
- **Smoke (gpu_test, 2 epochs, limited):** checkpoint writes, diagnostics
  appear, val NLL finite, quadrature rows produce gradient.
- **LR probe (~300 steps x 3 settings):** all `move/` groups within one order of
  magnitude; no group at ~0 (frozen) or >1e-1 (diverging).
- **Abort the full run if:** any val NLL is NaN, or the joint has not beaten its
  KDE prior by epoch 3.
- **Primary success metric:** joint val NLL versus the SUM of the phase-2
  marginal NLLs on the same rows. The difference is the dependence the joint
  captures, which is the entire reason the head exists.
