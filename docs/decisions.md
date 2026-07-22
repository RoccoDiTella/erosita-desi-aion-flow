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
| `logmstar` | LS photometric stellar mass (VAC) | DESI VAC `logmstar` |

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
  treatment deferred (ref: Sacchi+2022, 2022A&A...661A...3S).

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
