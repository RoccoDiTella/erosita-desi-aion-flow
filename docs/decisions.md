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
