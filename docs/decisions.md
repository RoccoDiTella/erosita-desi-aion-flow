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
`IAUName`, compare NWAY's chosen LS10 counterpart with ours per `targetid`.
Classes in `match_quality.csv` (30,441 targetids):

| class | n | meaning | kept? |
|---|---|---|---|
| `correct` | 26,632 (87.5%) | our counterpart = NWAY's secure counterpart | ✅ |
| `spurious` | 1,514 | NWAY flags the X-ray detection itself as likely spurious | ❌ |
| `ambiguous` | 1,033 | NWAY has no single secure counterpart | ❌ |
| `wrong` | 1,017 | NWAY picks a *different* optical object | ❌ |
| `not_in_NWAY` | 245 | X-ray source absent from the NWAY DR1 sample | ❌ |

**`keep = (class == correct)` only** — conservative by design.

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
  (documented as broadening, not deconvolution).
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
| `v1-clean-log_ml_flux_1-34083239` | V1, convolve | clean view (25,200) | 15 | *pending* | *pending* | running |
| `paperhead-clean-log_ml_flux_1-34089921` | **q4/l2**, convolve | clean view | 15 | *pending* | *pending* | queued |
| paper reproduction | q4/l2, none | noisy, paper split | 50 | *deferred* | | planned |
| other targets (log_lx, logmstar, hr32_u) | V1, convolve | clean view | 15 | | | deferred (first runs cancelled) |

**Queue decision (2026-07-21):** compare **V1 vs paper head, everything else
identical** (clean view, convolve, flux, 15 epochs) — the head A/B informs what
to try next. Per-target runs resume after. Every completed run gets the standard
**evaluation packet** (`scripts/make_run_packet.py`, ported from the RunPod
packet-v5): diagnostics PDF (scatter grid + IG histograms + calibration),
upset-style combo figure, per-spectype and per-redshift slices.

Comparison caveats: the cleaned-view test set (2,520 cleaned rows) ≠ the paper
test set (3,054 noisy rows) — deltas vs the paper are indicative, not
row-comparable (the provenance report quantifies overlap). HR reports
IG as primary (R² is not meaningful near the clip).
