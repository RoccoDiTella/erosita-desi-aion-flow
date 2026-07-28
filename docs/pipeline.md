# eROSITA × DESI × AION-flow — Experiment Pipeline

**Living document — keep in sync with the actual pipeline. Last updated: 2026-07-17.**
Render to PDF with `python docs/render_pipeline_pdf.py` (writes `docs/pipeline.pdf`).

## 1. Goal
Use a frozen astronomical foundation model (AION-1 base) as an information probe: predict eROSITA X-ray properties (broad-band flux, luminosity, hardness) and host stellar mass from non-X-ray DESI modalities (optical spectra, WISE, redshift, Legacy imaging) with a probabilistic normalizing-flow head. Report likelihood gain over a prior, not just point R². This repo extends the PAI26 paper with (a) a cleaned crossmatch and (b) uncertainty-aware, per-target training.

## 2. Compute
- **FASRC** (Harvard SLURM). Account **`finkbeiner_lab`** (confirmed working 2026-07-21); smokes on `gpu_test`, real runs on `gpu`. `siag_gpu` is *not* visible to us yet — it needs `siag_lab` group membership (Coldfront request pending).
- Env: `module load Miniforge3/26.1.0-fasrc01` + conda env + **cu124 torch wheel** + `pip install -e . polymathic-aion zuko safetensors wandb`. Built at `$AIONFLOW_ROOT/envs/aionflow` (torch 2.6.0+cu124).
- **Workflow:** develop locally → git push → `git pull` on the cluster; the SSH ControlMaster socket `~/.ssh/cm-fasrc` is used only to submit jobs and move data. All real work runs in `sbatch` jobs on compute nodes (login-node compute gets killed by arbiter2).
- **Tracking: Weights & Biases** (`--wandb`, project `erosita-desi-aion-flow`). Logs config, per-step batch NLL, per-epoch train/val NLL, and the final per-combination test metrics table. Auto-falls back to `offline` if a compute node lacks outbound internet (`wandb sync` later); a tracking failure never kills a job.

## 3. Data sources & download
| Modality | Source | Notes |
|---|---|---|
| X-ray catalog | SRG/eROSITA-DE **DR1 / eRASS1** (Merloni+2024) | broad band `ML_FLUX_1` + sub-bands P1–P4 rates/fluxes **with asymmetric errors** (LOWERR/UPERR). Delivered as the eROSITA×DESI match CSVs. |
| Optical spectra | **DESI DR1** coadd spectra | per-healpix `coadd-{survey}-{program}-{healpix}.fits`, trimmed to matched targetids, compiled to `erosita_spectra_merged_32k.hdf5`. |
| Imaging | **Legacy Survey DR10** griz cutouts | 160×160 px `fits_pool/<targetid>.fits` via legacysurvey.org cutout service (~11 GB). |
| Redshift / galaxy props | DESI DR1 VAC | z, `logmstar` (photometric), emission lines, `spectype` (QSO/GALAXY/STAR). |
| Foundation model | `polymathic-ai/aion-base` (HuggingFace, ~95 MB) | frozen tokenizer + backbone. |

All inputs are local; only `aion-base` is internet-bound. "Adding sources our match missed" would require new DESI DR1 spectrum downloads (deferred).

## 4. Crossmatch & cleaning
- **Original match (existing):** naive **5″ nearest-neighbor** (`crossmatch_radec.py`, astropy) between DESI target RA/Dec and the eROSITA catalog → 33,938 pairs → 30,441 unique DESI targets → **28,646 with a retrieved spectrum**.
- **Quality problems found:** separations pile up near the 5″ cut (median 3.08″, 52% >3″); **14% of eROSITA sources match >1 DESI target** (26% of pairs in multi-match groups) → the same X-ray label duplicated onto several spectra.
- **Cleaning (NWAY):** validate against **Salvato et al. 2025 (A&A 704, A344)** — the eRASS1 Bayesian NWAY counterpart catalog — via VizieR `J/A+A/704/A344/dr1ls10` (join on `IAUName`; compare their counterpart's `zsp` to our matched DESI `z`). Per-`targetid` class: **87.5% correct, 4.6% spurious, 3.3% wrong, 3.4% ambiguous, 0.8% absent** → `keep = correct` filter (`data/match_quality.csv`).
- **Sanity checks (cleaning is sound, 3 independent angles):** (a) `|z_ours − z_NWAY|` is **bimodal** — 88.7% <1e-4 (same object), 3.0% >0.1 (different), only 0.6% in the 0.01–0.1 valley → the z-cut is natural, not tuned; (b) mismatches show larger match separation (3.19″ vs 3.06″) but only weakly — separation can't discriminate at eROSITA's ~arcsec position errors, which is *why* prior-based NWAY is needed; (c) the trained model is **2× worse on mismatches** (R² 0.29 vs 0.57) — independent confirmation the flags localize real failures.
- **Pipeline order (IMPORTANT):** clean → **dedup to the NWAY counterpart per targetid** → **re-derive the train/val/test split on the cleaned sample** (targetid-grouped 80/10/10, seed 42, leakage-safe) → stage. Splits must be redone *after* cleaning.

<img src="figures/fig_cleanup.png" style="width:18cm" />

## 5. Targets & uncertainties
Four separate 1-D flow runs (`data/targets_extra.csv` carries the derived targets + errors, keyed by `targetid`):

| Target | Source | Error model |
|---|---|---|
| `log_ml_flux_1` | `log10(ML_FLUX_1)` | **split-normal** (two-piece Gaussian) from LOWERR/UPERR: `sig_hi=log10(1+UPERR/F)`, `sig_lo=-log10(1-LOWERR/F)` (capped). Median ≈0.17 dex; **98% asymmetric, low side heavier**. |
| `log_lx` | `ML_FLUX_1` + z (Planck18) | same dex σ as flux. |
| `logmstar` | DESI VAC photometric mass | **spectype systematic floor**: 0.20 dex (GALAXY) / 0.30 dex (QSO). No per-source posterior σ available (PROVABGS covers only 0.4% of this AGN sample). Caveat: 82% of sample is QSO → mass AGN-contaminated; evaluate split by `spectype`. |
| `hr32` (hardness) | `(P3−P2)/(P3+P2)` band rates | modeled in **arctanh space** `u=arctanh(HR)`, `sig_u=sig_hr/(1−HR²)`. Only ~17% have S/N>2 (σ≈signal) → **IG-primary**, R² secondary. HR43/Γ dropped. |

<img src="figures/fig_targets.png" style="width:17.5cm" />

**How errors enter the flow (split-normal convolution likelihood):**
`NLL_i = −log ∫ p_flow(t | x_i) · K(y_i | t; sig_lo_i, sig_hi_i) dt`, a fixed 1-D quadrature (~21–41 nodes over ±4σ), `error_mode ∈ {none, inject, convolve}` (default **convolve** = deconvolves; `inject` broadens; `none` = paper behavior).

<img src="figures/fig_errors.png" style="width:16.5cm" />

Measurement error is large relative to signal: for flux, σ≈0.17 dex is ~2/3 of the model's residual variance → the model sits near the measurement-noise floor, so error-aware training + a reported error floor matter.

**Deferred error refinement:** X-ray sub-band non-detections are **upper limits (censored)**, not missing/large-error; proper handling via a censored likelihood (ref 2022A&A...661A...3S). First pass uses split-normal σ on measured values.

## 6. Model architecture
<img src="figures/fig_architecture.png" style="width:17.5cm" />

Frozen **AION-base** tokenizer + backbone → per-modality token sequences (`spectra`=273, `image`=576, `WISE`=3, `z`=1; missing modalities omitted). Trainable head + flow:

- **V1 minimal head** (`AIONAttentionContext`, parameterized): per-modality affine calibration → **1 learned query + 1 attention block (8 heads)** cross/self-attention → 768 → MLP (→128) → 256-d context. (~7.8M params vs the paper's 4-query/3072-d ~16.8M.)
- **Flow:** Zuko conditional **NSF**, 1-D target, 8 transforms, context 256; standardized target; **KDE (Scott) prior** as the IG reference.
- Per-batch a modality combo is sampled; AION frozen; AdamW; checkpoint best all-inputs val NLL.
- **V2 (later):** replace frozen AION with a self-trained attention encoder over the raw spectrum.

**Original paper result** (q4/l2, epoch-13) — R² by modality combination; full spectra beat the emission-line baseline, all-modality is best:
<img src="figures/fig_results.png" style="width:16.5cm" />

## 7. Training & evaluation
- Four separate V1 runs (`--target {log_ml_flux_1,log_lx,logmstar,hr32}`), `--clean` (NWAY keep), `--error-mode convolve`, on `gpu`; smoke on `gpu_test` with a `--limit` subset.
- **Metrics:** primary **IG / exp(IG)** vs KDE prior (esp. HR); secondary **R²** + **reduced-χ²** using per-source σ + the measurement-error floor; **clean-vs-full delta**; per-`spectype` split for logmstar/HR.
- Every run is mirrored to **W&B** (`--wandb`), so training curves are watchable live and runs are comparable across targets/error-modes via the run tags (`<target>`, `<error_mode>`, `V1`).

## 8. Key findings so far
- Naive 5″ match is **~8% wrong** per NWAY; the paper's q4/l2 model is **~2× worse on the mismatches (R² 0.29 vs 0.57)**; dropping them lifts test R² from the published **0.549 → 0.567** (no retraining).
- X-ray targets are measurement-noise-dominated (flux σ≈0.17 dex; HR σ≈its own spread) → uncertainty handling is essential.
- **Hardness is not learnable at eRASS1 depth**: Var(u) < E[σ_u²] at every S/N gate, including both-detected. It survives only as an *implied* target, marginalized from a joint (P2,P3) flow.
- **The frozen AION representation is not the bottleneck**: a LoRA fine-tune (V3a) ties with the frozen encoder; a read-only CLS that leaves the data stream untouched (V3b) is what wins.
- Faint-band σ's in the eRASS1 catalog are systematically **overestimated** — P1 and P3 land far above their σ-derived R² ceilings, so those ceilings are lower bounds.

## 9. Status
Current best is the V3 multi-target run (job 35416432, 30 ep / 4 h): one frozen-encoder
pass, seven heads, test all-inputs **flux 0.604 · Lx 0.919 · logM★ 0.744 · P1 0.364 ·
P2 0.404 · P3 0.406**, plus a hardness posterior marginalized from the joint head. It
beats every specialized single-target baseline it replaces. The binding constraint is now
**overfitting** (train-probe gap 0.157 nats), not learning rate or capacity.

Per-run detail, the full registry, and the reasoning behind each choice live in
[`decisions.md`](decisions.md) — that is the source of truth for results. Presentation
artifacts: `bash docs/build_deck.sh <eval-dir> <vpai-csv> <vsimple-csv> [histories]`.

Open: harvest the 12-bin Shapley sweep; per-head LR schedules (convergence is very uneven —
P1 never learns, P2/P3 still descending at 30 epochs); the σ-inflation probes; eval-side
codec reuse; censored band-rate likelihood; the Buchner non-detection test.

## 10. Deferred / out of scope
Γ (=HR relabel), X-ray upper-limit censoring, full Salvato re-crossmatch of "the rest",
V2 self-trained encoder, 50-epoch paper reproduction. (**SFR landed 2026-07-28** as a
9th head from the DR1 CIGALE VAC — see `decisions.md`. Still open there: the joint
(logM★, logSFR) flow that would give sSFR as an implied target by the same exact
marginalization used for HR.)
