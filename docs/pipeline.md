# eROSITA × DESI × AION-flow — Experiment Pipeline

**Living document. Last updated: 2026-08-06 (rewritten against DR2 / eRASS:3).**
Render to PDF with `python docs/render_pipeline_pdf.py` (writes `docs/pipeline.pdf`).

Numbers marked *measured* were recomputed on this machine on 2026-08-06.
Where a figure below predates the DR2 relabel, it says so.

**Every sample count on this page describes the artifacts as built on
2026-08-04** — `targets_sidecar_dr2.csv` and `clean_split_dr2.csv`, 25,582 rows
each. The rebuild in §10 emits a different, smaller row set (projected 25,454
targets, ~22,800 split rows) and nothing in a filename distinguishes the two.
`docs/DATA.md` §0 is the as-built register; check it before quoting a number
from here.

## 1. Goal
Use a frozen astronomical foundation model (AION-1 base) as an information probe:
predict eROSITA X-ray properties (broad-band flux, luminosity, band fluxes,
hardness) and host properties (stellar mass, SFR) from non-X-ray DESI modalities
(optical spectra, WISE, redshift, Legacy imaging) with a probabilistic
normalizing-flow head. Report likelihood gain over a prior, not just point R².
This repo extends the PAI26 paper with (a) a cleaned crossmatch, (b) per-target
uncertainty-aware training, and (c) a joint head whose posterior *covariance*
carries quantities no marginal can express.

## 2. Compute
- **FASRC** (Harvard SLURM). Account **`finkbeiner_lab`**, real runs on **`gpu`**
  (A100-**80 GB**), smokes on `gpu_test`. Batch sizes are calibrated on 80 GB
  cards, which is what we are on, so no retuning is needed.
  *(This section briefly asserted `siag_lab` / `siag_gpu` on 2026-08-06. That
  was wrong and is reverted. Verified against the scheduler the same day:
  `sbatch --test-only -A siag_lab -p siag_gpu` returns "Invalid account or
  account/partition combination", while `-A finkbeiner_lab -p gpu` schedules.
  What misled us is that `siag_gpu` DOES now appear in `sinfo`. Partition
  visibility is not a membership test; only a dry-run submit is.)*
- Fairshare is charged almost entirely on GPU wall-time: TRES weight A100 =
  836.5, H200 = 2651.5, one CPU core = 1.0. An H200 only pays for itself above a
  3.2× speedup, which these batch-bound jobs do not reach — hence
  `posterior_structure.sbatch` moving off `gpu_h200`.
- Env: `module load Miniforge3/26.1.0-fasrc01` + conda env + cu124 torch wheel +
  `pip install -e . polymathic-aion zuko safetensors wandb`, at
  `$AIONFLOW_ROOT/envs/aionflow` (torch 2.6.0+cu124).
- **Workflow:** develop locally → git push → `git pull` on the cluster; the SSH
  ControlMaster socket `~/.ssh/cm-fasrc` is used only to submit and to move data.
  All real work runs in `sbatch` jobs (login-node compute gets killed by arbiter2).
- **Every launcher names its dataset.** `sbatch/_dataset.sh` requires
  `DATASET=dr2`, resolves the split/sidecar/staged trio, and gates on the
  sidecar's columns *and* on `median(flux_sig_lo) ∈ (0.10, 0.14)`. Before this,
  three launchers defaulted to DR1 and one defaulted to DR2, which made
  analysing a DR1 checkpoint against DR2 labels a silent, one-flag mistake.
- **Tracking: Weights & Biases** (`--wandb`, project `erosita-desi-aion-flow`).
  Falls back to offline automatically; a tracking failure never kills a job.

## 3. Data sources
| Modality | Source | Notes |
|---|---|---|
| X-ray catalogue | SRG/eROSITA-DE **DR2 / eRASS:3** | `eRASS3_Main_v1.3.fits` + `eRASSc3_Main_LS10.fits` (Main × Legacy DR10 with NWAY probabilities), public release 27 Jul 2026. Broad band `ML_FLUX_1` + sub-bands P1–P4 with asymmetric errors, per-band `DET_LIKE`, `ML_EXP_1`. |
| Optical spectra | **DESI DR1** coadd spectra | per-healpix `coadd-{survey}-{program}-{healpix}.fits`, trimmed to matched targetids. |
| Imaging | **Legacy Survey DR10** griz cutouts | 160×160 px at 0.262″/px. |
| Redshift / host props | DESI DR1 VACs | z, `spectype`; **CIGALE** (`IronPhysProp_v1.2.fits`) for M*, SFR, AGN fraction; `VAC_BHmass_338_v1.7.fits` for M_BH. |
| Foundation model | `polymathic-ai/aion-base` | frozen tokenizer + backbone. |

Full retrieval records, URLs and checksums: `docs/data_provenance.md`.
Sample contract, coverage tables and the selection function: `docs/DATA.md`.

## 4. Crossmatch, cleaning, and the DR1 → DR2 change
- **Original match:** naive **5″ nearest-neighbour** between DESI target RA/Dec
  and the eROSITA catalogue → 33,938 pairs → 30,441 unique DESI targets →
  28,646 with a retrieved spectrum.
- **Quality problem:** separations pile up at the cut (median 3.08″, 52% > 3″);
  14% of eROSITA sources match more than one DESI target, duplicating one X-ray
  label onto several spectra.
- **Cleaning (NWAY):** validated against Salvato et al. 2025 (A&A 704, A344).
  Per-targetid class: 87.5% correct / 4.6% spurious / 3.3% wrong / 3.4%
  ambiguous / 0.8% absent → `keep = correct` (`data/match_quality.csv`).
  Three independent confirmations that this localises real failures: the
  |z − z_NWAY| distribution is bimodal with an empty valley; mismatches have
  slightly larger separations; and the trained model is **2× worse on
  mismatches** (R² 0.29 vs 0.57).
- **`p_any` is not a substitute** for that filter and vice versa. `p_any` is
  P(the X-ray source has *any* LS10 counterpart); it never asks whether the DESI
  fibre sits on it. 336 of the 345 "wrong" sources have `p_any > 0.5`, median
  0.9994. Both cuts are needed.
- **DR2 is a purification.** eRASS:3 is ~2.7× the eRASS1 exposure here. On the
  23,283 sources both clean splits share (*measured*): median exposure
  120.9 → 330.4 s, median `flux_sig_lo` 0.1826 → 0.1181 dex. 1,917 DR1-clean
  rows dropped out — all of them `match_class == "correct"`, DR1 `DET_LIKE_0`
  median **7.53** against **14.12** for the survivors: the eRASS1 marginal tail
  that did not reproduce at 2.7× exposure. 2,299 rows are new. Sample as built
  2026-08-04: **25,582** rows, 24,796 unique detuids (*measured* 2026-08-06).
- **A fifth selection cut arrives with the rebuild.** `scripts/make_split.py`
  defaults to `--min-p-any 0.5`, an NWAY counterpart-reliability cut that is
  **not** in the 25,582-row artifacts and **is** in everything built after them.
  It costs roughly ten times as much GALAXY as QSO, and GALAXY is the science
  arm. Full statement, both sets of percentages and why neither is yet
  reproducible here: `DATA.md` §5.
- **Pipeline order:** clean → dedup to one row per targetid → **re-derive the
  split on the cleaned sample** → stage. Splits are always redone after cleaning.
  The rebuild chain is five steps and builds the manifest twice
  (`make_dr2_targets` → `make_targets_sidecar` → `build_manifest` →
  `make_split --require-spectrum` → `build_manifest --split`); the two-pass
  order is not optional and is written out in `DATA.md` §7.
- **Leak closed 2026-08-07.** Detections carrying more than one DESI fibre put
  the same X-ray photons on both sides of a targetid-grouped split — 773 such
  detuids on the pre-rebuild table, 756 on the rebuilt one. `scripts/make_split.py`
  groups on `ero_detuid` instead, with a keyed blake2b hash whose salt is written
  to the provenance JSON. It replaces `make_clean_split.py`. 7,667 of the 22,800
  targetids changed split as a result, so **no pre-2026-08-07 number is
  comparable to a post- one.**

<img src="figures/fig_cleanup.png" style="width:18cm" />

*(Figure predates DR2: the counts on it are the eRASS1 ones.)*

## 5. Targets and uncertainties
Targets and their split-normal error bars live in
`data/dr2/targets_sidecar_dr2.csv`, keyed by targetid. Coverage table in
`docs/DATA.md` §3.

| Target | Definition | Error model |
|---|---|---|
| `log_ml_flux_1` | log10(ML_FLUX_1), 0.2–2.3 keV | split-normal from LOWERR/UPERR: `sig_hi = log10(1+UPERR/F)`, `sig_lo = −log10(1−LOWERR/F)`. DR2 medians **0.1177 / 0.1053 dex** (*measured*; DR1 was 0.188 / 0.161). |
| `log_lx` | log10(4π D_L(z)² · F), Planck18 | same dex σ (the distance modulus is z-deterministic) |
| `log_flux_p1..p4` | band fluxes, P2 = 0.5–1.0, P3 = 1.0–2.0, P4 = 2.0–5.0 keV | split-normal, per band |
| `logmstar_cigale` | CIGALE stellar mass | `LOGM_ERR` |
| `log_sfr` | CIGALE SFR, 10 Myr average | `LOGSFR_ERR` |
| `log_mbh_pan25/vo09` | DR1 qmassiron VAC, MgII | catalogue errors |
| HR32 | never a direct target — implied from a (P2,P3) joint | derived |

**Availability is decided by DETECTION, not by the error bar.**
`DET_LIKE_MIN = 6.0`, one module constant, matching the eRASS Main catalogue's
own inclusion rule so the reported selection is the catalogue's rather than one
we invented. It was 5 until 2026-08-06. A band can pass a sigma cut while being
an upper limit, which hands the flow a non-measurement dressed as a measurement.

**The load-time sigma gate is retired.** `max_sigma = 1.0` removed **zero rows
in every band at either threshold** once detection was applied (table in
`docs/DATA.md` §5), and on the CIGALE targets it duplicated a cut already
applied at sidecar build time. `max_sigma` is `None` everywhere; the selection
function is now one cut per band.

<img src="figures/fig_targets.png" style="width:17.5cm" />

**How errors enter the objective — and why they no longer do.**
**No launcher can inject, but they say so in two different ways, because the two
training paths expose different flags.** `train-multi` has `--no-inject`, and
`sbatch/train_multi.sbatch:84` passes it unconditionally. The single-target path
has no such flag at all; it has `--error-mode`, and `sbatch/train.sbatch:84` and
`sbatch/train_smoke.sbatch:49` both hardcode `--error-mode none` with no
`ERROR_MODE` knob to override it. `sbatch/eval_multi.sbatch` has neither flag
because eval cannot inject. So "every launcher passes `--no-inject`" is a
convenient shorthand and is literally false — one launcher passes it; the other
two express the same decision as `--error-mode none`. (`CLAUDE.md:41` still
carries the shorthand.)
*(Corrected 2026-08-06: this section, and `CLAUDE.md`, previously declared
`--error-mode inject --inject-samples 8`, and until the same day
`train.sbatch:53` and `train_smoke.sbatch:38` passed `--error-mode "$ERROR_MODE"`
under comments advertising inject as the adopted mode.)* The estimand is p(y|x). Injection
smears each label by its own split-normal error, and on the X-ray targets that
error is most of the residual variance, so the flow is pulled toward the
measurement kernel and away from the conditional. Errors belong in the reported
floor. Two related facts that were also stated wrongly: `--error-mode` defaults
to **`none`**, not `convolve`, and it exists only on the **single-target** path
— `train-multi` has no such flag at all.

Convolve is retained as an optional latent tool:
`NLL_i = −log ∫ p_flow(t|x_i) · K(y_i | t; σ_lo, σ_hi) dt`, fixed quadrature.
**σ-conditioning remains banned**: feeding a per-source error as a model input
leaks the answer and is unavailable at deployment.

<img src="figures/fig_errors.png" style="width:16.5cm" />

**Deferred:** sub-band non-detections are **upper limits (censored)**, not
missing. Proper treatment needs a censored likelihood (Tubín-Arenas et al. 2024
for the eROSITA-DE upper-limit database; Seppi et al. 2022 for
detection-threshold characterisation).

## 6. Model architecture
<img src="figures/fig_architecture.png" style="width:17.5cm" />

*(Figure shows the V1 single-target head; the trained architecture is V3b below.)*

Frozen **AION-base** tokenizer + backbone → per-modality token sequences
(`spectra` = 273, `image` = 576, WISE = 3, z = 1; missing modalities omitted).

- **V3b, the trained architecture:** per-target **CLS vectors** + **shared
  per-block Q/V read adapters**. The data stream is read-only and stays under
  `no_grad`, so the encoder is untouched; one shared 768→512→256 MLP over the
  stacked CLS states feeds one small NSF flow per target.
- **Flow:** Zuko conditional NSF, 8 transforms, context 256, standardized
  target, KDE (Scott) prior as the information-gain reference. `context_dim` is
  a constructor argument, not a hard requirement — three call sites hardcode
  256, which is the real obstacle to widening it.
- **Joint head:** currently 4-D over **(logmstar_cigale, log_sfr, log_lx,
  log_flux_p3)** with `log_flux_p3` marginalisable by quadrature when missing.
  Flow column order is **JOINT_PAIR declaration order**, which is *not*
  MULTI_TARGETS order — those four dims sit at target columns (8, 7, 1, 5).
  Address it with `joint_dims()` / `joint_col()` / `target_col()`; positional
  indexing here is silently wrong rather than loudly wrong, and produced a bug
  that survived a whole run cycle.
- Per batch a modality combo is sampled; AION frozen; AdamW; detached-EMA loss
  normalisation so harder targets do not dominate the gradient.

**Why the joint is not decoration.** Refit under an identical protocol, the
joint beats the sum of its own marginals by **+0.80 nats, flat to ±0.04 across
every body** from epoch 2 to 10. There is irreducible structure among
(M*, SFR, Lx, P3) that independent heads cannot represent, and a better
representation does not absorb it. HR32 exploits the same mechanism: hardness is
a monotone function of the log-flux *difference*, so p(HR|x) is the (P2,P3)
posterior marginalised along a shear — a real information gain for a target that
was never trained.

**Original paper result** (q4/l2, epoch-13), unchanged and shown for reference:
<img src="figures/fig_results.png" style="width:16.5cm" />

## 7. Training and evaluation
- One `train-multi` job per run (`sbatch/train_multi.sbatch`), bucketed with
  bucket accumulation, `--no-inject`, heads selected with `--drop-heads`.
- **Metrics:** primary **IG / exp(IG)** against a KDE prior; secondary **R²** and
  reduced-χ² using per-source σ and the measurement-error floor; per-class
  split. For the joint, the dependence against the summed marginals.
- **One class column, named.** Per-class splits are defined by DESI `spectype`
  (QSO 22,299 / GALAXY 3,277 / STAR 6). `cigale_spectype` is *not* a second
  opinion about class — it is CIGALE's label on the rows CIGALE fitted, so it
  answers "does this row have an SED fit", and using it as the class arm drops
  513 galaxies and hides all 6 stars. The live sidecar carries only
  `cigale_spectype`, so today's arms are the wrong ones; plan step 12 adds
  `spectype` and `make_run_packet.py` will switch to it automatically. Check the
  `class_col` stamp before comparing two `by_spectype.csv` files. Full statement
  in `DATA.md` §3.
- **Per-combo row sets must be declared.** R², RMSE and IG all recompute their
  denominators on whatever rows a combo supports, so a difference between two
  combos is only a statement about *inputs* when the row set is held fixed
  (`--sample common`). Otherwise it is partly a statement about row sets.
- Eval must match the training likelihood: a convolve-trained checkpoint is
  scored through the same kernel, read from its own config.

## 8. Key findings so far
- Naive 5″ matching is ~8% wrong per NWAY; the paper's model is ~2× worse on the
  mismatches (R² 0.29 vs 0.57); dropping them lifts test R² 0.549 → 0.567 with
  no retraining.
- **Hardness is not learnable at eRASS1 depth**: Var(u) < E[σ_u²] at every S/N
  gate. It survives only as an implied target marginalised from a (P2,P3) joint,
  where the correlated posterior more than doubles the signal (corr +0.238 vs
  +0.10 for independent bands).
- **The frozen AION representation is not the bottleneck**: a LoRA fine-tune
  (V3a) ties with the frozen encoder; the read-only CLS (V3b) is what wins.
- Faint-band σ's in the eRASS1 catalogue are systematically **overestimated** —
  P1 and P3 land above their σ-derived R² ceilings, so those ceilings are lower
  bounds. Whether this survives at eRASS:3 depth is untested.
- **Multi-task transfer is a broad win**: six of seven heads beat their
  specialised single-target runs in one pass.

## 9. Status
The quoted best is the V3 multi-target run (job 35416432, 30 ep / 4 h): test
all-inputs **flux 0.604 · Lx 0.919 · logM★ 0.744 · P1 0.364 · P2 0.404 ·
P3 0.406**, beating every specialised baseline it replaces.

**Read those numbers with the caveat attached.** They are measured on the **DR1
labels and the DR1 test set**. Three uncontrolled deltas separate them from
anything the DR2 rebuild will produce: the relabel itself (deeper photons,
1,917 rows out and 2,299 in), a detuid-grouped re-split that necessarily
reassigns rows, and the missing-modality masking fix, which changes the input
distribution every image-bearing combo was trained on. The rebuild declares a
**fresh test set**; no DR1-era figure should be quoted beside a DR2 one as a
before/after.

The binding constraint on the DR1 runs was **overfitting** (train-probe gap
0.157 nats), not learning rate or capacity.

Per-run detail and the reasoning behind each choice live in
[`decisions.md`](decisions.md), which is the source of truth for results.
The ordered rebuild plan is [`rebuild_plan_2026-08-06.md`](rebuild_plan_2026-08-06.md).

## 10. Open
- The rebuild itself: DR2 target table → sidecar → detuid split → re-stage
  (plan steps 11-13, 17), which makes the staged HDF5 inputs-only.
- Missing-modality masking inside `encode_tokens`, and presence-aware combo
  sampling so an image-less source never draws an image combo.
- Named joints (`--joint NAME=dimA,dimB`) with checkpoint remapping, so a P2×P3
  run is loadable by eval.
- Reliability: wire `NWAY_p_any` from `eRASSc3_Main_LS10.fits`, the only column
  covering both the current sample and the expansion.
- The 104,945-row expansion: blocked on the CIGALE VAC download, on spectra and
  cutouts still arriving, and on z ≤ 0 rows.
- Per-head LR schedules (P1 never learns; P2/P3 still descending at 30 epochs);
  censored band-rate likelihood; the Buchner non-detection test.

## 11. Deferred / out of scope
Γ (a deterministic relabel of HR), X-ray upper-limit censoring, a full Salvato
re-crossmatch of "the rest", the V2 self-trained encoder, the 50-epoch paper
reproduction.
