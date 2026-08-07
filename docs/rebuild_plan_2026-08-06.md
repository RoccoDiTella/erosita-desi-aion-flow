# DR2 rebuild implementation plan (2026-08-06)

Produced by an 8-agent workflow: 4 parallel specs (data layer, single-phase
training, missing-modality masking, cleanup inventory), 3 adversarial critiques,
1 synthesiser. Every claim marked verified was re-derived from files on this
machine. Nothing has been executed.

---

# Rebuild implementation plan: Run A (collider) and Run B (modality)

Repo root: `/home/roccoditella/astroai/erosita-desi-aion-flow`. Data root: `/home/roccoditella/astroai/stanford_deadline/data`.
Everything below marked "verified" I re-derived from the files on this machine, not from the specs.

---

## 1. CORRECTIONS

Where your stated facts are wrong or incomplete. Ordered by how much they change the plan.

**C1. The NWAY reliability cut cannot be applied to either run's sample as described (decision 4).**
`new_targets_nway.csv` has **zero** targetid overlap with `clean_split_dr2.csv` (verified: 0 of 25,582; also 0 of 25,582 against the sidecar). Its 104,945 rows are the expansion only. So "p_any>0.5 keeps 97,343 / >0.8 keeps 90,955 / >0.9 keeps 86,463" (all verified) are statements about the *expansion*, not about a single source either run would train on today. There is no `nway_p_any` for any current-sample row anywhere in `data/dr2/`.
Consequence: `clean_split_dr2_nway.csv` cannot be built from that file. The reliability signal that does cover the current 25,582 is `NWAY_p_any` in `eRASSc3_Main_LS10.fits`, which is DR2-native and covers both samples in one convention. That is the column to wire in.

**C2. `p_any` and `match_quality.keep` are orthogonal, not alternatives.**
336 of the 345 sources `match_quality` calls "wrong" have `p_any > 0.5`, median 0.9994 (spec-measured, independently reproduced by its critique). `p_any` = P(this X-ray source has *any* LS10 counterpart). It does not test whether the DESI fibre sits on that counterpart. Both filters are needed and neither substitutes for the other.

**C3. The CIGALE labels already carry the DR1 keep cut, silently.**
`scripts/make_targets_sidecar.py:105-107` builds `want` from `mq.loc[mq.keep]` (verified). Every `logmstar_cigale` (19,210), `log_sfr` (17,144) and `log_mbh_*` value in the sidecar is keep=True by construction. Two consequences: (i) applying `keep` to Run A's 17,118-row M*+SFR+Lx sample costs **exactly zero rows** (verified: 17,118 with all three); (ii) rebuilding the sidecar for the 104,945 expansion with that gate in place yields **zero CIGALE labels**, because `match_quality` has no rows for those targetids. This gate must be removed and replaced by a `--universe` argument.

**C4. The three stacked band gates are one gate.** Measured on `targets_sidecar_dr2.csv`:

| band | finite flux | +max_sigma 1.0 | DET>5 | DET>5 & sigma | DET>6 | DET>6 & sigma |
|---|---|---|---|---|---|---|
| P1 | 23,364 | 22,583 | 14,122 | **14,122** | 12,847 | **12,847** |
| P2 | 25,275 | 25,095 | 20,924 | **20,924** | 19,759 | **19,759** |
| P3 | 25,113 | 24,842 | 20,269 | **20,269** | 19,102 | **19,102** |
| P4 | 17,219 | 14,399 | 3,736 | **3,736** | 3,071 | **3,071** |

The load-time `max_sigma=1.0` gate (`multitarget.py:196-199`) removes **0 rows in every band** once DET_LIKE is applied, at either threshold. It is entirely subsumed. Delete it so the selection function is reportable as one cut. P2∧P3: 17,029 at DET>5, 15,589 at DET>6 (both verified; the data spec's 17,031/15,591 are off by 2 because they assume the SIG_CAP clamp it also proposes).

**C5. `det_like_0 > 6` for all 25,582 rows (verified).** Adding a broad-band detection gate at 5 or 6 is a documented no-op on today's sample. It only bites on the expansion. Also: `log_lx` is finite for **25,549**, not 25,582 — 33 rows are censored by SIG_CAP at sidecar build.

**C6. "21,758 QSO / 2,764 GALAXY" is `cigale_spectype`, not DESI spectype.** The DR2 sidecar has **no `spectype` column at all** (verified: 47 columns, only `cigale_spectype`, which is QSO 21,758 / GALAXY 2,764 / **NaN 1,060** and contains zero STARs). DESI spectype for the same rows is QSO 22,299 / GALAXY 3,277 / STAR 6. Run A's QSO negative control and GALAXY science arm are defined differently under the two, and 1,060 rows have no CIGALE class. Stratification must be sourced from the DESI side, carried into the new target table.

**C7. `--export=ALL` is a hard crash today, not a silent DR1 run.** `targets_sidecar.csv` (the DR1 default at `sbatch/train_multi.sbatch:30`) has 39 columns and contains none of `det_like_p1..p4`, `det_like_0`, `log_ml_flux_1`, `flux_sig_lo/hi`, `log_lx` (verified). `load_multi_target_matrix` raises at `multitarget.py:206-213` before the first forward. The real silent mixup is the reverse direction: `sbatch/posterior_structure.sbatch:27-28` defaults to DR2 and will happily analyse a DR1-trained checkpoint.

**C8. Flipping the `sidecar` flags does not change the sample, only the label.** Row membership is set by `build_dataloaders(..., target_name="log_ml_flux_1")` -> `_finite_target_rows` (`data_to_aion_embeddings.py:553-593`), which reads `log_ml_flux_1` **from the staged HDF5** (DR1) and drops every non-finite row. Nine call sites hardcode it: `multitarget.py:523`, `eval_multitarget.py:70`, `hr_from_joint.py:115`, `posterior_structure.py:201`, `refit_heads.py:255`, `throughput_probe.py:72`, `cls_smoke.py:101` and `:111`, `validate_staged.py:387`. So decision 1 as scoped fixes the label but leaves the DR1 *detection limit* selecting the sample, and once the staged file loses the column those nine sites `KeyError`. This is the larger half of decision 1.

**C9. `logmstar` cannot be flipped to `sidecar: True`.** It is `sidecar: False` (`multitarget.py:41`) and is read from the staged HDF5, so "absent from the DR2 sidecar" is true but not the operative fact. Flipping it makes the column all-NaN and `TargetStandardizer.fit` raises. The reason the SFR-vs-mass guard at `eval_multitarget.py:176` has never fired is that the head is **dropped from the trained head set**, not that the sidecar lacks it. Keep the `_ALL_TARGETS` entry (checkpoint compat, see C11); drop it per run.

**C10. `ConditionalNSFFlow` does not hard-require context 256.** `normalizing_flow.py:201` makes `context_dim` a constructor argument; `distribution()` validates against `self.context_dim` (verified). Changing the conditioning width is free at the flow. The real obstacle is three hardcoded `MultiTargetFlows()` calls (`eval_multitarget.py:64`, `hr_from_joint.py:108`, `posterior_structure.py:194`) that would fail to load a wider-context checkpoint.

**C11. A newly trained P2xP3 joint would be unloadable by eval.** `configure_heads` names the last head `"joint"` (`multitarget.py:96`, `:115`); `configure_heads_from_config` restores the 2-D joint only when the stored `heads` list literally contains `"p2xp3_joint"` (`:134-136`), a string produced only by the legacy retro-name at `:148`. Train Run B today and eval rebuilds a 4-feature joint and fails on shape. Run B's headline experiment needs the joint declared in `config.json`.

**C12. The missing-image guard cannot be keyed on `combo`.** In the bucketed path the `combo` handed to `encode_tokens` is `bucket["union"]` (`multitarget.py:739`, `:752`), not the per-source combo; per-source exclusion arrives through `modality_dropout`. Any check of the form "does this row have at least one modality in `combo`" passes rows that are fully masked in practice. The guard must be evaluated on the merged `input_mask_dict`.

**C13. A fully masked row does not raise; it reads uniform attention over the tokens you excluded.** `cls_read_step` fills invalid logits with `-torch.finfo(dtype).max`, not `-inf` (`data_to_aion_embeddings.py:772`), and the CLS path returns `group_ids = torch.zeros(...)` at `:1112-1115` before `_group_ids_from_modality_mask` is ever called, so the "unmapped token ids" RuntimeError at `:947` cannot fire on this path. The data spec's proposed pre-flight test (b) proves nothing. `ComboSampler.default()` samples `('image',)` as a legitimate combo (`attention_pooling_head.py:333-338`), so this is reachable for every image-less source.

**C14. Run A is not implementable by any data-layer change.** There is no path for a non-AION scalar into the flow context (contexts are exactly `head(cls_seq)`, `multitarget.py:757-758`); `JOINT_PAIR` is a module constant with no CLI; and `posterior_structure.py:216` samples the **full unconditional joint**, i.e. r(Lx, SFR) marginalised over M*, not conditioned on it. Run A is blocked on model-layer work, not on DR2 targets.

**C15. `make_dr2_targets.py:100` fabricates labels for non-positive z.** `np.clip(z, 1e-4, None)` manufactures a `log_lx` roughly 7 dex too small instead of leaving NaN. Harmless today (current-sample min z is positive) but the expansion has **957 rows with z <= 0** (901 STAR, 56 GALAXY) and 1,667 with z < 1e-4, min z = **-0.00182** (all verified). `validate_staged.py:221-225` asserts `z.min() > 0` and will fail the staged expansion.

**C16. Detuid leakage is real and survives a targetid split.** 773 detuids carry more than one DESI fibre in the current sample (verified). 50 detuids are shared between the current 25,582 and the 104,945 expansion despite zero targetid overlap (verified) — merging the two under a targetid split puts the same X-ray photons in two splits.

**C17. `IronPhysProp_v1.2.fits` (the CIGALE VAC) is not on this machine** (verified by `find` over `/home/roccoditella/astroai`; only `VAC_BHmass_338_v1.7.fits` is present). The sidecar cannot be rebuilt for the expansion until it is located. Run A's labels for anything beyond the current 25,582 are blocked on this.

**C18. CLAUDE.md:37 is stale in three ways, not one.** `main.py:883` sets `--error-mode ... default="none"`; `train-multi` has no `--error-mode` at all; and the same line still declares the operating mode to be `--error-mode inject --inject-samples 8`, which decision 5 reverses. `sbatch/train_multi.sbatch` still passes `--inject-samples 50` and never `--no-inject` (verified), even though `main.py:1014` defines the flag. The Pointers block also still says account `finkbeiner_lab`, `-p gpu`, and every sbatch header still reads `#SBATCH -p gpu` or `gpu_h200` with no `--account` line.

---

## 2. ORDERED IMPLEMENTATION PLAN

Effort key: trivial < 1h, small = half day, medium = 1-2 days, large = 3+ days.

### (i) Blocking correctness fixes — nothing runs before these

**1. Back up the four unbacked netscratch runs.** `bash scripts/backup_run.sh mt-v2-accum-35073203 mt-v3b-8head-34994658 mt-v3-lrfix-35416432 mt-v4-sfr-35828655`, then re-run it for `p1-dr2-37257713` (its home backup predates `poststruct_allmod/noWISE`). These are on purge-eligible netscratch and `mt-v3-lrfix-35416432` is the "current best" every doc quotes. Do this before touching anything. **trivial**

**2. Extract eval internals to `shareable_aion_flow/eval_core.py`.** Move the joint block, combo-table builder, SFR-vs-mass baseline and HR summary out of `scripts/eval_multitarget.py:main()` into importable functions with an injectable `encode` callable. `scripts/eval_multitarget.py` keeps only argparse, checkpoint load, device plumbing, CSV writing. Land this **before** the behaviour fixes so the tests demonstrably fail on today's code first. This is why `j2, j3 = JOINT_IDX` survived a whole run cycle. **medium**

**3. Name-resolve the joint everywhere.** Add to `shareable_aion_flow/multitarget.py` (beside `_joint_idx`, ~line 89): `joint_dims()` (flow-column order), `joint_col(name)` (position inside the flow feature vector), `target_col(name)` (column in the target matrix), `joint_availability(targets)` (the `have_req`/`have_all` rule, extracted verbatim from `multi_target_nll:340-352`). Then fix:
- `scripts/eval_multitarget.py:117` — replace `j2, j3 = JOINT_IDX` with name resolution; guard the HR block at `:126-131` on `log_flux_p2` and `log_flux_p3` both being present and index by `joint_col`, not 0/1; print a skip reason otherwise. `:149` `"head": "p2xp3_joint"` -> `HEAD_NAMES[-1]`, plus a `joint_dims` column. Note the module is imported as `_mt`, not `mt` (`:55-56`).
- `scripts/hr_from_joint.py:111` — same unpack; additionally `SystemExit` when `len(joint_dims()) != 2`, because `line_log_density` is an exactly-2-D shear quadrature.
- `scripts/eval_multitarget.py:176` — `"logmstar"` -> select from `("logmstar_cigale", "logmstar")` by availability, record which was used, and `print` an explicit skip line when neither is there. Add a test asserting every head-name literal in eval exists in `_ALL_TARGETS`.
Report the fully observed and quadrature-marginalised joint populations as **two numbers** on both the train and eval sides; do not try to make one eval number match the trainer's count-weighted mixture of two dimensionalities. **small**

**4. Tests for steps 2-3: `shareable_aion_flow/tests/test_eval_core.py`.** Joint block runs on both joint shapes; HR uses P2/P3 by name against a 3-D joint ordered `(log_sfr, p2, p3)`; `len(joint_dims()) == flows.joint.features`; SFR-vs-mass baseline fires for the default head set; end-to-end `run_eval` on the existing tiny staged fixture with a stub encoder, on CPU, with no AION and no checkpoint. **medium**

**5. Missing-modality masking inside `encode_tokens`** (`shareable_aion_flow/data_to_aion_embeddings.py:1004-1122`). Add `modality_presence(batch)` and `group_presence(present)` next to `TOKEN_KEYS_BY_MODALITY` (`:48-53`) as the single definition of "this input exists", keyed per token key so a missing W3 does not kill W1/W2. Add `sanitize_missing` so a non-finite input cannot poison a masked read (0.0 * NaN = NaN). Extract the mask merge as a pure `build_input_mask(token_dict, present, modality_dropout)` so it is testable without `aion` installed. Then:
- **Assert token-key coverage**: for every group in `combo`, at least one of its keys must appear in `token_dict`. Without this a renamed AION key turns the entire fix into a silent no-op (`aion` is not installed here; the literals have never been checked against a real `codec.encode`).
- **Evaluate the empty-row guard on the merged `input_mask_dict`, not on `combo`** (C12). Return a per-source `has_tokens` mask.
- Delete the blank-image test at `multitarget.py:964-966` and `images=sub[5]` at `:736-740`; migrate `test_blank_cutout_drops_the_image_modality_for_that_source_only` to the encode_tokens level.
**medium**

**6. Honour `has_tokens` at every consumer, in the same change as step 5.** Trainer (`multitarget.py:757`), both validation loops (`:816`, `:836`), `eval_multitarget.py:99-124`, `hr_from_joint.py:147`, `posterior_structure.py:214`. Mask the row's targets to NaN before `multi_target_nll` (which already treats NaN as unavailable) and count the exclusions. If any NaN reaches `clip_grad_norm_` (`:779`/`:788`) it poisons the whole optimizer step, not one row. **small**

**7. Presence-aware combo sampling.** `ComboSampler.sample_supported(present)` in `shareable_aion_flow/attention_pooling_head.py:311-338`: filter sizes to those with at least one supported combo **before** the uniform size draw (an image-less source has zero size-4 combos and the naive version indexes an empty tuple). Compute the presence matrix once per batch on CPU, not per source per group on device. Wire into `multitarget.py:725`; for the non-bucketed path subset the rows instead. Re-calibrate `--bucket-chunk`: filtering shifts bucket occupancy and the "heavy bucket ~46%" figure the OOM headroom rests on no longer holds. **medium**

**8. `shareable_aion_flow/tests/test_missing_modality.py`.** Each sentinel flagged; blank image masked with `modality_dropout=None` and byte-identical with and without it; fully masked row refused; sanitize keeps present rows bit-identical; sampler never draws an unsupported combo; `bucket_modality_dropout` expresses combo membership only. Plus a numerical companion asserting `cls_read_step` output *changes* when masked data tokens change, so the guard is documented as load-bearing rather than defensive. **medium**

**9. Decouple sample membership from the DR1 flux column** (C8). `data_to_aion_embeddings.py`: `_finite_target_rows` returns `rows` immediately when `target_name is None`; `AIONHDF5Dataset` emits NaN in the target slot and skips `TARGET_ERROR_COLS` lookup; keep the 10-tuple arity, append `has_image`/`has_wise` as elements 10-11 so existing positional indices are stable. Pass `target_name=None` at **all nine** call sites listed in C8. **small**

**10. Target-spec fixes** (`shareable_aion_flow/multitarget.py:39-45`). `log_ml_flux_1` and `log_lx` -> `"sidecar": True`, `det: ("det_like_0", <threshold>)`. `max_sigma: None` on every sidecar target (C4: it removes zero rows and is a second unreportable cut on the CIGALE labels, which `make_targets_sidecar.add()` already gated at 1.0). **Keep** the `logmstar` entry in `_ALL_TARGETS` — deleting it breaks `configure_heads_from_config`'s name-derived drop set and makes V3 unloadable (C9) — and drop it per run via `--drop-heads`. **trivial**

**11. Detuid-grouped hash split: `scripts/make_split.py` (new).** Keyed blake2b over `ero_detuid` with a recorded salt distinct from `--seed`, 0.80/0.10/0.10. Inputs: the new DR2 target table plus a `has_spectrum` flag. Emit `targetid,split` and `split_provenance.json` (salt, fracs, group key, sidecar sha256, per-split counts, n_changed vs previous). Do **not** inherit the old labels; they are targetid-grouped and preserve the 560 leaky rows. Check per-split fractions against a tolerance and `n_changed`; the "no detuid straddles a split" assertion is true by construction and tests nothing. **small**

**12. `scripts/make_dr2_targets.py`: carry reliability, photometry, DESI metadata and the grouping key.** Add `NWAY_p_any`, `NWAY_p_i`, `NWAY_p_single`, `NWAY_match_flag`, `NWAY_Separation_LS10_ERO`, `NWAY_threshold6`, `LS10_OBJID/BRICKID/RELEASE`, `LS10_flux_w1/w2/w3` (+ivars), `Exgal_prob_STAREX` (capital E), `simbad_known_galactic`, `class_gal_exgal`, plus `spectype`, `zwarn`, `survey`, `program`, `healpix` from the `--desi` frame, and `desi_ls10_sep_arcsec`. **Remove the z clip at `:100`** — leave `log_lx` NaN for z <= 0 rather than fabricating it (C15). Drop the 128 rows matched to `NWAY_match_flag == 2` secondary candidates, or re-point them at the primary. Do **not** adopt the SIG_CAP clamp the data spec proposes: with injection off the sigma is ignored, so censoring a flux whose lower bar swallows it is protective, and the clamp buys ~2 rows after the DET_LIKE gate the same spec adds. **medium**

**13. `scripts/make_targets_sidecar.py:105-107`: remove the `match_quality.keep` gate on `want`** (C3); replace with `--universe <dr2 target table>`; drop `--match-quality`. Log the `--max-sigma` rejections separately from `failed`/`sentinel`/`broad_pdf`. **small**

**14. Launcher hygiene: `sbatch/_dataset.sh` (new) with no default.** `: "${DATASET:?set DATASET=dr2 explicitly}"`, resolve `CLEAN_SPLIT_CSV`/`EXTRA_TARGETS_CSV`/`STAGED_DIR`, then a preflight that checks required columns and asserts `median(flux_sig_lo)` in `(0.10, 0.14)` — a property of the photons, not of a filename, so a renamed or half-merged sidecar is caught (DR2 measured 0.1177). Give the `dr1` branch its own column list or delete the branch: the DR1 sidecar has none of the DR2 columns and would exit before reaching the median check. Source it from `train_multi.sbatch`, `eval_multi.sbatch`, `posterior_structure.sbatch`. Extend the existing preflight at `train_multi.sbatch:60-69` rather than adding a second heredoc. Add `--no-inject` to every launcher and delete `INJECT_SAMPLES` (C18). Fix `CLAUDE.md:37` on all three counts and the Pointers block on account/partition. **small**

**15. Delete the two-phase path — but port two things first.** Port `scripts/refit_compare.py:89-108`'s dependence-captured metric into a per-epoch `val/dependence_<joint>` in the trainer, then delete `scripts/refit_heads.py`, `scripts/refit_compare.py`, `sbatch/refit_heads.sbatch`, and `tests/test_multitarget.py:311-349`. In `multitarget.py` delete `--joint-only` (`main.py:1035-1040`) and `--snapshot-every` (`:1049-1052`) and their branches at `:572`, `:586`, `:660`, `:695-698`, `:763`, `:783`, `:793`, `:894-895`. **Keep the EMA pin** but re-derive it from `N_HEADS == 1` (`:710-711`): with one head, weight `1/EMA` is a second LR schedule fighting the cosine, and Run A can hit that. Hold the deletion until the deck stops showing the two-stage slides (`docs/make_two_stage_figures.py` reads `results/dr2_37257713/refit_epoch*.json`). **small**

**16. Named joints: `JointSpec` + `--joint NAME=dimA,dimB[:marginal=dimC]`** (`multitarget.py:82-148`, `:236-246`, `:392-414`, `main.py:983-1067`). `MultiTargetFlows.joints` becomes an `nn.ModuleDict` keyed by name; add `remap_legacy_state` so `joint.*` keys load into `joints.<name>.*`; persist `joints` into `config.json` and read it first in `configure_heads_from_config`, keeping the two legacy paths. Reject a joint declaring more than one marginalisable dim (today those rows are silently discarded). **Verify the remap loads `mt-v3-lrfix-35416432` and `mt-v4-sfr-35828655` before step 15 deletes anything.** Rewrite the six `test_multitarget.py` tests that assume `N_HEADS == N_TARGETS + 1` or read `mt.JOINT_PAIR`/`mt.JOINT_IDX` (`:59`, `:171-188`, `:224-227`, `:243-258`, `:274-275`, `:449`). **medium**

**17. Rebuild: DR2 target table -> sidecar -> split -> stage.** Run steps 12/13/11 in order, then re-stage. `data_to_aion_embeddings.py` becomes inputs-only: delete `:291-301` (the DR1 finite-flux row filter and the `read_ml_flux_1` call, which sits **outside** the range the data spec names), the `dropped_nonfinite_targets` key at `:328`, and `:358-372` in the split writer including the second `read_ml_flux_1`. Set `EXTRA_TARGET_COLUMNS = ()`; keep `spectype` (metadata, sourced from the DESI side per C6). Write `has_image`/`has_wise`. **medium**

### (ii) Run A enablement (collider)

**18. `ScalarConditioner` + checkpointing** (`multitarget.py`, new section after ~`:214`). Standardized log M* -> small embedding concatenated onto every head's CLS context; `ctx_dim = 256 + out_dim`. Hard-fail at startup if a `COND_SCALARS` name is also in `MULTI_TARGETS`, and never load `_sig_lo`/`_sig_hi` for a conditioning scalar (sigma-conditioning ban).
Three things the spec omits and that are fatal without them:
- **`save()` (`:684-692`) must store `cond_state_dict`.** Today it stores only encoder/head/flows/standardizers/ema, so `posterior_structure` would rebuild a *randomly initialized* conditioner, concatenate it, sample, and print a number with no error. Run A's only deliverable is that number.
- **`context_dim` must be recorded in `config.json` and passed at all three `MultiTargetFlows()` call sites** (`eval_multitarget.py:64`, `hr_from_joint.py:108`, `posterior_structure.py:194`), which hardcode the 256 default and would fail `load_state_dict` on flow layer 0.
- The `missing`-embedding branch as sketched discards the observed scalars whenever *any* is absent (`torch.where(present.all(1), e, miss)`). Harmless at one scalar, wrong the moment `COND_SCALARS` grows. Also drop the `2**arange(8)` Fourier features to something far lower: `sin(128 z)` on a standardized scalar is enough capacity to memorise M* per source, in a model simultaneously predicting SFR from the same CIGALE fit.
**large**

**19. `--require-cond` as a filtered split CSV, not a NaN mask.** There is no per-sidecar-column row filter in `build_dataloaders`. Masking to NaN does not drop the 6,372 M*-less rows: they still cost forwards, still enter `n_seen`, and the `log_lx` marginal head still trains on them, so "at fixed M* is never evaluated on sources whose M* is unknown" would be false. Emit `clean_split_dr2_runA.csv` at split time and note that `steps_per_epoch`/`total_steps` (`:604-610`) follow from `len(train_loader)`. Verified: SFR is a **strict subset** of M* (0 rows with SFR and no M*; 2,066 M*-only), so this costs zero SFR labels. **small**

**20. `--fixed-combo` / `--val-combo`** (`multitarget.py:653`, `:717-806`, `:809-851`; `main.py` train-multi block). Run A pins `spectra+z`, which means it needs **no cutouts at all** and is not blocked on the image download. `SystemExit` if `--bucketed` is also passed (because bucketing is pointless with one combo, not because it errors — `('spectra','z')` is a valid member of the `spectra` bucket). Persist both into `config.json`; make `eval_multitarget`/`hr_from_joint`/`posterior_structure` default to `config['fixed_combo']` and `SystemExit` when asked for a combo the model never saw. Note `eval_multitarget.py:125`/`:152` gate the HR and all-inputs blocks on the literal `"spectra+z+wise+image"` — replace with `combo_name(MODALITIES)` or those branches can never fire under a fixed-combo run. **medium**

**21. Selection metric** (`multitarget.py:852-903`). Replace `val_pair_mean` (`:860`, which averages a 4-D joint NLL against 1-D marginals and gives every head an equal vote) with `--select-metric`: `head:<name>@complete` for Run A, driven by the fully observed 2-D density the correlation estimand comes from. Rename `save()`'s `"val_multi_nll_sum"` key to `"val_select_score"` and present it as a bug fix: it has always been called with `val_pair_mean` (`:890`, `:895`, `:899`), never the sum. Extend `val_r2` (`:852-857`, which loops `flows.flows` only) to report per-dim R2 for joints, since Run A's only R2-bearing head may be the joint. **medium**

**22. `scripts/posterior_structure.py:173-176`, `:194-196`, `:207-237`.** Add `--joint NAME`; rebuild the conditioner from the checkpoint; restrict pooled sources to those having **every** modality in `--combo` and record `n_sources_conditioned` / `n_excluded_missing_modality` in the JSON; `--group-csv` can point at the sidecar's `cigale_spectype` for the QSO/GALAXY arms, with the 1,060 NaNs reported (C6). Assert in the docstring that with M* as an input, `r(log Lx, log sSFR) == r(log Lx, log SFR)` **exactly**, so nobody derives it and reports the same number twice. **medium**

**23. Run A launcher `sbatch/train_run_a.sbatch`.** `DATASET=dr2`, `--fixed-combo spectra+z`, `--cond-scalar logmstar_cigale --require-cond`, `--joint lx_sfr=log_lx,log_sfr:marginal=log_sfr`, `--drop-heads` everything except `log_lx` and `log_sfr` (keep both marginals in the same job: the joint-vs-independent comparison then comes free in-run and the trunk gets three gradients), `--no-inject`, `--select-metric head:lx_sfr@complete`. **small**

### (iii) Run B enablement (modality / information gain)

**24. Manifest: `scripts/build_manifest.py` (new).** One row per targetid: `targetid, ero_detuid, source_row, split, has_spectrum, has_z, has_wise, has_image, spectype, zwarn, z, nway_*, ls10_objid, mq_keep, mq_match_class, det_like_0, det_like_p1..p4`. **Define `has_wise` on flux > 0 (or ivar > 0) per band, not on finiteness**: LS10 W-band fluxes are finite for 25,582 of 25,582, so a finiteness rule makes the column and the WISE half of the presence sniff permanently inert while Run B's IG table treats WISE as universally available. `has_z` needs `zwarn`, which `new_targets_nway.csv` does not carry — take it from the DESI properties join. **small**

**25. Per-combo sample declaration in `eval_core`.** `--sample {common,native,both}`; every table row stamped with `sample`, `n_test`, `n_sample`, `n_common`, `frac_of_test`. Enforce in code that an IG delta may only be taken within `sample="common"`, where the row set is identical (`information_gain_delta` raises on mismatched `sample` or `n`). **R2 and RMSE have the same problem**: `eval_multitarget.py:140` recomputes the variance denominator on each combo's own rows, so under `native` they are also incomparable — flag them, not just IG. Report each no-image combination twice (full sample and `has_image` subset) so the image delta is like-for-like. **medium**

**26. Training-side rule, not only eval-side subsetting.** Step 7 already prevents an image-less source drawing an image combo. Without it, `--sample=common` reports the dilution but the *training objective* still contains rows that read uniform attention. Both are needed. **(covered by step 7)**

**27. `--select-metric mean_ig`** for Run B: mean over declared heads of `(NLL_val - NLL_prior)/d_h`, with `d_h = len(spec.dims)` for a joint, KDE priors fit once on the train view and logged at startup so the constant offset is auditable. `--select-heads` **must include the joints**: selecting on marginals alone while the stated deliverable is "does the p2xp3 joint improve HR" means the dependence metric is logged with no vote. Note `d_h` normalisation is only correct for a joint with no marginalisable dim; assert that. **medium**

**28. Recompute the independent-marginal HR baseline in-run** (`eval_multitarget.py:216` currently prints a hardcoded `0.551` from a different run). Sample `flows.flows[p2]` and `flows.flows[p3]` independently on the same rows with the same `C_P2`/`C_P3` constants. This is the only way to answer Run B's headline question on this model. **small**

**29. `scripts/make_source_hdf5.py` (new)** for the expansion: `fetch_desi_spectra.merge()` for spectra plus a join for `redshift`, `flux_w1/w2/w3`, `target_ra/dec`, `spectype`; assert the grid (LAM0 3600.0, DLAM 0.8, NBIN 7781) rather than trust it; re-runnable as shards land. **`new_targets_nway.csv` cannot be fed to `make_dr2_targets.py --desi` as-is**: it has `mean_fiber_ra/dec`, not `target_ra/dec`, so `:72`'s `usecols` KeyErrors, and using fibre positions for the expansion while the current sample used target positions puts two astrometric references in one 1-arcsec match. Resolve the position convention explicitly. **medium**

**30. Compact image storage + validator.** `image_flux` of shape `(n_with_image, 4, 160, 160)` plus an `int32 image_row` (-1 = none): ~9.8 GB against ~43 GB dense over 105k rows on purged netscratch. Note this breaks `validate_staged.py:120-128`, which flags any dataset whose `shape[0] != len(desi_targetid)`, and any by-row image reader (`scripts/make_image_saliency.py`). Update `scripts/validate_staged.py`: new REQUIRED tuple (`:29-42`) forbidding `log_ml_flux_1`/`log_lx`/`ml_flux_1`/`logmstar`/`hr32_u`/`flux_sig_*`; replace the `log_lx` re-derivation (`:337-342`) with a sidecar-side identity check; report full-split modality coverage rather than `images[:64]`; new checks that no detuid spans two splits and that every split targetid has a staged row; relax the `z.min() > 0` assertion at `:221-225` to a reported count once step 12 stops fabricating. **medium**

**31. Run B launcher `sbatch/train_run_b.sbatch`.** `--bucketed --accumulate-buckets`, `--joint p2xp3=log_flux_p2,log_flux_p3`, `--joint mstar_sfr=logmstar_cigale,log_sfr`, `--drop-heads log_flux_p4 log_mbh_pan25 log_mbh_vo09 logmstar`, `--no-inject`, `--val-combo mixed` (seeded once per source, cached, and the train probe at `:809-825` needs its own cached assignment or `gap` is meaningless), `--select-metric mean_ig` with both joints in `--select-heads`. **small**

### (iv) Nice-to-have

**32.** `scripts/make_run_packet.py:53-71` — read `spectype` and sigma from the sidecar/target table, not the staged HDF5. It fails soft today (`:66` guards on column presence) so the breakage surfaces in a plot, not at load. **trivial**
**33.** Rewrite `docs/DATA.md`, `docs/decisions.md`, `docs/pipeline.md`, `docs/targets.md` against DR2. Record: DR1 vs DR2 depth (ML_EXP_1 120.9 -> 330.4 s; flux_sig_lo 0.1825 -> 0.1177); why the 1,917 DR1-clean rows vanished (all match_class "correct", none with an eRASS:3 counterpart within 1 arcsec, DR1 DET_LIKE_0 median 7.53 vs 14.12 for survivors — the eRASS1 marginal tail that did not reproduce at 2.7x exposure, i.e. a purification); what p_any measures and does not (C2); that the CIGALE labels carry the keep cut (C3); that the sigma gates are subsumed (C4). **medium**
**34.** `docs/figures/data_counts.csv` regenerate from DR2 (`build_deck.sh:39-46` still points `make_data_figures.py` at the DR1 split). **small**
**35.** Decide the fate of `shareable_aion_flow/main.py`'s single-target path, `scripts/codec_leakage_probe.py`, `scripts/make_sed_figure.py`, `scripts/make_image_saliency.py`, `scripts/line_shapley.py`, `shareable_aion_flow/evals.py` — all break on the inputs-only staged file or on the empty-input guard. Retire or update explicitly rather than mid-run. **small**

---

## 3. OPEN DECISIONS FOR THE USER

**D1. Reliability cut: which column, which threshold.**
Given C1, the only column that covers both samples is `NWAY_p_any` from `eRASSc3_Main_LS10.fits`. My recommendation: **flat `p_any > 0.5`, carried as a column so 0.8/0.9 is a re-split away, plus `match_quality.keep == False` dropped for the 25,311 current-sample rows whose `LS10_OBJID` is unchanged between DR1 and DR2** (271 differ, so the DR1 verdict is void for those; that gives 695 transferable rejects rather than 946).
What needs you: the cut is **~7x more expensive for galaxies than for QSOs**, and GALAXY is the science arm. On the Run A sample, `p_any>0.5` costs QSO -0.9% and GALAXY -6.8%; `p_any>0.8` costs QSO -2.1% and GALAXY -14.6%, taking the GALAXY test set from ~161 to ~138. Counterpart ambiguity correlates with extended faint hosts, which are exactly the galaxies where sSFR is measurable, so this is a selection correlated with the estimand applied hardest to the arm that is not the negative control. Note the catalogue's own recommendation is *looser* than any of these: `NWAY_threshold6` has median 0.045 over our rows and 25,136 of 25,218 already clear their own row's threshold.

**D2. DET_LIKE 5 or 6.**
Recommendation: **6**, matching the eRASS Main inclusion rule and your stated preference for catalogue conventions. It is a no-op for the broad band on today's sample (C5). The cost is real: on the combined 101,029 it takes P2 62,046 -> 54,928 (-11%), P3 59,273 -> 52,477 (-11%), P1 -16%, P4 -25%, and the **P2-AND-P3 joint 37,175 -> 31,363 (-16%)**. If the P2xP3 joint is the point of Run B, 5 buys 5,812 more joint rows. This is the one place where consistency and the experiment pull against each other. Also note `tests/test_multitarget.py:449` asserts `('det_like_p3', 5.0)` literally.

**D3. How M* enters Run A, and whether to do the cheap version first.**
Three routes: (a) `ScalarConditioner` side-channel (step 18) — full new-code path, ~large, and its silent-failure surface (unsaved weights, hardcoded 256) is what the plan spends most of its risk budget on; (b) tokenize M* through an AION scalar codec — keeps "the model only sees AION tokens" true but pushes CIGALE masses through a codec quantized on a different VAC's distribution, and `aion` is not installed here so nobody can check what it exposes; (c) **train the 3-D joint `(M*, Lx, SFR)` with no conditioning at all, then estimate r(Lx, SFR | M* = observed) by importance-resampling posterior draws with a narrow kernel on the M* axis.**
Recommendation: **land (c) first**, because it needs zero new model code, runs on a Run-B-shaped model, and gives you a number this week plus an independent cross-check on (a). Then build (a) as the deliverable. The two must agree; if they do not, one of them is wrong and you want to know that before the estimand goes in a draft.

**D4. Run on the current 25,582 now, or block on the expansion re-stage?**
Recommendation: **run both on the current 25,582 as soon as (i) lands**. Run A at `spectra+z` needs no cutouts at all. The expansion is blocked on locating `IronPhysProp_v1.2.fits` (C17, hard blocker for Run A labels), on ~29% of spectra still downloading, and on cutouts at ~22.6% with a ~78h ETA. A full re-stage of ~101k rows onto purged netscratch is a multi-day operation. Treat the expansion as a second pass. What needs you: the expansion is substantially fainter (P2 detection fraction drops from 77.2% to 46.6%, P4 from 12.0% to 2.0%), so per-head numbers will not be comparable between the two passes either.

**D5. Fresh test set.**
The detuid-grouped hash split necessarily reassigns rows relative to both `clean_split.csv` and `clean_split_dr2.csv`, so **no V3 number (flux 0.604 / Lx 0.919 / M* 0.744) is measured on the new test set**. Combined with the masking fix, which changes the input distribution any image-containing combo was trained on, and with the DR2 relabel, that is three uncontrolled deltas. Recommendation: declare a fresh test set and state it once, since both runs are from scratch anyway. You need to know before the split is frozen if any deck or draft compares against those figures directly.

---

## 4. CLEANUP

Verdicts. **Run item 1 of the plan (back up the four unbacked runs) before executing any DELETE.**

| Path | Size | Verdict | Note |
|---|---|---|---|
| `stanford_deadline/fits_pool.zip` | 10.6 GiB | DELETE | Zip of an already-unzipped live pool; 27,373 files both sides. Diff the member list first. |
| `aion_project/.../fits_incoming/` | 7.1 GiB | DELETE | 100% contained in fits_pool (0 basename misses). See open note on the colleague-named 3.9 G subdir. |
| `aion_project/runs/2026-04-12-runpod.../aion_hdf5_metadata_20260412.zip` | 1.94 GiB | DELETE | RunPod transfer bundle; keep the sibling README/logs. |
| `data/dr2/eRASSc3_Main_LS10_Public_27Jul2026.fits.gz` | 1005 MB | DELETE | Keep the decompressed `.fits` (that is what `make_dr2_targets.py` opens); record the URL in DATA.md. |
| `data/erosita_desi_matches_Xray_properties.csv` | 86.7 MB | DELETE | md5-identical duplicate; `build_deck.sh:35` uses the aion_project raw copy. |
| `data/erosita_desi_dr1_matches_all_properties.csv` | 28.8 MB | DELETE | Same; full md5 both sides before removing. **Do not delete before step 12** — it is the only source of `spectype`/`zwarn` for the current sample. |
| `netscratch outputs/*/last.pt` | 2.11 GiB | DELETE | `backup_run.sh` already treats these as transient; no job in queue. |
| `netscratch outputs/*/wandb/` | 21 MB | DELETE | Synced to cloud; bundle with the last.pt pass. |
| `netscratch outputs/{p1-smoke-*,p1-dr2-smoke-37256927}` | 237 MB | DELETE | Plumbing smokes; referenced by nothing. |
| `aion_project/trash/` | 4.3 MB | DELETE | Four months old, named disposable by its author. |
| `.pytest_cache`, `**/__pycache__` | 796 KB | DELETE | Gitignored; includes stale cpython-312 pyc. |
| `netscratch outputs/{v1-*,v2-spec-drop-*,v3-cls-*,paperhead-*}` (13 dirs) | 3.43 GiB | ARCHIVE | Already mirrored to `/n/home02/rditella/aionflow_results`; decisions.md §6 rows stay true from the home copies. |
| `netscratch outputs/p1-joint-36980372` | 911 MB | ARCHIVE | Mirrored; superseded by decision 3. Its `snapshots/` only fed refit_heads. |
| `netscratch .../data/staged_paper_smoke/` | 894 MB | ARCHIVE | Keep if you want to smoke the Run A/B launchers, which is what it is for. |
| `aion_project/data/staged/raw_tvsplit/` | 1.9 GiB | ARCHIVE | April RunPod split; keep summary.json + probe sample, drop the three HDF5s. Grep for `raw_tvsplit` first. |
| `netscratch .../logs/` (pre-DR2) | 1.2 MB | ARCHIVE | Tar to `$AIONFLOW_ROOT/backups/`; convention already exists. |
| `data/match_validation_slice.csv` | 6.2 MB | ARCHIVE | Intermediate of `match_quality.csv`; referenced nowhere. |
| `data/targets_bands.csv` (+ FASRC twin) | 5.3 MB | ARCHIVE | Explicitly labelled superseded at `train_multi.sbatch:28`. Leave `targets_extra.csv` alone until the DR2 loader path is grepped. |
| `scripts/refit_heads.py`, `refit_compare.py`, `sbatch/refit_heads.sbatch` | 24 KB | ARCHIVE | Per plan step 15: port the dependence metric, then delete once the deck drops the two-stage slides. |
| `docs/{make_coverage_shapley_figure,make_shapley_heatmap,make_token_redshift_figure,make_gap_figure}.py` | 21 KB | ARCHIVE | Each produces exactly one unconsumed figure. `make_gap_figure` shares an input with the live `make_overfit_figure`. |
| `docs/figures/` orphan PNGs (13 files) | 1.35 MB | ARCHIVE | Verified individually. Keep `fig_architecture/cleanup/errors/results/targets`, `fig_interaction_orders`, `fig_modality_interactions`, `fig_v3b_*`, `fig_overfit`. |
| `docs/pipeline.md`, `docs/targets.md` | 19 KB | ARCHIVE or rewrite | See D-note: both are living docs and both are now wrong in load-bearing ways (DR1 counts, wrong error-mode default, V1 architecture, wrong account, one broken image ref). Retiring pipeline.md also retires `make_pipeline_plots.py` and its five figures. |
| `docs/figures/data_counts.csv` | 132 B | ARCHIVE | Regenerate from DR2 (plan step 34); current deck understates the sample. |
| `~/.claude/plans/curious-tinkering-fern.md` | 6.4 KB | ARCHIVE | CLAUDE.md's "plan of record" pointer chain ends in a stale doc. |
| `netscratch outputs/{mt-v2-accum,mt-v3b-8head,mt-v3-lrfix-35416432,mt-v4-sfr}` | 720 MB | **KEEP, back up first** | Only copies, on purge-eligible netscratch. `mt-v3-lrfix` is the quoted "current best"; `mt-v4-sfr` is almost certainly the deck's results figure. |
| `netscratch outputs/p1-dr2-37257713` | 1.1 GB | **KEEP, re-back-up** | Only DR2-trained model; 4 live deck figures. Its home backup predates `poststruct_allmod/noWISE`. |
| `data/dr2/poisoned_shards_backup/` | 17 MB | KEEP | The live refetch is ~47% done. Wait for 4,117/4,117 with 0 retries, then diff sizes. |
| `results/` (paper-era) | 220 KB | KEEP | Published reference values cited by README.md; `baseline_lines_only_clean.csv` is a `build_deck.sh` default. |
| `summer2026/predicting_xray/` | 42 MB | KEEP | Saliency side-project's own workspace; NOTES.md is the only record of the tokenizer measurements. |
| `astroai/{spark,old,platonic_universe}`, `aion_project/{runs,checkpoints,workshop_paper_*}` | 224 GB | OUT OF SCOPE | Other projects. `aion_project/shareable_aion_flow/data/raw/` (13 GB) is **live** for this project. |

Two fetchers are running right now (`fetch_desi_spectra.py`, `fetch_ls_cutouts.py`, under a watchdog) writing into `data/dr2/`. Nothing under `data/dr2/` should be touched until both finish.

---

## 5. WHAT COULD STILL GO WRONG

1. **The masking fix is a no-op if an AION token key is renamed.** `TOKEN_KEYS_BY_MODALITY` is the only place those literals live, `aion` is not installed in this checkout, and every mask loop skips silently on `token_key not in token_dict`. The existence of `_spec_key` (`:952-956`) is evidence the codebase already does not trust the spectrum key name. Step 5's coverage assertion is the only thing standing between you and reproducing today's bug with better comments.

2. **Run A's conditioner can fail silently end to end.** Even with `cond_state_dict` saved and `context_dim` persisted, nothing forces `posterior_structure` to reconstruct the *same* standardizers. A mismatch produces a finite, plausible-looking correlation. Mitigation: store a fingerprint (e.g. the CLS-context hash on a fixed 32-row probe batch) in the checkpoint and assert it at load. This is the single highest-consequence residual risk in the plan.

3. **Conditioning on CIGALE M* while predicting CIGALE SFR hands the model one side of that SED fit's internal degeneracy.** It does not mechanically induce r(Lx, SFR) — Lx is from eROSITA — but the SFR posterior *width* becomes partly a statement about the fit rather than about star formation, and width is what the correlation is normalised by. The GALAXY arm that has to carry the claim is 161 val / 171 test sources before any reliability or modality cut.

4. **The selection function is four stacked non-random cuts, not one.** CIGALE fit success (8,464 excluded), the reliability cut (differentially hitting galaxies, D1), the DET_LIKE gate, and `has_image`. Run A's QSO negative control is only interpretable if all four are reported as one selection statement. If they are reported as four footnotes the result is not defensible.

5. **`sample="common"` makes Run B's headline IG a ~22%-of-sample number on a subsample nobody has shown is random.** Cutouts arrive in whatever order the fetcher works through, which may correlate with sky position, depth or magnitude. Check COMMON against the full sample in z, magnitude and spectype before quoting any image delta. A cheaper alternative exists: an S/Z/W-only common sample (~100% of rows) for the 7 image-free combos, plus a separate image-arm table.

6. **Presence-aware combo sampling changes bucket occupancy by roughly 4x on image combos.** The `--bucket-chunk` calibration and the OOM headroom it protects were tuned on the old mix. Under the never-OOM policy this needs a throughput probe before the first long run, not after.

7. **The expansion has three properties the current sample does not, and each breaks something.** 957 rows with z <= 0 (validator assertion, a tokenizer that has never seen a negative z, and the fabricated-`log_lx` clip); 1,909 STARs whose `log_lx` is meaningless; and fibre-position-only astrometry in `new_targets_nway.csv`, which puts two references in one 1-arcsec match. None of these is visible in the current 25,582.

8. **Test churn around named joints is larger than it looks.** `_stub_flows()` sets `flows.joint` directly and every loss/influence test depends on it; four more tests assume `N_HEADS == N_TARGETS + 1` or read `mt.JOINT_PAIR`/`mt.JOINT_IDX`. If they are patched rather than rewritten, the checkpoint-compatibility guarantee the plan rests on is untested, and that failure costs a whole run.

9. **Nothing here re-runs the spec-z audit on the expansion.** The current sample has `match_quality`; the 104,945 do not, and per C2 `p_any` cannot substitute. The expansion therefore carries roughly 1.3% wrong-counterpart contamination that will be invisible in every metric and will be attributed to model error.

10. **`log_ml_flux_1` and `log_lx` are the same photons.** At fixed z, and z is an input, `log_lx` is an affine relabel of the broad-band flux. Run B carrying both marginals trains one target twice and double-counts it in `mean_ig`. Dropping either breaks continuity with V3's headline numbers, which D5 already breaks anyway.
