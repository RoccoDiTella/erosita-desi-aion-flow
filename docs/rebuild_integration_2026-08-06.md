# Rebuild integration report (2026-08-06)

Three concurrent implementation streams (eval extraction, data-prep, launcher/docs
hygiene), each adversarially reviewed, then integrated. Ten cross-stream
contradictions found. Nothing has been run on a GPU.

---

## 1. Tests

`python3 -m pytest shareable_aion_flow/tests -q` -> **86 passed, 0 failed, 0 skipped, 0 errors** (8.2 s, run twice).

| file | tests |
|---|---|
| test_validate_staged.py | 19 |
| test_multitarget.py | 18 (lead, unchanged) |
| test_eval_core.py | 13 (new, eval) |
| test_clean_view.py | 8 |
| test_attention_pooling_head.py | 8 |
| test_cls_readonly.py | 5 |
| test_line_shapley.py / test_error_aware_flow.py | 4 each |
| test_train_loops.py | 3 |
| test_normalizing_flow.py / test_evals_schema.py | 2 each |

Nothing failed, so no fix was applied and no stream broke a test it does not own.

## 2. Ownership

Clean. Lead files carry mtime `08-06 14:38`, before any stream started (14:50 onward): `shareable_aion_flow/multitarget.py`, `shareable_aion_flow/tests/test_multitarget.py`. `data_to_aion_embeddings.py` (08-04) and `attention_pooling_head.py` (07-28) are untouched, so **plan step 17 has not landed**.

One modified file belongs to no stream: `scripts/night_monitor.sh`, mtime `08-06 10:11`, which predates all stream work. It is the `ls` to `find` ARG_MAX fix for the live cutout monitor, uncommitted from the earlier download session. Not a violation, but it is riding in the working tree with the rebuild.

Untracked adds all land where expected: `sbatch/_dataset.sh` (hygiene), `scripts/make_split.py` + `scripts/build_manifest.py` (dataprep), `shareable_aion_flow/eval_core.py` + `tests/test_eval_core.py` (eval).

## 3. Contradictions between streams

**C1. The sample size splits in two, and nothing says so. (highest value)**
On-disk artifacts are 25,582 rows (verified: `targets_sidecar_dr2.csv` and `clean_split_dr2.csv` are both 25,583 lines). Every doc, comment and fixture in the tree is written to that number, including files the other two streams wrote: `shareable_aion_flow/eval_core.py:82`, `tests/test_eval_core.py:454`, `scripts/make_run_packet.py:55`, `sbatch/posterior_structure.sbatch:46`, `sbatch/_dataset.sh:57-61`, `docs/DATA.md` (7 sites), `docs/decisions.md`, `docs/targets.md:7`, `docs/pipeline.md:79`.
dataprep's new chain emits **25,454** (128 NWAY-secondary fibres dropped) and a split of **22,800** (18,275/2,274/2,251), against the documented 20,465/2,548/2,569. `scripts/build_manifest.py:10-11` quotes "25,582 of 25,582" in the docstring of a script whose own documented pipeline emits 25,454. The docs are right about today's file and silently wrong about the file the rebuild produces; no marker distinguishes the two row sets.

**C2. p_any: applied by default in one stream, "not yet wired in" in the other.**
`scripts/make_split.py:94` defaults `--min-p-any 0.5`, a fifth selection cut costing 12.5% of GALAXY against 1.2% of QSO. `docs/DATA.md:155-181` states the selection function as **four** stacked cuts and says the covering p_any column "is not yet wired in", and no doc or decision records 0.5 as chosen. Running the documented chain changes the science sample in a way the selection-function section denies.

**C3. Two definitions of "this input exists", and they disagree.**
`build_manifest.wise_presence`: `flux > 0 AND ivar > 0` per band, OR across bands. `has_z`: finite AND `z > 0` AND `zwarn == 0`.
`eval_core.modality_presence:74-96`: wise = `finite AND > 0` (**no ivar term**); z = `torch.isfinite(redshift)` (**no zwarn, no z > 0**).
So `--sample common` admits the 352 zwarn-flagged rows the manifest declares unusable, plus the flux>0/ivar<=0 rows. Nothing consumes the manifest's `has_*` columns except `make_split --require-spectrum`, so the manifest is currently write-only and the eval-side duplicate is what any IG row set is computed from.
Sub-conflict on provenance: `eval_core.py:82` calls the batch wise tensor "the LS10 W-band fluxes"; `build_manifest.py:9-15` explicitly refuses LS10 as a substitute and reads DESI-side `flux_w*`. `data_to_aion_embeddings.py:483` confirms the staged tensor is `flux_w1..w3`, so build_manifest is right and eval_core's comment misattributes the model's own input.

**C4. `--sample` breaks five deck consumers, and the launcher never passes it.**
`--sample both` emits two rows per (head, input_group). No consumer filters on `sample`:
- `docs/make_modality_upset.py:185` `dict(zip(...))` keeps the last row, i.e. native, discarding the common-sample IG that is the point of the flag.
- `docs/make_html_deck.py:59,67` same last-wins dict.
- `docs/make_slides.py:148` one visual row per CSV row, so 30 rows.
- `docs/make_results_figure.py:88` and `docs/make_v3b_figures.py:47` filter `input_group == all-inputs` and now get 2 rows where 1 is expected. These two were not flagged in the eval review.
`n_test` also changed meaning (combo-independent pool, not rows scored) while `make_slides.py:481` and `make_html_deck.py:70,82` still publish it as the n. Separately, `sbatch/eval_multi.sbatch` (hygiene) passes no `--sample`, so the launcher silently runs `native`, which no previously published number used, and `eval_multitarget.py` prints the head set and joint dims but never the resolved sample.

**C5. `hr_implied_target.csv` is now unproducible but still requested.**
hygiene made the HR stage opt-in (`HR_REF_CSV` empty by default) and eval made `hr_from_joint.py` SystemExit on anything but an exactly-2-D (P2,P3) joint; the current joint is 4-D with no P2. So the file is unreachable for the default config, yet `docs/build_deck.sh:70,94` still passes `--hr-csv "$EVAL/hr_implied_target.csv"` unconditionally. Both consumers guard on `.exists()`, so this degrades to a silently missing HR panel rather than a crash. eval's new `hr_joint_summary.csv` has no consumer at all.

**C6. Class column silently switches on the rebuild.**
The rebuilt sidecar is a strict superset of today's (zero current-only columns) and now also carries DESI `spectype`. `scripts/make_run_packet.py:58` prefers `spectype` over `cigale_spectype`, so `by_spectype.csv` changes class definition the moment the sidecar is rebuilt, while `sbatch/posterior_structure.sbatch:49` still defaults `GROUP_COL=cigale_spectype`. decisions.md records the two disagree (DESI has STARs, CIGALE has none). Stamped via `class_col`, so detectable rather than silently wrong.
Also `posterior_structure.sbatch:45-47` claims `--group-csv` "defaults to" the sidecar; line 48 is `GROUP_CSV="${GROUP_CSV:-}"`, so grouping is off and the by-class report is not produced.

**C7. 768 / 773 / 756 multi-fibre detuids.** `make_dr2_targets.py:257` is correct on its own table (768 excess **rows**, 25,454 - 24,686). `make_split.py:6` restates the same 768 as a count of **detections**, which is a different quantity (756 on that table). CLAUDE.md:38, DATA.md:73,245, pipeline.md:82, decisions.md:67 all say 773, correct on the 25,582-row file. Two row sets, one genuine mislabel.

**C8. Step 13 landed without the defect step 33 documented.** `make_targets_sidecar.py:191` `ssfr_bad` still ignores `--max-sigma`, so `ref_log_ssfr` still covers ~2,066 SFR values judged too uncertain to train on. hygiene wrote this up in targets.md and decisions.md §11.4 as a handoff; the rewrite did not take it. Conversely decisions.md §11.4 describes the script in the present tense as gating on `match_quality.keep`, which `--universe` removed.

**C9. Documented chain omits a required step.** `docs/DATA.md:236-244` lists make_dr2_targets, make_targets_sidecar, make_split. `make_split.py`'s own docstring requires the two-pass manifest -> split -> manifest order and `--require-spectrum data/dr2/manifest_dr2.csv`. `build_manifest.py` appears in no doc.

**C10. `build_manifest` cannot define `has_wise` for the expansion.** `wise_presence` requires `flux_w*`/`flux_ivar_w*`; `make_dr2_targets` emits those as `ls10_flux_w*`/`ls10_flux_ivar_w*`, which build_manifest explicitly refuses as substitutes. They arrive only via `--desi`, which exists for the current sample only. Latent, not a today-break.

## 4. Surviving literals and references

**DET_LIKE_MIN: clean.** Zero numeric detection literals in code; zero occurrences of the retired 5.0 anywhere in the tree. `sbatch/_dataset.sh:67` imports the constant; both test files assert against `mt.DET_LIKE_MIN`. Remaining `det_like > 6` strings are prose in DATA.md:93,110,166, targets.md:31,189, pipeline.md:106, decisions.md:1302 and the comment at multitarget.py:40. All legitimate restatements.

**`--inject-samples`: legitimate, inert.** Defined at `main.py:959,1015` (default 50), documented as inert at `train_multi.sbatch:67-68`, which passes `--no-inject` unconditionally at :84. No launcher sets it. Remaining hits are dated history in decisions.md and the corrected CLAUDE.md:40-41.

**`--error-mode`: two stale survivors.** `main.py:883` (single-target, default `none`) is correct. **Not legitimate:** `sbatch/train.sbatch:53` and `sbatch/train_smoke.sbatch:38` still pass `--error-mode "$ERROR_MODE"` under comments at :26 and :23 calling inject "the adopted training mode", the exact error CLAUDE.md:37 was corrected for, in unowned files. Consequence: CLAUDE.md's new invariant "every launcher passes `--no-inject`" is **false**; only `train_multi.sbatch` does. `docs/make_pipeline_plots.py:213` carries it in a figure title.

**`refit_heads`: legitimate-pending.** `scripts/refit_heads.py`, `scripts/refit_compare.py`, `sbatch/refit_heads.sbatch` all survive; step 15 is not in any stream's scope. Note `refit_heads.sbatch:33-34` still defaults to the DR1 trio, has no `--account`, and is `-p gpu_h200`.

**`--joint-only`: legitimate-pending, but re-wired.** `main.py:1035` plus references in lr_report.py, refit_heads.py, make_two_stage_figures.py. hygiene kept the knob live at `train_multi.sbatch:81` (`${JOINT_ONLY:+--joint-only}`) in the launcher it rewrote, though step 15 deletes the flag.

**`finkbeiner_lab`: one deliberate use, one unreconciled contradiction.** The storage root in `.fasrc.env` is deliberate, evidenced from `results/dr2_37257713/`, and marked UNVERIFIED in-file. But `FASRC_NOTES.md:6,59` still reads "Verified 2026-07-21: not yet granted" and "When siag_lab lands: flip .fasrc.env", while `.fasrc.env` now asserts "siag_lab membership landed", CLAUDE.md:48 and pipeline.md:22 assert it as fact, and `scripts/submit.sh:11` passes `-A siag_lab` on **every** submission. FASRC_NOTES.md is unowned and was not reconciled; if the grant has not landed, every job now fails at sbatch where the old config worked.

**DR1 filenames.** Legitimate: `--hr-ref-csv targets_extra.csv` (only hr32 source), CLAUDE.md:44 (explains the retention), test fixtures. Left behind and load-bearing:
- `scripts/fasrc_stage_data.sh:24,28` pushes only the DR1 pair, so a fresh stage-in leaves the cluster unable to satisfy `_dataset.sh`.
- `sbatch/prepare_data*.sbatch`, `refit_heads.sbatch:33-34`, `train.sbatch:29` still default to the DR1 trio and do not source `_dataset.sh`.
- `sbatch/prepare_data_paper.sbatch:40` still calls `make_clean_split.py`, so the targetid-grouped seeded split DATA.md:243 says is superseded stays reachable from a launcher.
- `scripts/eval_multitarget.py:81` and `scripts/hr_from_joint.py:140` still hardcode `target_name="log_ml_flux_1"`, so the DR1 detection limit still selects the eval sample. Both are eval-owned, so the data stream cannot fix them; plan C8 names both.
