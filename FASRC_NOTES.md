# FASRC operations notes (SLURM)

Working notes for running this repo on the Harvard FASRC cluster (SLURM).

## What we run on TODAY (authoritative)
- **Account `finkbeiner_lab`. Real runs `-p gpu` (A100-**80GB**). Smokes `gpu_test`.
  Storage root `/n/netscratch/finkbeiner_lab/Lab/rditella/aionflow`.** This is
  what `.fasrc.env` sets and what `scripts/submit.sh` passes. Re-verified against
  the scheduler **2026-08-06**:
  - `sacctmgr -nP show assoc user=rditella` → **finkbeiner_lab only**
  - `sbatch --test-only -A finkbeiner_lab -p gpu` → schedules; `gpu_test` → schedules
  - `sbatch --test-only -A siag_lab -p siag_gpu` → *"Invalid account or account/partition combination"*
- Batch sizes are calibrated on 80 GB cards, which is what we are on, so no
  retuning is needed.

## GPU plan (aspirational, per Doug / FASRC — NOT in effect)
- **Destination: `siag_gpu`** — 32× A100-SXM4-**40GB** (4/node), dedicated to siag_lab. Requires **`siag_lab` group membership** (request at portal.rc.fas.harvard.edu; PI approves). Verified 2026-07-21 and **re-verified 2026-08-06: still not granted** — `-A siag_lab` has no association and `-A finkbeiner_lab` is not permitted on the partition.
- **Until then**: smokes on `gpu_test`; real runs on **`gpu`** (A100-80GB) under `finkbeiner_lab`. `gpu_h200` is eligible and has a lighter queue (~124 vs ~440 pending, measured 2026-07-21) but costs 3.2× the fairshare per hour (TRES weight H200 2651.5 vs A100 836.5), so it is not the default for batch-bound jobs.
- When siag_lab lands: flip `.fasrc.env` `FASRC_PART=siag_gpu`, `FASRC_ACCOUNT=siag_lab`, and retune batch size for 40GB cards. **Prove it with `sbatch --test-only` first** — see the trap under "Discovered settings". The values were briefly flipped to siag_lab/siag_gpu on 2026-08-06 on the mistaken belief that the grant had landed; every submission would have failed at sbatch. Reverted the same day.

## Cluster etiquette (we are guests on finkbeiner_lab's allocation)
- **No login-node compute, ever** — arbiter2 kills it, and half-finished work masquerades as done. Everything through `sbatch`.
- **Validate before GPU**: `scripts/validate_staged.py` gates `train.sbatch`; smokes go to `gpu_test` (shared MIG slices, small batch), never to `gpu`.
- **One full GPU run at a time** until we're on our own `siag_gpu` allocation; inspect smoke output before queueing a long run — don't chain a 12 h job blind.
- **Right-size requests** (time/mem/CPUs) so the scheduler can backfill around us; chain stages with `--dependency=afterok` instead of holding resources idle.
- Copy keeper checkpoints/results off **netscratch** (purged!) to holylabs/home after each real run.

## Lessons learned (each cost us real time — don't relearn them)
1. **Idempotency by count, not existence.** An arbiter2-killed unzip left `fits_pool` at 10,355/27,373 files; a `[ -d dir ]` check skipped the repair forever, and staging silently dropped every imageless object → 9,592 rows staged instead of ~22k. Compare on-disk counts to the archive manifest.
2. **Silent row loss is the default failure mode.** Every filtering step must account loudly for what it dropped (`missing_fits_count` was sitting in summary.json all along; the validator now fails on >20% loss).
3. **Trust probes, not bookkeeping.** `sacctmgr show assoc` said we had no account; `sbatch --test-only` proved `finkbeiner_lab` works.
4. **RHEL unzip's zip-bomb heuristic false-positives** on big self-built zip64 archives ("overlapped components"). Verify integrity independently (byte-compare + local `unzip -t`), then `UNZIP_DISABLE_ZIPBOMB_DETECTION=TRUE` for that one extraction.
5. **`gpu_test` slices are shared** — a neighbor held 15 GB of "our" A100 and batch 256 OOM'd. Smoke batch sizes say nothing about `gpu` runs.
6. **Compute nodes have outbound internet** (pip, wandb online both work from jobs).
7. **netscratch small-file I/O is slow** — stage into HDF5, never loose-file datasets.
8. **Keep the socket dumb**: submit/squeue/tail/rsync only, with `ServerAliveInterval=60`. A dropped socket then costs nothing.
9. **Train/eval likelihoods are one contract**: convolve-trained flows must be eval'd through the same measurement kernel or IG is understated (fixed in `evals.py`; `error_mode` read from the checkpoint config).
10. **`gpu_test` QOS: max 2 submitted jobs per user** (`QOSMaxSubmitJobPerUserLimit`, 2 running, ≤8 GPUs/512G aggregate). Plan gpu_test work in pairs; a third submit is rejected outright, so don't build dependency chains assuming more slots.
11. **Check measured timings before launching/right-sizing jobs** (sacct Elapsed):

| Step | Measured | Notes |
|---|---|---|
| env build (idempotent re-run) | 2:15 | pip already cached |
| fits_pool unzip (27,373 files) | ~22 min | netscratch small-file I/O bound |
| staging, 2k rows | 1:16 | smoke subsets |
| staging, 9.6k rows | 3:36 | ~2,700 rows/min |
| staging, 25.2k rows (cleaned) | ~10 min | |
| validator w/ model forward (gpu_test) | ~5 min | |
| V1 smoke, 667 rows × 2 epochs (MIG, bs 32) | 6:16 | incl. 15-combo eval |
| 15-epoch train, 20k rows, bs 448, A100-80 | **V1 1:34 · paper head 1:43** | ~6–7 min/epoch incl. per-combo val; +eval |
| run packet + backup (shared, 4c) | ~2 min | |
| full 50-epoch train (est.) | ~5–6 h | from measured epoch times |

## Workflow (adopted): develop locally, git for code, FASRC only for jobs + data
- **Local (pop-os):** edit + run tests in `stanford_deadline/.venv`; commit to a branch; `git push`. Source of truth. Working branch: `multitarget-clean-errors`.
- **Code → FASRC via git** (repo is PUBLIC → HTTPS clone/pull needs no auth on the cluster):
  `git clone https://github.com/RoccoDiTella/erosita-desi-aion-flow.git && cd erosita-desi-aion-flow && git checkout multitarget-clean-errors` (update: `git pull`). Then copy the gitignored config once (`rsync .fasrc.env` into the cluster repo).
- **Data → FASRC via rsync**: `scripts/fasrc_stage_data.sh` (verify-before-push allowlist).
- **Compute = jobs ONLY. Never run env build / unzip / prepare-data / training on the login node** — arbiter2 kills login-node compute (it did once). Submit to compute nodes:
  1. `scripts/submit.sh sbatch/setup.sbatch` — build conda env + unzip fits_pool
  2. `scripts/submit.sh sbatch/prepare_data.sbatch` — cleaned/deduped/re-split staged HDF5
  3. `scripts/submit.sh sbatch/train_smoke.sbatch` — V1 log_flux smoke (`gpu_test`, `--error-mode none`)
  4. `DATASET=dr2 RUN_TAG=<name> scripts/submit.sh --export=ALL sbatch/train_multi.sbatch` — the multi-target run of record

  *(Corrected 2026-08-06: step 3 said `--error-mode convolve`. The single-target
  launchers now pass a hardcoded `--error-mode none` and expose no knob that can
  select `inject`; `train_multi.sbatch` passes `--no-inject`, which is the same
  decision on the flag that path actually has. See `docs/decisions.md` §3 and
  §11.6. Every launcher that consumes the split/sidecar/staged trio sources
  `sbatch/_dataset.sh`, which requires `DATASET=dr2` and refuses to guess.)*
  The socket is only for `submit.sh` / `squeue` / `tail -f logs/…` — a socket drop cannot hurt a queued/running job.

**Fill the `<...>` placeholders below from the Phase 0 discovery output.**

## Discovered settings (Phase 0, 2026-07-13)
- User `rditella`; login node `holylogin05`. Groups: **finkbeiner_lab**, itc_lab, starfish_users, training, cluster_users.
- Home: `/n/home02/rditella` (95 GB, ~empty) — repo + `.fasrc.env` live here (backed up). Not for conda env / heavy I/O.
- SLURM account (`#SBATCH -A`): **`finkbeiner_lab`** — the only association `sacctmgr` returns for this user (2026-07-13 and again 2026-08-06). `siag_lab` is what FASRC signup guidance pointed at and requires siag_lab group membership, which has not been granted; request via the FASRC portal.
- Partitions: smoke = **`gpu_test`** (12 h, A100 MIG 3g.20gb); real = **`gpu`** (A100-80GB, 3-day). `siag_gpu` (32× A100-40GB, siag_lab dedicated) is the eventual destination. Other fallback: `gpu_requeue` (preemptible, needs checkpointing).
  > **CORRECTED 2026-08-06.** This line used to say `siag_gpu` is "hidden in `sinfo` until you're in the group", and `scripts/fasrc_preflight.sh` used that as its membership test. **That is now false: `siag_gpu` IS visible in `sinfo` (14-day walltime) without siag_lab membership.** Partition visibility settles nothing. The only test that settles it is a dry-run submit, which queues no job and costs seconds:
  > ```bash
  > sbatch --test-only -A siag_lab -p siag_gpu --gpus 1 -t 00:05:00 --wrap true
  > ```
  > On 2026-08-06 that returned *"Invalid account or account/partition combination"*. Preflight step 3 now runs exactly this rather than reading `sinfo`. Do not flip `.fasrc.env` because `sinfo` looks encouraging.
- Modules: **`Miniforge3/26.1.0-fasrc01`** (conda/mamba); CUDA `cuda/12.4.1-fasrc01` available → install **torch cu124 wheel** (wheel ships its own CUDA runtime; module-load cuda optional).
- Storage for env + data + HF cache + outputs: **`/n/netscratch/finkbeiner_lab/Lab/rditella/aionflow`** (fast, PURGED — keepers go to `/n/home02/rditella/aionflow_results`). *(Phase 0 listed this as "TO CONFIRM under siag_lab". It is now confirmed in active use: the checkpoint path recorded in `results/dr2_37257713/posterior_structure_*.json`, written 2026-08-05, is under it, and every backup in `scripts/backup_run.sh` reads from the same tree.)* The SLURM account and the filesystem are separate grants, so this root would not automatically move with an account change; re-run `scripts/fasrc_preflight.sh` if it ever does.
- Login: `ssh rditella@login.rc.fas.harvard.edu` (interactive; password + 2FA). Non-interactive access via ControlMaster socket `~/.ssh/cm-fasrc`.

## What we push to FASRC (EXPLICIT ALLOWLIST — nothing else)
Be deliberate: transfer only these. No blanket rsync of the workspace.

**Code** — the git-tracked clean repo only:
- `git clone https://github.com/RoccoDiTella/erosita-desi-aion-flow` on the cluster (or `git archive`/`rsync` of tracked files). **Never** send `.venv/`, `outputs/`, `runs/`, `__pycache__/`, or local scratch.

**Data** — the minimal bundle (lay out under `<storage>/data/`):
- **`clean_split_dr2.csv` + `targets_sidecar_dr2.csv` (19 MB) → `data/` FLAT**, not `data/dr2/`. These are the labels and the row selection a run consumes; `sbatch/_dataset.sh` resolves `$AIONFLOW_DATA/<name>` and exits the job if either is absent. **Until 2026-08-06 the allowlist pushed only the DR1 pair, so a fresh stage-in produced a cluster where every launcher died at the gate.**
- `erosita_spectra_merged_32k.hdf5` (2.0 GB) → `data/raw/erosita_desi/`
- `erosita_desi_matches_Xray_properties.csv`, `erosita_desi_dr1_matches_all_properties.csv` (116 MB) → `data/raw/erosita_desi/` — the second is the only source of DESI `spectype`/`zwarn` for the current sample, which `build_manifest.py` needs for `has_z`
- `match_quality.csv` (3.6 MB) → `data/` — the spec-z audit, an input to `scripts/make_split.py`
- `targets_extra.csv` (5.3 MB) → `data/` — DR1, retained as the **only** source of `hr32` (`--hr-ref-csv`)
- `targets_sidecar.csv` (12 MB) → `data/` — DR1, **not trainable** (it has none of the nine detection columns the DR2 target spec requires, and `_dataset.sh`'s `dr1` branch is a hard refusal). Kept only to reproduce DR1-era artifacts.
- manifests (`aion_tvsplit_manifest.csv`, `aion_targetids_*` ) (15 MB) → `data/manifests/`
- `fits_pool.zip` (11 GB, the one big file) → unzip to `data/raw/legacysurvey/fits_pool/`
- `survey-bricks-dr10.fits.gz` (13 MB) → `data/raw/legacysurvey/`
- HF cache `models--polymathic-ai--aion-base` (95 MB) → `<storage>/hf_cache/` (or let it auto-download)

`scripts/fasrc_stage_data.sh` is the executable copy of this list and existence-checks every source before it sends anything. If the two lists ever disagree, the script is the one that runs.

**NEVER push:** `all_fits_batches_1_to_4.zip` (5.4 GB dup), the stray root junk file, unrelated HF models (Llama/SDSS/etc.), `aion_project/` research tree, any `.venv/`, `.runpod.env`.

Source paths on the workstation:
- `~/astroai/stanford_deadline/data/dr2/{clean_split_dr2,targets_sidecar_dr2}.csv`
- `~/astroai/stanford_deadline/data/{match_quality,targets_extra,targets_sidecar}.csv`
- `~/astroai/stanford_deadline/aion_project/shareable_aion_flow/data/raw/erosita_desi/*`
- `~/astroai/stanford_deadline/fits_pool.zip`
- `~/astroai/stanford_deadline/aion_project/shareable_aion_flow/data/manifests/*`
- `~/.cache/huggingface/hub/models--polymathic-ai--aion-base`

## Socket (non-interactive access) — keep it alive, run heavy work detached
Open on the workstation (pop-os), with keepalives so it doesn't idle-drop:
```bash
ssh -fN -M -S ~/.ssh/cm-fasrc -o ControlPersist=8h -o ServerAliveInterval=60 -o ServerAliveCountMax=3 rditella@login.rc.fas.harvard.edu
```
The socket is only for launching + monitoring. **Run long steps detached** (`nohup <cmd> >log 2>&1 &` or `sbatch`) so a socket drop can't kill them — a drop once killed an in-flight env build + unzip (exit 255). Verify: `ssh -S ~/.ssh/cm-fasrc -O check <host>`.

## Staging procedure (verify-before-push — use the scripts, not a manual rsync)
1. Open the socket, then **`bash scripts/fasrc_preflight.sh`** — checks the socket, **settles account+partition eligibility with `sbatch --test-only`** (not with `sinfo`; see the correction above), **lists candidate storage** so you set `AIONFLOW_ROOT` right, and **write-tests** it + reports free space. If it says NOT writable, fix `AIONFLOW_ROOT` in `.fasrc.env` and re-run. Nothing transfers and no job is queued.
2. **`bash scripts/fasrc_stage_data.sh --dry-run`**, review, then run without `--dry-run`. It pushes **only the allowlist** to the confirmed path, verifies every source exists first, and **refuses** if the destination isn't writable. Afterward, unzip `fits_pool.zip` on the cluster into `raw/legacysurvey/fits_pool/`.

This is how we guarantee data lands in the correct place — no blanket rsync, destination is write-tested, sources are existence-checked.

## Environment (Phase 2)
```bash
module load Miniforge3/26.1.0-fasrc01           # provides conda/mamba
mamba create -y -p <storage>/envs/aionflow python=3.11
source activate <storage>/envs/aionflow          # or: mamba activate <storage>/envs/aionflow
# CUDA-matched torch wheel (ships its own CUDA runtime; no module-load cuda needed for prebuilt wheels):
pip install torch --index-url https://download.pytorch.org/whl/cu124
pip install -e .            # the repo (pyproject extras: [flow]=zuko, [aion]=polymathic-aion)
pip install "polymathic-aion[torch]" zuko safetensors
export HF_HOME=<storage>/hf_cache
```

## Infra gate (run before ANY training) — `scripts/fasrc_validate.sh`
Imports (`aion, zuko, safetensors, torch`) + AION codec smoke on a staged spectrum + a CUDA tensor smoke. Must pass on a `gpu_test` node. (Ported from the RunPod "imports + codec smoke + CUDA tensor" gate.)

## Run (Phase 3–4)
Always through `scripts/submit.sh`, which injects `-A finkbeiner_lab` from
`.fasrc.env`. Never a bare `sbatch` unless you mean to rely on the `#SBATCH
--account` header.
```bash
# staged superset (image-backed) — CPU job. DR1-era inputs; see the header
# comment in that file and plan step 17, which supersedes it.
scripts/submit.sh sbatch/prepare_data_paper.sbatch
# V1 minimal-head smoke — gpu_test, single-target path
scripts/submit.sh sbatch/train_smoke.sbatch   # --num-queries 1 --num-layers 1 --context-hidden 128
# the multi-target run of record — DATASET is required and has no default
DATASET=dr2 RUN_TAG=mt-dr2 scripts/submit.sh --export=ALL sbatch/train_multi.sbatch
```
Results land on the shared FS under `outputs/<run-id>/` — no RunPod-style pull/backup step.

## Science invariants (carry over, do not break)
- Canonical target `log_ml_flux_1`; `log_lx` is redshift-dominated (secondary).
- **Leakage safety:** the split must be grouped on **`ero_detuid`**, not on `desi_targetid`.
  *(Corrected 2026-08-06: this line said `desi_targetid`.)* 773 detuids carry more
  than one DESI fibre in the live 25,582-row sample (measured), so a
  targetid-grouped split puts the same X-ray photons in train and test.
  `scripts/make_split.py` is the detuid-grouped splitter; `make_clean_split.py`
  is the superseded targetid-grouped one and is called by no launcher.
- Primary metric held-out R²; secondary IG / exp(IG) vs the KDE Scott prior; emission-line baseline = `lines_oii_z`.
- Every run leaves config + metrics + a short note under `outputs/<run-id>/`.

## Architecture variants (Phase 4)
- **V1 (now):** minimal frozen-AION head — `--num-queries 1 --num-layers 1 --context-hidden 128` (head ~7.8M params vs paper ~16.8M).
- **V2 (next):** self-trained attention encoder replacing frozen AION (patchify raw spectrum + sinusoidal wavelength PE + a few self-attention layers → same pooling→MLP→NSF). Tests whether we need AION at all.
