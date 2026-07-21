# FASRC operations notes (SLURM)

Working notes for running this repo on the Harvard FASRC cluster (SLURM).

## GPU plan (authoritative, per Doug / FASRC)
- **Destination: `siag_gpu`** — 32× A100-SXM4-**40GB** (4/node), dedicated to siag_lab. Requires **`siag_lab` group membership** (request at portal.rc.fas.harvard.edu; PI approves). Verified 2026-07-21: not yet granted — `-A siag_lab` has no association and `-A finkbeiner_lab` is not permitted on the partition.
- **Until then**: smokes on `gpu_test`; real runs on **`gpu_h200`** (H200 141GB, eligibility verified via `--test-only`, much lighter queue than `gpu`: ~124 vs ~440 pending) or `gpu` (A100-80GB, busy).
- When siag_lab lands: flip `.fasrc.env` `FASRC_PART=siag_gpu`, `FASRC_ACCOUNT=siag_lab`, and retune batch size for 40GB cards.

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
| full 50-epoch train | **extrapolate from repro smoke first** | never submit blind |

## Workflow (adopted): develop locally, git for code, FASRC only for jobs + data
- **Local (pop-os):** edit + run tests in `stanford_deadline/.venv`; commit to a branch; `git push`. Source of truth. Working branch: `multitarget-clean-errors`.
- **Code → FASRC via git** (repo is PUBLIC → HTTPS clone/pull needs no auth on the cluster):
  `git clone https://github.com/RoccoDiTella/erosita-desi-aion-flow.git && cd erosita-desi-aion-flow && git checkout multitarget-clean-errors` (update: `git pull`). Then copy the gitignored config once (`rsync .fasrc.env` into the cluster repo).
- **Data → FASRC via rsync**: `scripts/fasrc_stage_data.sh` (verify-before-push allowlist).
- **Compute = jobs ONLY. Never run env build / unzip / prepare-data / training on the login node** — arbiter2 kills login-node compute (it did once). Submit to compute nodes:
  1. `scripts/submit.sh sbatch/setup.sbatch` — build conda env + unzip fits_pool
  2. `scripts/submit.sh sbatch/prepare_data.sbatch` — cleaned/deduped/re-split staged HDF5
  3. `scripts/submit.sh sbatch/train_smoke.sbatch` — V1 log_flux smoke (`gpu_test`, `--error-mode convolve`)
  The socket is only for `submit.sh` / `squeue` / `tail -f logs/…` — a socket drop cannot hurt a queued/running job.

**Fill the `<...>` placeholders below from the Phase 0 discovery output.**

## Discovered settings (Phase 0, 2026-07-13)
- User `rditella`; login node `holylogin05`. Groups: **finkbeiner_lab**, itc_lab, starfish_users, training, cluster_users.
- Home: `/n/home02/rditella` (95 GB, ~empty) — repo + `.fasrc.env` live here (backed up). Not for conda env / heavy I/O.
- SLURM account (`#SBATCH -A`): **`siag_lab`** (per FASRC signup guidance). Requires **siag_lab group membership** — request via the FASRC portal if `groups` does not yet list it (the Phase 0 output showed finkbeiner_lab + itc_lab but not siag_lab yet).
- Partitions: smoke = **`gpu_test`** (12 h, A100 MIG 3g.20gb); real = **`siag_gpu`** (32× A100, siag_lab dedicated — hidden in `sinfo` until you're in the group). Fallbacks: `gpu` (3-day) / `gpu_requeue` (preemptible, needs checkpointing).
- Modules: **`Miniforge3/26.1.0-fasrc01`** (conda/mamba); CUDA `cuda/12.4.1-fasrc01` available → install **torch cu124 wheel** (wheel ships its own CUDA runtime; module-load cuda optional).
- Storage for env + data + HF cache + outputs: **TO CONFIRM** writable path under siag_lab. Candidates: `/n/netscratch/siag_lab/.../rditella/aionflow` (fast, purged — preferred for active compute) or `/n/holylabs/LABS/siag_lab/Users/rditella/aionflow` (persistent). Check via socket (`ls -ld /n/netscratch/siag_lab /n/holylabs/LABS/siag_lab`, test mkdir).
- Login: `ssh rditella@login.rc.fas.harvard.edu` (interactive; password + 2FA). Non-interactive access via ControlMaster socket `~/.ssh/cm-fasrc`.

## What we push to FASRC (EXPLICIT ALLOWLIST — nothing else)
Be deliberate: transfer only these. No blanket rsync of the workspace.

**Code** — the git-tracked clean repo only:
- `git clone https://github.com/RoccoDiTella/erosita-desi-aion-flow` on the cluster (or `git archive`/`rsync` of tracked files). **Never** send `.venv/`, `outputs/`, `runs/`, `__pycache__/`, or local scratch.

**Data** — the minimal bundle (lay out under `<storage>/data/`):
- `erosita_spectra_merged_32k.hdf5` (2.0 GB) → `data/raw/erosita_desi/`
- `erosita_desi_matches_Xray_properties.csv`, `erosita_desi_dr1_matches_all_properties.csv` (116 MB) → `data/raw/erosita_desi/`
- manifests (`aion_tvsplit_manifest.csv`, `aion_targetids_*` ) (15 MB) → `data/manifests/`
- `fits_pool.zip` (11 GB, the one big file) → unzip to `data/raw/legacysurvey/fits_pool/`
- `survey-bricks-dr10.fits.gz` (13 MB) → `data/raw/legacysurvey/`
- HF cache `models--polymathic-ai--aion-base` (95 MB) → `<storage>/hf_cache/` (or let it auto-download)

**NEVER push:** `all_fits_batches_1_to_4.zip` (5.4 GB dup), the stray root junk file, unrelated HF models (Llama/SDSS/etc.), `aion_project/` research tree, any `.venv/`, `.runpod.env`.

Source paths on the workstation:
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
1. Open the socket, then **`bash scripts/fasrc_preflight.sh`** — checks the socket, confirms account + `siag_gpu`, **lists candidate storage** so you set `AIONFLOW_ROOT` right, and **write-tests** it + reports free space. If it says NOT writable, fix `AIONFLOW_ROOT` in `.fasrc.env` and re-run. Nothing transfers.
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
```bash
# stage splits (image-backed) — CPU job
sbatch sbatch/prepare_data.sbatch
# V1 minimal-head smoke — gpu_test
sbatch sbatch/train_smoke.sbatch     # V1: --num-queries 1 --num-layers 1 --context-hidden 128
```
Results land on the shared FS under `outputs/<run-id>/` — no RunPod-style pull/backup step.

## Science invariants (carry over, do not break)
- Canonical target `log_ml_flux_1`; `log_lx` is redshift-dominated (secondary).
- **Leakage safety:** splits keyed on unique `desi_targetid`; verify against `aion_tvsplit_manifest.csv`.
- Primary metric held-out R²; secondary IG / exp(IG) vs the KDE Scott prior; emission-line baseline = `lines_oii_z`.
- Every run leaves config + metrics + a short note under `outputs/<run-id>/`.

## Architecture variants (Phase 4)
- **V1 (now):** minimal frozen-AION head — `--num-queries 1 --num-layers 1 --context-hidden 128` (head ~7.8M params vs paper ~16.8M).
- **V2 (next):** self-trained attention encoder replacing frozen AION (patchify raw spectrum + sinusoidal wavelength PE + a few self-attention layers → same pooling→MLP→NSF). Tests whether we need AION at all.
