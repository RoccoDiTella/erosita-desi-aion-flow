# Data

**Rewritten 2026-08-06 against DR2 (eRASS:3).** Everything below marked
*measured* was recomputed on this machine on that date; anything not marked is
either a published catalogue property or is called out as unverified.

This model is trained on an **eROSITA × DESI** matched sample, enriched with
**WISE** photometry and **Legacy Survey** image cutouts. All of the underlying
data are public, but the raw catalogues and the cutout pool are large and are
**not** tracked in this repository. This page is the data contract: what a run
consumes, where each file comes from, and what the current sample actually is.

---

## 0. As-built register — which file every number on this page describes

**Read this before quoting any count from this repo.** The sample size forks.
The rebuilt chain of §7 emits **25,454 / 22,800**; the superseded pre-rebuild
artifacts hold **25,582**. The `_v2` suffix is what distinguishes them, and
`sbatch/_dataset.sh` resolves the two sets under `DATASET=dr2v2` and `dr2`
respectively. Much of the prose below still quotes 25,582 — those passages
describe the superseded row set and are marked where they matter. Say which
row set a number is about before putting it on a slide.

| artifact | path (under `../stanford_deadline/`) | built | rows | split |
|---|---|---|---:|---|
| **target table, live** | `data/dr2/dr2_targets.csv` | 2026-08-06 17:40 | **25,454** (52 cols, 24,686 detuids) | — |
| **sidecar, live** | `data/dr2/targets_sidecar_dr2_v2.csv` | 2026-08-06 17:41 | **25,454** (78 cols) | — |
| **manifest, live** | `data/dr2/manifest_dr2.csv` | 2026-08-06 17:42 | **25,454** (30 cols) | 18,275 / 2,274 / 2,251 |
| **split, live** | `data/dr2/clean_split_dr2_v2.csv` | 2026-08-06 17:41 | **22,800** | 18,275 / 2,274 / 2,251 |
| **staged, live** | `data/dr2/staged_v2/` | 2026-08-07 07:12 | **22,800** (inputs only) | 18,275 / 2,274 / 2,251 |
| superseded sidecar | `data/dr2/targets_sidecar_dr2.csv` | 2026-08-04 16:56 | 25,582 (47 cols) | — |
| superseded split | `data/dr2/clean_split_dr2.csv` | 2026-08-04 17:03 | 25,582 | 20,465 / 2,548 / 2,569 |

The SUPERSEDED pair was measured on 2026-08-06 with `pandas.read_csv`: 25,582
rows in each; the sidecar has 47 columns and carries `cigale_spectype` but
**no** `spectype` and **no** `nway_p_any`; `split.value_counts()` = train
20,465 / val 2,548 / test 2,569; 24,796 unique `ero_detuid`, 773 of them
carrying more than one DESI fibre. `clean_split_dr2.csv` sha256
`04442bab346448ab8448c1df7520a3523b2670af1127ab580ed6d5444ee53fbe`.

**The rebuild has RUN.** 25,454 and 22,800 were projections in
`docs/rebuild_integration_2026-08-06.md` §3 C1 (25,454 after dropping 128
NWAY-secondary fibres; 22,800 after the reliability and spectrum cuts) and are
now measured: the chain of §7 was executed on 2026-08-06/07 and hit both
exactly. Re-measured with `pandas.read_csv` on 2026-08-07 — row counts and
column counts as tabled, 24,686 unique `ero_detuid`, `clean_split_dr2_v2.csv`
sha256 `874503a21b1e7ae11f50af01dc3acecb3f673559b1b33a8d1d44a56c4d7fc5ed`.

The un-suffixed `targets_sidecar_dr2.csv` / `clean_split_dr2.csv` pair is kept
unmodified so an already-published number stays reproducible. It is the only
thing `DATASET=dr2` still resolves, and no new work should use it.

**What the rebuild made stale, and has NOT yet been re-measured.** Everything
keyed to 25,582: the coverage table in §3, the per-band detection counts in §5,
the class counts in §3, `targets.md` Tier 1 and Tier 3, `decisions.md` §11.4 and
§11.5. Those passages describe the superseded row set. They are not marked
individually — that is what this section is for. Re-measure before quoting any
of them.

> **Closed 2026-08-07.** This used to warn that `make_split.py`'s docstring
> wrote `--out data/dr2/clean_split_dr2.csv`, overwriting the live split in
> place so that no file on disk would carry the row set the published numbers
> were measured on. The rebuild writes `_v2` names; `make_split.py` and
> `make_targets_sidecar.py` now say so in their docstrings and say why; and
> `sbatch/_dataset.sh` `DATASET=dr2v2` resolves them. The un-suffixed pair is
> retained unmodified.

---

## 1. What a run consumes

Three files, and the launcher will not guess between them
(`sbatch/_dataset.sh`, `DATASET=dr2v2`):

| role | file | what it carries |
|---|---|---|
| **inputs** | `$AIONFLOW_STAGED/desi_{train,val,test}.hdf5` | DESI spectra + ivar, redshift, WISE `flux_w1/w2/w3`, `griz` cutouts, `desi_targetid`, and the four presence flags. **No labels.** |
| **labels** | `data/dr2/targets_sidecar_dr2_v2.csv` | every regression target, its split-normal error bars, and the per-band detection likelihoods that decide availability |
| **split** | `data/dr2/clean_split_dr2_v2.csv` | `targetid,split` — a row-selection view over the staged file |

**`staged_v2/` writes INPUTS ONLY.** Both problems this used to describe are
gone, and both were the same bug: a second, older copy of the truth sitting
where something might read it.

- The staged HDF5 used to carry a DR1 `log_ml_flux_1`, and
  `build_dataloaders(..., target_name="log_ml_flux_1")` dropped rows where it
  was non-finite — so a DR1 *detection limit* helped select a DR2 sample at
  2.7x less exposure. Staging now writes no label at all, `target_name=None` is
  how a multi-target caller says "select no rows", and `_dataset.sh` gates on
  the labels being ABSENT so a stale reader fails loudly.
- `staged_v2/` holds exactly the 22,800 targetids of `clean_split_dr2_v2.csv`,
  built 2026-08-07 (18,275 / 2,274 / 2,251). The old 24,181-of-25,582 shortfall
  was the pre-rebuild staged copy and no longer applies.

---

## 2. Public data sources

- **SRG/eROSITA-DE DR2 / eRASS:3** — the X-ray catalogue and the source of every
  target. Local copies in `data/dr2/`:
  `eRASS3_Main_v1.3.fits` (2.14 GB) and `eRASSc3_Main_LS10.fits` (1.79 GB, the
  Main catalogue pre-joined to Legacy Survey DR10 with NWAY counterpart
  probabilities), from the public release dated 27 Jul 2026. The `.fits.gz` of
  the second is redundant with the decompressed file and is on the delete list.
- **DESI DR1** — optical spectra and redshifts.
  DESI Collaboration (2025), [arXiv:2503.14745](https://arxiv.org/abs/2503.14745) ·
  <https://data.desi.lbl.gov>
- **DESI Legacy Imaging Surveys DR10** — `griz` cutouts and WISE forced photometry.
  Dey et al. (2019), AJ 157, 168, [doi:10.3847/1538-3881/ab089d](https://doi.org/10.3847/1538-3881/ab089d) ·
  cutout service `https://www.legacysurvey.org/viewer/cutout.fits?ra=<RA>&dec=<DEC>&layer=ls-dr10&pixscale=0.262&bands=griz&size=160`
- **DESI DR1 CIGALE VAC** (`IronPhysProp_v1.2.fits`) — stellar mass, SFR, AGN
  fraction. 7,322,716,800 bytes, 17,149,172 rows, sha256
  `8752fe4e5ce357472fb22e38bc2c8a235efabf311e3e593520ced786eded2cc9`.
  **On disk and checksum-verified (2026-08-06)** at
  `../stanford_deadline/data/vac/IronPhysProp_v1.2.fits`; the 78-column rebuilt
  sidecar was built from it. See
  `docs/data_provenance.md` for the working access route (the documented
  `data.desi.lbl.gov/public/` path 404s; use the `desidata` S3 bucket) and for
  why every CIGALE label in the current sidecar predates it.
- **DESI DR1 black-hole mass VAC** — `VAC_BHmass_338_v1.7.fits`, 109 MB, present.
- **AION** — the frozen foundation model.
  Parker et al. (2025), [arXiv:2510.17960](https://arxiv.org/abs/2510.17960) ·
  `pip install polymathic-aion`, weights `polymathic-ai/aion-base`.

---

## 3. The current sample

**Everything in this section describes `clean_split_dr2.csv` + `targets_sidecar_dr2.csv`
as built 2026-08-04, the "live" rows of the §0 register.** Re-measured
2026-08-06.

**25,582 rows, 24,796 unique `ero_detuid`.** Split 20,465 / 2,548 / 2,569
train/val/test.

That detuid count is the reason the split is not yet leakage-safe: **773
detuids carry more than one DESI fibre**, so a targetid-grouped split can put
the same X-ray photons on both sides of the wall. Plan step 11 replaces it with
a detuid-grouped hash split.

**Two different quantities count this leak, and they are not interchangeable.**
Confusing them is how 768 and 773 both ended up in the tree as "the" number:

| row set | rows | detuids | detuids with >1 fibre | excess rows |
|---|---:|---:|---:|---:|
| `targets_sidecar_dr2.csv`, live, built 2026-08-04 | 25,582 | 24,796 | **773** | **786** |
| `dr2_targets.csv`, rebuilt (**not built**; from `make_split.py`'s docstring) | 25,454 | 24,686 | 756 | 768 |

The live row is *measured* on this machine 2026-08-06; the rebuild row is quoted
from `scripts/make_split.py`'s docstring table and has never been produced here.
*Excess rows* = rows minus detuids, i.e. how many rows a targetid-keyed split
could place apart from a row sharing its photons. `make_split.py` reports both
at run time and writes both into the split provenance JSON
(`n_detuids_multi_fibre` and `n_excess_rows`), so quote that file rather than
this table once the rebuild has run.

### Class: which column is authoritative for what

Both columns exist, they disagree, and the disagreement is not noise. Measured
2026-08-06 on the live 25,582 rows (DESI `spectype` joined from
`data/erosita_desi_dr1_matches_all_properties.csv`, which covers 25,582 of
25,582):

| column | QSO | GALAXY | STAR | unclassified | authoritative for |
|---|---:|---:|---:|---:|---|
| DESI `spectype` | 22,299 | 3,277 | 6 | 0 | **class**: the QSO negative control, the GALAXY science arm, any per-class split |
| `cigale_spectype` (sidecar) | 21,758 | 2,764 | 0 | 1,060 | **CIGALE fit success only**: which rows have an SED fit at all |

`cigale_spectype` is not a second opinion about what a source *is*. It is
CIGALE's own label on the rows CIGALE fitted, so its 1,060 NaNs mean "no fit",
not "unclassified object", and its zero STARs mean the fit was never attempted
on them, not that the sample has none. **Using it to define a GALAXY arm silently
drops 513 galaxies and hides all 6 stars inside the missing 1,060.**

The live sidecar has **no DESI `spectype` column at all** (verified: 47 columns,
`cigale_spectype` present, `spectype` absent), which is why the class arms are
today defined by the wrong column. Plan step 12 carries `spectype` across.

**This switches under you at rebuild time.** `scripts/make_run_packet.py`
resolves the class column in preference order `("spectype", "cigale_spectype")`,
so `by_spectype.csv` changes its class definition — 22,299/3,277/6 instead of
21,758/2,764/1,060 NaN — on the first run after the sidecar gains `spectype`,
with no flag and no error. The packet stamps the resolved name into a
`class_col` column, so the change is *detectable*; check that column before
comparing two `by_spectype.csv` files. `sbatch/posterior_structure.sbatch` still
defaults `GROUP_COL=cigale_spectype` and so will *not* follow, which means the
packet and the posterior report can disagree about what "GALAXY" means in the
same run. That launcher is not owned by this stream; the needed change is
`GROUP_COL="${GROUP_COL:-spectype}"` once the sidecar carries it.

### Label coverage on the clean split

Finite = the catalogue reports a value. Trained = what survives the detection
gate the loader applies (`det_like > DET_LIKE_MIN`, currently 6). Only the
second column is the sample a head is fit on. Both *measured* on the **live
25,582-row** pair; every row of this table moves when §7 runs.

| target | finite | trained |
|---|---:|---:|
| `z` | 25,582 | 25,582 |
| `log_ml_flux_1` | 25,549 | 25,549 |
| `log_lx` | 25,549 | 25,549 |
| `log_flux_p1` | 23,364 | 12,847 |
| `log_flux_p2` | 25,275 | 19,759 |
| `log_flux_p3` | 25,113 | 19,102 |
| `log_flux_p4` | 17,219 | 3,071 |
| `logmstar_cigale` | 19,210 | 19,210 |
| `log_sfr` | 17,144 | 17,144 |
| sSFR (M* and SFR both) | 17,144 | 17,144 |

The 33 rows with a non-finite `log_ml_flux_1` are censored by the sidecar's own
SIG_CAP at build time, not by the detection gate: `det_like_0 > 6` for **all**
25,582 rows (min exactly 6.00, median 31.27, *measured*), because 6 is the
eRASS Main catalogue's own inclusion rule and the sample is drawn from it.

`logmstar_cigale` and `log_sfr` show no gap between finite and trained, but that
is not because they are ungated — it is because
`scripts/make_targets_sidecar.py` applied the DR1 `match_quality.keep` filter
and a 1.0 dex error cut **at build time**, before the value was ever written.
See §5.

---

## 4. DR1 → DR2: a purification, not a relabel

eRASS:3 is roughly 2.7× the eRASS1 exposure over this footprint, and the effect
on our sample is large enough that no DR1 number transfers.

Measured on the **23,283 sources the two clean splits share**, which is the only
like-for-like comparison available:

| quantity | DR1 (eRASS1) | DR2 (eRASS:3) |
|---|---:|---:|
| median `ML_EXP_1` / `ero_exp` (s) | 120.9 | 330.4 |
| median `flux_sig_lo` (dex) | 0.1826 | 0.1181 |

Over the full DR2 sample the median `flux_sig_lo` is **0.1177** and
`flux_sig_hi` is **0.1053**. `sbatch/_dataset.sh` asserts the median lands in
(0.10, 0.14) before a job starts — a property of the photons rather than of a
filename, so a DR1 file renamed to the DR2 name is caught at launch.

**1,917 rows that were in the DR1 clean split are not in the DR2 one, and 2,299
are new.** The 1,917 are not a data loss:

- all 1,917 are `match_class == "correct"` and `keep == True` in
  `match_quality.csv` (*measured*) — nothing was rejected for match quality;
- their DR1 `DET_LIKE_0` median is **7.53**, against **14.12** for the 23,283
  survivors (*measured*) — they are the eRASS1 marginal tail;
- per the rebuild plan, none has an eRASS:3 counterpart within 1 arcsec.

That is the signature of marginal detections that did not reproduce at 2.7×
exposure. Removing them is a **purification**: the sample got deeper and
cleaner at the same time. Any comparison to a DR1-era number is therefore
confounded by depth and by row set at once.

---

## 5. What the selection function actually is

**Five** stacked non-random cuts, not one. Reporting them as five footnotes is
not defensible; this is the list to state together. Cuts 1-4 are already inside
the live 25,582-row pair. Cut 5 is *not*: it is the default of the rebuild's
splitter and will be inside the next artifact.

1. **Crossmatch reliability (spectroscopic).** A naive 5″ nearest-neighbour
   match is ~8% wrong per Salvato et al. (2025) NWAY. `match_quality.csv`
   classes each targetid and `keep = (match_class == "correct")` drops the rest.
2. **CIGALE fit success.** The M*/SFR labels exist only where the SED fit
   converged: 19,210 and 17,144 of 25,582.
3. **X-ray detection.** `det_like > 6` per band. A no-op for the broad band on
   this sample, and a 16-25% cut on P1/P4.
4. **`has_image`.** Cutouts are still downloading — **25,130** files over 10 kB
   in `data/dr2/fits_pool_dr2/` at 2026-08-06 16:51 (*measured*; 25,060 at
   16:42 and 24,357 at 15:00 the same day, i.e. this number moves within the
   hour and is only ever a floor). Any image-bearing modality combination is therefore evaluated
   on a subsample nobody has yet shown is random in z, magnitude or class.
   `build_manifest.py --fits-pool` is what measures `has_image` at build time;
   quote that, not a number from this page.
5. **NWAY counterpart reliability, `p_any > 0.5`.** See below. This is a real
   cut, it is on by default, and it falls ~10× harder on galaxies than on
   quasars.

### Cut 5: `p_any > 0.5`, and what it costs the science arm

**Chosen deliberately at 0.5.** `scripts/make_split.py:130` takes `--min-p-any`
with a default of `0.5`, so running the documented chain of §7 applies it
whether or not the caller names it. It is a **fifth stacked cut**, not a
diagnostic column. (This section previously said the covering `p_any` column
"is not yet wired in". That was true of the artifacts on disk and is no longer
true of the code: the splitter applies it, and `0` is the only way to turn it
off.)

> **Flagged handoff, not this stream's file.** The comment above that argument
> (`make_split.py:123-132`) still reads "0.5 is a placeholder default, not a
> recorded choice" and still says this page "states the selection function as
> four stacked cuts with p_any *not yet wired in*". Both halves are now stale:
> 0.5 **is** the recorded choice (here and in `decisions.md` §11.9), and this
> section states five cuts. A reader who opens the splitter first will conclude
> the threshold is unsettled. The comment needs to be brought in line with the
> decision it implements.

The cut is not free to *run*, either: `make_split.py:185-187` raises
`SystemExit` when `--min-p-any > 0` and the target table has no `nway_p_any`
column, and no file in `data/dr2/` carries that column today (verified
2026-08-06). `scripts/make_dr2_targets.py:56` maps `NWAY_p_any -> nway_p_any`,
so the chain is coherent — but only in the order §7 gives it. Step 4 cannot be
run against the live sidecar.

**The cost is not class-neutral, and that is the whole reason it is documented
here rather than in a changelog.** GALAXY is the science arm and QSO is the
negative control, so a reliability cut that removes an order of magnitude more
galaxies than quasars is a selection *correlated with the estimand*, applied
hardest to the arm that carries the result:

| scope | QSO lost | GALAXY lost | source |
|---|---:|---:|---|
| full sample, `p_any > 0.5` | **1.2%** | **12.5%** | `rebuild_integration_2026-08-06.md` §3 C2 |
| Run A sample (M*+SFR+Lx complete), `p_any > 0.5` | 0.9% | 6.8% | `rebuild_plan_2026-08-06.md` §3 D1 |
| Run A sample, `p_any > 0.8` | 2.1% | 14.6% | `rebuild_plan_2026-08-06.md` §3 D1 |

**Neither pair of percentages was reproducible on this workstation on
2026-08-06**, and they disagree by roughly a factor of two on GALAXY. The reason
is mechanical: no current-sample row carries `nway_p_any` anywhere in
`data/dr2/` (verified — the live sidecar's 47 columns do not include it), so the
cut cannot be measured until `make_dr2_targets.py` carries `NWAY_p_any` over
from `eRASSc3_Main_LS10.fits`. The two rows above are also measured on different
row sets, which is enough to explain part of the gap but has not been shown to
explain all of it. **Re-measure both on the rebuilt target table before either
number is quoted in a draft**; `make_split.py` prints the per-class cost at run
time and writes it to `p_any_cost_by_spectype` in the split provenance JSON,
which is the number to trust.

One further trap in that provenance record: the per-class breakdown keys on the
`spectype` column of the target table, so it reports the *DESI* classes of §3.
If the table ever loses `spectype`, the breakdown silently disappears rather
than falling back to `cigale_spectype` — which is the right behaviour, because a
CIGALE-class breakdown of this cut would be a different statement.

### Two things about `p_any` that are easy to get wrong

`NWAY_p_any` is P(this X-ray source has **any** LS10 counterpart). It does
**not** test whether the DESI fibre sits on that counterpart. Cut 5 and cut 1
are orthogonal, not alternatives: 336 of the 345 sources `match_quality` calls
"wrong" have `p_any > 0.5`, median 0.9994. Both are needed, and `make_split.py`
applies both.

Second: `new_targets_nway.csv` (104,945 rows, the expansion) has **zero**
targetid overlap with the current sample — 0 of 25,582 — so its `p_any`
distribution says nothing about the rows we train on, and every threshold ever
quoted from it (97,343 at >0.5, 90,955 at >0.8, 86,463 at >0.9) is a statement
about the expansion. The column that covers both samples in one convention is
`NWAY_p_any` in `eRASSc3_Main_LS10.fits`.

### The keep cut is already inside the CIGALE labels

The sidecar in use today was built with its target universe taken from
`match_quality.loc[keep]`, so **every** `logmstar_cigale`, `log_sfr` and
`log_mbh_*` value in it is `keep == True` by construction. Two consequences:
applying `keep` again to the 17,118-row M*+SFR+Lx sample costs exactly zero rows
(*measured*), and rebuilding for the 104,945-row expansion with that gate in
place would yield **zero** CIGALE labels, because `match_quality` has no rows
for those targetids. Plan step 13 replaces the gate with an explicit
`--universe` argument; the labels in the *current* file carry it either way, so
this stays true of anything trained before the sidecar is rebuilt.

### The load-time sigma gate is gone

`MULTI_TARGETS` used to carry `max_sigma: 1.0`, dropping any label whose mean
split-normal error exceeded 1 dex. Measured against the detection gate at either
threshold, it removes **zero rows in every band**:

| band | finite | + max_sigma | DET>5 | DET>5 & sigma | DET>6 | DET>6 & sigma |
|---|---:|---:|---:|---:|---:|---:|
| P1 | 23,364 | 22,583 | 14,122 | **14,122** | 12,847 | **12,847** |
| P2 | 25,275 | 25,095 | 20,924 | **20,924** | 19,759 | **19,759** |
| P3 | 25,113 | 24,842 | 20,269 | **20,269** | 19,102 | **19,102** |
| P4 | 17,219 | 14,399 | 3,736 | **3,736** | 3,071 | **3,071** |

It is entirely subsumed, and on the CIGALE targets it was a *second* copy of a
cut `make_targets_sidecar.add()` had already applied at build time. It is now
`None` everywhere, so the selection function is one cut per band and can be
stated in a line.

---

## 6. The expansion (not yet trained on)

DR2 offers 104,945 targets our 5″ match never claimed. It is a second pass, not
this one, and it is blocked on four things:

- CIGALE labels: the VAC IS on disk, and coverage on the expansion measures the
  same as on the current sample (76.8% mass / 66.8% SFR on the split);
- ~29% of its spectra and most of its cutouts are still downloading;
- 957 of its rows have `z <= 0` (901 STAR, 56 GALAXY), min −0.00182, which
  breaks the staged validator's `z.min() > 0` assertion and used to fabricate a
  `log_lx` ~7 dex too small via a `np.clip(z, 1e-4, None)`;
- 50 detuids are shared with the current sample **despite zero targetid
  overlap**, so merging under a targetid split re-creates the leak.

It is also substantially fainter: the P2 detection fraction drops from 77.2% to
46.6%, P4 from 12.0% to 2.0%. Per-head numbers will not be comparable across
the two passes.

---

## 7. Rebuilding the sidecar and the split

Paths below are relative to the **workstation** data root
`../stanford_deadline/`, the same root as the §0 register — which is why
`--source-hdf5` reaches into `aion_project/…` while everything else sits under
`data/`. There is no `data/raw/` on this machine (*verified* 2026-08-06). On the
cluster the same file lands flat at `$AIONFLOW_DATA/raw/erosita_desi/`, per the
allowlist in `scripts/fasrc_stage_data.sh`.

**Five steps, and the manifest is built twice.** This page previously listed
three steps and omitted `scripts/build_manifest.py` entirely, which cannot
work: `make_split.py --require-spectrum` reads the manifest, and the manifest's
`split` column is only meaningful once the split exists. Neither can go first,
so the manifest is built once without `--split` and again with it.

```bash
# 1. eRASS:3 -> targets + errors + det_like + NWAY + DESI metadata + ero_detuid
python scripts/make_dr2_targets.py     ...  --out data/dr2/dr2_targets.csv

# 2. + CIGALE, + BH mass -> the label sidecar.
#    `_v2`, like step 4, and for the same reason: the un-suffixed pair is what
#    reproduces the published number and must not be overwritten in place.
python scripts/make_targets_sidecar.py ...  --out data/dr2/targets_sidecar_dr2_v2.csv

# 3. manifest, pass 1: availability only, NO --split yet.
#    --source-hdf5 is REQUIRED for the current sample: that is where its spectra
#    live. --spectra-shards holds the 104,945-row EXPANSION only and shares zero
#    targetids with the current sample, so passing it alone sets has_spectrum
#    False for every row and step 4 then exits "every row was filtered out".
python scripts/build_manifest.py --targets data/dr2/dr2_targets.csv \
    --desi data/erosita_desi_dr1_matches_all_properties.csv \
    --match-quality data/match_quality.csv \
    --fits-pool aion_project/shareable_aion_flow/data/raw/legacysurvey/fits_pool \
    --source-hdf5 aion_project/shareable_aion_flow/data/raw/erosita_desi/erosita_spectra_merged_32k.hdf5 \
    --out data/dr2/manifest_dr2.csv

# 4. detuid-grouped hash split + provenance JSON, gated on having a spectrum
#    (--min-p-any defaults to 0.5: cut 5 of the selection function, see §5)
python scripts/make_split.py --targets data/dr2/dr2_targets.csv \
    --match-quality data/match_quality.csv --min-p-any 0.5 \
    --require-spectrum data/dr2/manifest_dr2.csv \
    --previous data/dr2/clean_split_dr2.csv \
    --out data/dr2/clean_split_dr2_v2.csv

# 5. manifest, pass 2: same command as step 3 plus the split it now knows about
python scripts/build_manifest.py ... --split data/dr2/clean_split_dr2_v2.csv \
    --out data/dr2/manifest_dr2.csv
```

**Why two passes.** A split whose fractions include sources with no spectrum
describes a sample that cannot train — a source with no spectrum has no staged
row at all — so step 4 must see `has_spectrum`, which only step 3 measures.

**Which spectra flag, and why getting it wrong empties the split.**
`build_manifest.py` measures `has_spectrum` against wherever the spectra
actually are, and the two samples keep theirs in different places:
`--source-hdf5` is the merged file holding the **current** sample's spectra,
`--spectra-shards` is the `.npz` shard directory holding the **expansion's**.
The two share zero targetids. Point `--spectra-shards` at
`spectra_dr2_new.shards` for a current-sample manifest and every row gets
`has_spectrum = False`; `build_manifest` warns, and then step 4's
`--require-spectrum` drops the lot and exits with *"every row was filtered
out"*. Pass **both** only when building a merged manifest over both samples.
`build_manifest.py:258` refuses to run if neither is given, so the failure mode
is a wrong flag rather than a forgotten one.
`build_manifest.py` is also the only place `has_wise` is defined correctly
(flux > 0 **and** ivar > 0 per band, not finiteness: LS10 and DESI write a
finite number in every W band for all 25,582 rows, so a finiteness rule would
mark WISE universally present and make the whole WISE ablation inert).

`--out` in step 4 is deliberately **not** `clean_split_dr2.csv`: writing the new
split over the old one destroys the only copy of the row set every published
number was measured on. See the hazard box in §0.

`scripts/make_split.py` supersedes `scripts/make_clean_split.py`, which was
targetid-grouped and seeded. `make_clean_split.py` is **superseded, not
deleted**, and was reachable from `sbatch/prepare_data_paper.sbatch` until
2026-08-06; that call is now a hard refusal, because a targetid-grouped split
puts the same X-ray photons in train and test for 773 detuids (§3).

The split in `clean_split_dr2.csv` today is still the old, targetid-grouped
kind.

A staged smoke set without AION or zuko:

```bash
python -m shareable_aion_flow.main prepare-data --output-dir /tmp/smoke --limit 12 --overwrite
```

Note `--limit N` splits N *evenly* across train/val/test by design, not 80/10/10.
