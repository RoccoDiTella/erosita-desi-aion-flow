# Available and potential targets

What we can predict, where each label comes from, what it costs to obtain, and
whether it carries a per-object uncertainty.

**Rewritten 2026-08-06 against DR2.** Coverage is measured on
`data/dr2/clean_split_dr2.csv` × `data/dr2/targets_sidecar_dr2.csv` **as built
2026-08-04, n = 25,582** — not the DR1 split (n = 25,200) and **not** the row
set the rebuild will produce. Every count marked *measured* was recomputed on
this machine on 2026-08-06; percentages are of 25,582. Older statements that
are now wrong are corrected in place with a visible note rather than deleted.

> **Every n and every percentage on this page goes stale when the rebuild
> runs.** The rebuilt chain is projected to emit 25,454 target rows and a split
> of ~22,800 against today's 25,582 / 20,465-2,548-2,569, and it adds a fifth
> selection cut (`p_any > 0.5`) that is not in these artifacts. Nothing in a
> filename distinguishes the two row sets. The as-built register is
> `docs/DATA.md` §0; the selection function is `docs/DATA.md` §5. Re-measure
> this page against the new files rather than scaling these numbers.

The organising distinction, which explains the whole R² ordering in the results:

| group | measured by | is it an input? |
|---|---|---|
| **A. X-ray** | eROSITA eRASS:3 | **no** — an independent instrument |
| **B. Spectroscopic** | DESI spectrum | **yes**, the model sees the spectrum |
| **C. Photometric** | grz + WISE | **yes**, the model sees both |

Only group A is genuine prediction. B and C are closer to *inversion*:
recovering an expensive fit from data the model already holds. Useful, but it
should be described that way. It is why the X-ray flux sits near R² 0.60 while
log M★, fitted from the inputs, reaches 0.74 in the same run (job 35416432; both
DR1-era, see the caveat in `pipeline.md` §9).

---

## Tier 1 — trained today

`finite` = the catalogue reports a value. `trained` = what survives the
detection gate the loader applies (`det_like > DET_LIKE_MIN`, 6). Only the
second is the sample a head is fit on. All *measured* on the DR2 clean split.

| target | group | definition | finite | trained | error |
|---|---|---|---:|---:|---|
| `log_ml_flux_1` | A | log10 ML_FLUX_1, 0.2–2.3 keV | 25,549 | 25,549 | yes, asymmetric |
| `log_lx` | A+z | log10(4π D_L² · F), Planck18 | 25,549 | 25,549 | yes, inherits flux |
| `log_flux_p1` | A | band flux | 23,364 | **12,847** | yes, asymmetric |
| `log_flux_p2` | A | band flux, 0.5–1.0 keV | 25,275 | **19,759** | yes, asymmetric |
| `log_flux_p3` | A | band flux, 1.0–2.0 keV | 25,113 | **19,102** | yes, asymmetric |
| `log_flux_p4` | A | band flux, 2.0–5.0 keV | 17,219 | **3,071** | yes, asymmetric |
| `logmstar_cigale` | B+C | CIGALE stellar mass | 19,210 | 19,210 | yes, `LOGM_ERR` |
| `log_sfr` | C | CIGALE `LOGSFR`, 10 Myr average | 17,144 | 17,144 | yes, `LOGSFR_ERR` |
| `log_mbh_pan25` | B | DR1 qmassiron VAC, MgII, iron-corrected | 9,465 | 9,465 | yes |
| `log_mbh_vo09` | B | same VAC, Vestergaard & Osmer 2009 | 9,459 | 9,459 | yes |
| HR32 | A | implied from a (P2,P3) joint, never trained | — | — | derived |

**`logmstar` (FastSpecFit) is retired as a target.** It is kept as an entry in
`_ALL_TARGETS` purely so name-derived head sets keep every pre-2026-08
checkpoint loadable, and is dropped per run with `--drop-heads logmstar`.
FastSpecFit has **no AGN component**, so on an 87%-QSO sample it attributes the
accretion-disk continuum to stars. CIGALE fits the AGN explicitly, and taking
mass and SFR from the *same* fit is also what makes sSFR well defined.
*(This page previously listed `logmstar` as a Tier 1 target with n = 25,200 and
a fabricated 0.2/0.3 dex class-level error floor.)*

**The band gap is the story of this table.** P4 has 17,219 catalogue values and
3,071 detections. The 14,148 difference are not missing measurements — they are
upper limits, and feeding them as if they were measurements is the likeliest
reason the faint-band σ's look overestimated. They are left NaN, i.e. genuinely
absent, until a censored likelihood exists.

---

## Tier 2 — CIGALE VAC columns, one download away

**Superseded 2026-08-07: the CIGALE VAC IS on this machine.** A correction dated
2026-08-06 said it was not, on a `find` that missed it; it was downloaded and
checksum-verified later the same day to
`../stanford_deadline/data/vac/IronPhysProp_v1.2.fits` (7,322,716,800 bytes,
17,149,172 rows, published sha256 `8752fe4e…` matched), and the 78-column
rebuilt sidecar was built from it. See `docs/data_provenance.md`. Everything below
is therefore a **projection from the columns declared in its FITS header**,
except where a value already reached us through the existing sidecar.

The sidecar's CIGALE columns were derived before that audit and cover only the
current sample; they cannot be regenerated for the 104,945-row expansion until
the file is on disk.

| target | definition | n (*measured* where present) | error | note |
|---|---|---:|---|---|
| **sSFR** | `LOGSFR − LOGM` | 19,210 as `ref_log_ssfr` | partial — see below | **No DESI VAC publishes sSFR as a column** (CIGALE 69 cols, FastSpecFit 817, stellar-mass-emline 420 all checked). Only PROVABGS has one and it overlaps us by 114 sources. So it is derived — but by an exact definition with no free parameter, from two columns of the *same* fit |
| `AGNFRAC` | AGN share of total IR | 24,522 (95.9%) | **no** | directly quantifies the AGN contamination that compromises the SFR/M★ labels |
| `AGNLUM` | AGN bolometric luminosity (W) | 24,522 (95.9%) | **no** | highest coverage of anything on this page |
| `NUVR`, `UV`, `VJ`, `GR` | rest-frame colours | not yet carried | yes | UVJ quiescent / star-forming classification |

**`ref_log_ssfr` is defined on a wider sample than the targets it validates —
this is a defect (*measured* 2026-08-06).** It is finite for 19,210 rows,
exactly the `logmstar_cigale` population, while `log_sfr` is finite for 17,144.
The 2,066-row difference is the `--max-sigma 1.0` gate: `make_targets_sidecar.py`
applies it inside `add()` to the *targets*, but computes `ref_log_ssfr` from the
raw `LOGSFR`/`LOGM` arrays, which never see it. So the reference used to check a
model-implied sSFR includes 2,066 SFR values judged too uncertain to train on.
On the overlap the identity holds exactly (max |ref − (SFR − M*)| = 5.3e-15).

**sSFR error caution.** σ(sSFR) needs the *correlation* between `LOGM_ERR` and
`LOGSFR_ERR`, and CIGALE publishes no posterior covariance even though both come
from one fit. In SED fitting these are typically anti-correlated — attributing
more light to young stars raises SFR and lowers M★ — so treating them as
independent understates the uncertainty. `ref_log_ssfr_sig_indep` is named for
exactly that assumption. Reading the correlation off a joint (M*, SFR) posterior
instead is the point of the joint head.

**The main sequence is NOT flat — corrected 2026-07-29, re-measured on DR2.**
An earlier note here claimed slope 0.076 (R² 0.001) and blamed AGN
contamination. That was a cross-catalogue artifact: it compared **FastSpecFit's**
mass against **CIGALE's** SFR. With both from the same CIGALE fit the relation
is entirely normal. On the DR2 clean split (n = 17,144, *measured*):
**slope +0.730, intercept −5.536, R² 0.303, scatter 0.581 dex** — squarely in
the literature 0.7–1.0 range, and within noise of the DR1 figures (+0.745 /
0.316 / 0.588) since DR2 changed the X-ray labels, not the CIGALE ones. It also
reconciles the independent Hα measurement (R² 0.295 on 536 star-forming
sources), which had been disagreeing with the broken number all along.

**sSFR value caution (this one stands).** Median log sSFR is **−8.14** yr⁻¹
(*measured*) where normal star-forming galaxies sit at −9 to −10 —
implausibly high, consistent with AGN light being counted as star formation.
That is about absolute values, not the relation. The sd of log sSFR is 0.598
dex, *narrower* than log SFR's 0.695, because the M*–SFR correlation suppresses
it.

---

## Tier 3 — DESI DR1 black-hole mass VAC (109 MB, on disk)

`VAC_BHmass_338_v1.7.fits`. 490,648 quasars at z < 1.6, MgII-based,
iron-corrected.

| column | n on DR2 clean split | error | note |
|---|---:|---|---|
| `LOGMASS_DAS_PAN25` → `log_mbh_pan25` | **9,465 (37.0%)** | yes | + SHEN11, LE20, YU23 — **five calibrations**, so the systematic can be measured rather than assumed |
| `LOGMASS_DAS_VO09` → `log_mbh_vo09` | **9,459 (37.0%)** | yes | Vestergaard & Osmer 2009 |
| `L3000_DAS`, `FWHM_DAS` | carried in the DR1 sidecar only | yes | continuum luminosity at 3000 Å, MgII line width |

*(Corrected: this page said 11,734 rows / 46.6% for both mass columns. That was
the DR1 sidecar. The drop is a row-set change, not a catalogue change.)*
Coverage is capped because the catalogue stops at z < 1.6 and requires MgII.
SHEN11/LE20/YU23 correlate with VO09 at 0.99+ — LE20 is exactly 1.0000, a pure
rescaling — so training them would be training the same target several times;
they ride along as columns in the DR1 sidecar and are **not** carried into the
DR2 one.

This is the target that best fits a preference for catalogue values: published,
with errors, and with five independent calibrations.

---

## Tier 4 — FastSpecFit (~70 GB download, not fetched)

`fastspec-iron-main-dark.fits` (43 GB) + `-main-bright.fits` (27 GB) covers 94%
of our sample. Data Lab hosts neither `fastspecfit` nor `cigale`, so there is no
TAP shortcut for either.

| target | projected n | error | note |
|---|---|---|---|
| `VDISP` | **~15–25%, projected** | yes | stellar velocity dispersion. Needs visible stellar absorption, which an AGN continuum swamps; realistically near the GALAXY subset (3,277 by DESI `spectype`, 12.8%) plus low-luminosity AGN. Expect FastSpecFit to *report* far more at the fitting floor with large errors |
| `DN4000` | high | yes | 4000 Å break, stellar age indicator |
| `AGE`, `ZZSUN` | high | yes | light-weighted age, metallicity |
| `ABSMAG_*` | high | yes | rest-frame absolute magnitudes |
| broad-line widths + errors | 87.7% (DR1-era) | yes | our `agngal` copy has `*_sigma` but **no error on sigma** |

---

## Tier 5 — inferred, not catalogued (formula required)

These involve a modelling or calibration choice, so they are *our* numbers
rather than a catalogue's. Counts here are DR1-era unless marked.

| target | formula | n | note |
|---|---|---:|---|
| **Eddington ratio** | λ = L_bol / L_Edd, L_Edd = 1.26e38 (M/M☉) erg s⁻¹, L_bol ≈ 5.15 · L3000 (Richards+2006 at 3000 Å) | ~9,500 on DR2 | both ingredients catalogued **with errors**; only the bolometric correction is a choice. Verify L3000 units before use |
| Hα SFR | log SFR = log L(Hα) − 41.27 (Kennicutt & Evans 2012), dust via Balmer decrement A(Hα) = 5.86 log₁₀[(Hα/Hβ)/2.86] | **536** | independent of any SED fit, which is its value — far too few to train on. Use as a validation set for the CIGALE labels |
| dust A_V | Balmer decrement, as above | ~5,700 | |
| BPT class | [NII]/Hα vs [OIII]/Hβ, Kauffmann+2003 / Kewley+2001 | 5,140 | 536 star-forming, 994 composite, 3,610 AGN |
| Γ (photon index) | deterministic transform of HR under a power law | as HR | no new information beyond HR |
| model-implied sSFR | difference of the log SFR and log M★ **posteriors**, read off the joint | all | this is the version that gets the error correlation right instead of assuming it; `ref_log_ssfr` is the reference it is checked against |

---

## Tier 6 — not available as targets

| quantity | why |
|---|---|
| **N_H / obscuration** | eRASS assumes a fixed Γ = 2 and never fits absorption. The single most valuable missing X-ray quantity, and exactly why HR is used as the proxy |
| X-ray bands P5–P9 | columns exist; DR1-era sources with S/N > 2 numbered 5, 46, 5, 0, 0 — empty. **Not re-measured at eRASS:3 depth**, where 2.7× the exposure could change it |
| source extent (`EXT`, `EXT_LIKE`) | identically zero across the DR1 sample (26,632 rows) — every source point-like. **Unverified on DR2** |
| group membership (`GroupSize`) | ~18% coverage, DR1-era |
| `DET_LIKE_0` | **not a target, but no longer idle.** This page used to dismiss it as "instrumental: it scales with exposure, not with physics". That is true and is precisely why it is now the **availability gate** for every X-ray head, at `DET_LIKE_MIN = 6.0` — the eRASS Main catalogue's own inclusion rule. Predicting it would be predicting the survey; gating on it is stating the selection function |

---

## Where the errors are

| have a per-object error | do not |
|---|---|
| log flux, P1–P4, log Lx (eROSITA LOWERR/UPERR) | `AGNLUM`, `AGNFRAC` |
| `logmstar_cigale`, `log_sfr` (CIGALE, but no covariance between them) | `logmstar` (FastSpecFit) — retired rather than given a fabricated floor |
| `log_mbh_pan25/vo09`, L3000, FWHM (BH VAC) | broad-line widths in our `agngal` copy |

---

## Suggested order

1. **Blocking:** download `IronPhysProp_v1.2.fits` (7.32 GB). Without it there
   are no M*/SFR labels for the 104,945-row expansion at all, and the current
   ones cannot be regenerated.
2. **Free, once it lands:** `AGNFRAC`/`AGNLUM` for the whole sample, and the
   rest-frame colours.
3. **Fix `ref_log_ssfr`** to see the same `--max-sigma` gate as its two parents,
   so the reference and the targets describe one sample.
4. **Then:** Eddington ratio, once the L3000 units are checked.
5. **Only if VDISP is wanted for itself:** the 70 GB FastSpecFit pull. Its
   riders (DN4000, age, metallicity, line-width errors) are worth more than
   VDISP alone, which is likely usable for well under a quarter of the sample.
