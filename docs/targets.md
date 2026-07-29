# Available and potential targets

What we can predict, where each label comes from, what it costs to obtain, and
whether it carries a per-object uncertainty. Coverage is measured on the **clean
split (n = 25,200)** unless stated.

The organising distinction, which explains the whole R² ordering in the results:

| group | measured by | is it an input? |
|---|---|---|
| **A. X-ray** | eROSITA eRASS1 | **no** — an independent instrument |
| **B. Spectroscopic** | DESI spectrum | **yes**, the model sees the spectrum |
| **C. Photometric** | grz + WISE | **yes**, the model sees both |

Only group A is genuine prediction. B and C are closer to *inversion*: recovering
an expensive fit from data the model already holds. Useful, but it should be
described that way. It is why log flux sits at R² 0.61 while log M★ and log SFR,
both fitted from the inputs, reach 0.77 and 0.82.

---

## Tier 1 — trained today

| target | group | definition | n | error |
|---|---|---|---|---|
| `log_ml_flux_1` | A | log10 ML_FLUX_1, 0.2–2.3 keV | 25,200 | yes, asymmetric |
| `log_lx` | A+z | log10(4π D_L² · F), Planck18 | 25,200 | yes, inherits flux |
| `log_flux_p1/p2/p3` | A | band fluxes | 20,019 / 23,803 / 23,151 | yes, asymmetric |
| `logmstar` | B+C | FastSpecFit stellar mass, via `agngal` | 25,200 | **no** — class floor |
| `log_sfr` | C | CIGALE LOGSFR, 10 Myr average | 17,294 | yes, LOGSFR_ERR |
| HR32 | A | implied from the joint (P2,P3) flow | 2,169 test | n/a, derived |

---

## Tier 2 — catalogue values, already on disk, zero download

All from the CIGALE VAC we already pulled (`IronPhysProp_v1.2.fits`, 7.3 GB).

| target | definition | n | error | note |
|---|---|---|---|---|
| **sSFR** | `LOGSFR − LOGM` | 17,294 (68.6%) | partial — see below | **No DESI VAC publishes sSFR as a column** (checked CIGALE 69 cols, FastSpecFit 817, stellar-mass-emline 420). Only PROVABGS has one, and it overlaps us by 114 sources. So this is derived — but by an exact definition with no calibration or free parameter, from two columns of the *same* fit, which puts it in a different class from M_BH or Eddington ratio |
| `AGNLUM` | AGN bolometric luminosity (W) | 24,592 (97.6%) | **no** | highest coverage of anything on this page |
| `AGNFRAC` | AGN share of total IR | 24,592 (97.6%) | **no** | directly quantifies the AGN contamination that compromises the SFR/M★ labels |
| CIGALE `LOGM` | stellar mass, independent of FastSpecFit | ~24,600 | **yes**, LOGM_ERR | would replace the fabricated 0.2/0.3 dex floor on our current mass target |
| `NUVR`, `UV`, `VJ`, `GR` | rest-frame colours | ~24,600 | yes | UVJ quiescent / star-forming classification |

**sSFR error caution.** σ(sSFR) needs the *correlation* between `LOGM_ERR` and
`LOGSFR_ERR`, and CIGALE publishes no posterior covariance. In SED fitting these
are typically anti-correlated — attributing more light to young stars raises SFR
and lowers M★ — so treating them as independent would understate the uncertainty,
not overstate it.

**sSFR value caution.** Median log sSFR is **−8.15** yr⁻¹, whereas normal
star-forming galaxies sit at −9 to −10. Implausibly high, consistent with AGN
light being counted as star formation. Relatedly, the star-forming main sequence
fitted on our training sample has slope **0.076** (R² 0.001) against a literature
0.7–1.0 — flat, because CIGALE's M★ and SFR are both AGN-compromised here. Read
that as a warning about the labels, not a result about galaxies.

---

## Tier 3 — catalogue values, small download (109 MB, already fetched)

**DESI DR1 black-hole mass VAC**, `dr1/vac/dr1/qmassiron/v1.7/VAC_BHmass_338_v1.7.fits`.
490,648 quasars at z < 1.6, MgII-based, iron-corrected.

| column | n (our sample) | error | note |
|---|---|---|---|
| `LOGMASS_DAS_VO09` | 11,734 (46.6%) | yes | Vestergaard & Osmer 2009 |
| `LOGMASS_DAS_PAN25` | 11,734 | yes | + SHEN11, LE20, YU23 — **five calibrations**, so the systematic can be measured rather than assumed |
| `L3000_DAS` | 11,751 | yes | continuum luminosity at 3000 Å |
| `FWHM_DAS` | 11,751 | yes | MgII line width |

log M_BH runs 7.75–9.31 (p5–p95), median error **0.40 dex** with a tail to 5.5 —
so this needs the same error gate we applied to SFR. Coverage is capped at 46.6%
because the catalogue stops at z < 1.6 and requires MgII.

This is the target that best fits a preference for catalogue values: it is
published, has errors, and offers five independent calibrations.

---

## Tier 4 — catalogue values, large download (~70 GB FastSpecFit)

`fastspec-iron-main-dark.fits` (43 GB) + `-main-bright.fits` (27 GB) covers 94%
of our sample. Data Lab does not host FastSpecFit, so there is no TAP shortcut.

| target | projected n | error | note |
|---|---|---|---|
| `VDISP` | **~15–25%, projected** | yes | stellar velocity dispersion. Needs visible stellar absorption, which an AGN continuum swamps; realistically usable near the GALAXY subset (3,490 = 13.8%) plus low-luminosity AGN. Expect FastSpecFit to *report* far more at the fitting floor with large errors |
| `DN4000` | high | yes | 4000 Å break, stellar age indicator |
| `AGE`, `ZZSUN` | high | yes | light-weighted age, metallicity |
| `ABSMAG_*` | high | yes | rest-frame absolute magnitudes |
| broad-line widths + errors | 87.7% | yes | our `agngal` copy has `*_sigma` but **no error on sigma** |

---

## Tier 5 — inferred, not catalogued (formula required)

These involve a modelling or calibration choice, so they are *our* numbers rather
than a catalogue's.

| target | formula | n | note |
|---|---|---|---|
| **Eddington ratio** | λ = L_bol / L_Edd, with L_Edd = 1.26e38 (M/M☉) erg s⁻¹ and L_bol ≈ 5.15 · L3000 (Richards+2006 bolometric correction at 3000 Å) | ~11,700 | both ingredients now catalogued **with errors**; only the bolometric correction is a choice. Verify L3000 units before use |
| Hα SFR | log SFR = log L(Hα) − 41.27 (Kennicutt & Evans 2012), dust via Balmer decrement A(Hα) = 5.86 log₁₀[(Hα/Hβ)/2.86] | **536** | independent of any SED fit, which is its value — but far too few to train on. Use as a validation set for the CIGALE labels |
| dust A_V | Balmer decrement, as above | ~5,700 | |
| BPT class | [NII]/Hα vs [OIII]/Hβ, Kauffmann+2003 / Kewley+2001 | 5,140 | 536 star-forming, 994 composite, 3,610 AGN |
| Γ (photon index) | deterministic transform of HR under a power law | as HR | no new information beyond HR |
| model-implied sSFR | difference of the log SFR and log M★ posteriors | all | valid only if their errors are conditionally independent — exactly what the copula diagnostic measures. Training a *direct* sSFR head as well settles it empirically: if the head is sharper than the difference, the errors were correlated |

---

## Tier 6 — not available

| quantity | why |
|---|---|
| **N_H / obscuration** | eRASS1 assumes a fixed Γ = 2 and never fits absorption. This is the single most valuable missing X-ray quantity, and is exactly why HR is used as the proxy |
| X-ray bands P5–P9 | columns exist, but sources with S/N > 2 number 5, 46, 5, 0, 0 — empty |
| source extent (`EXT`, `EXT_LIKE`) | identically zero for all 26,632 — every source is point-like |
| `DET_LIKE_0` | exists for all, but is instrumental: it scales with exposure, not with physics |
| group membership (`GroupSize`) | only 18% coverage |

---

## Where the errors are

| have a per-object error | do not |
|---|---|
| log flux, P1–P4, log Lx, HR32 (eROSITA) | `logmstar` (current) — a class-level floor stands in |
| log SFR, catalogue sSFR, CIGALE LOGM | AGNLUM, AGNFRAC |
| log M_BH, L3000, FWHM (BH VAC) | broad-line widths in our `agngal` copy |

---

## Suggested order

1. **Now, free:** sSFR (derived by definition), AGNFRAC, AGNLUM, CIGALE LOGM (for its real error).
2. **Now, 109 MB:** log M_BH — catalogued, five calibrations, with errors.
3. **Then:** Eddington ratio, once M_BH is in and the L3000 units are checked.
4. **Only if VDISP is wanted for itself:** the 70 GB FastSpecFit pull. Its riders
   (DN4000, age, metallicity, line-width errors) are worth more than VDISP alone,
   which is likely usable for well under a quarter of the sample.
