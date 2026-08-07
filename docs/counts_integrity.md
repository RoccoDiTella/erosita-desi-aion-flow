# APE_CTS integrity: what we are certain of, and what we are not

**Measured 2026-08-07 on all 1,975,540 rows of `eRASS3_Main_v1.3.fits`.** This
page exists because the decision we reached ("drop, do not impute") is only
defensible if the reasoning behind it is written down, and because the claim
"it wrapped" turns out to be two claims of very different strength.

---

## 1. The defect is upstream, and it is declared in the file

`APE_CTS_n` is stored as `TFORM='I'` — the FITS standard's **16-bit signed
integer**, range −32,768 to 32,767 — with `TUNIT='count'`. That is eSASS's
choice in the published catalogue, not an artifact of our reader: we only ever
`fits.open(..., memmap=True)` and read columns, and the file has not been
rewritten since download on 2026-08-04.

It was a per-column decision, not a blanket one. In the same table:

| column | TFORM | type |
|---|---|---|
| `APE_CTS_1`, `APE_CTS_P2` | `I` | int16 |
| `APE_BKG_1` | `E` | float32 |
| `ML_CTS_1` | `E` | float32 |
| `APE_EXP_1` | `E` | float32 |

Presumably the reasoning was that counts in a small aperture fit in 16 bits.
For 99.998% of sources they do.

---

## 2. What we ARE certain of

**A negative `APE_CTS` is corrupt.** This needs no model and no assumption: a
count of photons in an aperture is a non-negative integer by construction. There
is no astrophysics that produces −32,540 photons. Whatever produced that value,
the value is not a measurement.

**Integer overflow is overwhelmingly the likely mechanism.** Every negative row
sits at the **99.99th–100th percentile of `ML_FLUX`** in its band. Band 1's 20
negative rows have median `ML_CTS` ≈ 55,251 net counts and median `ML_RATE`
72.3 ct/s. A source with ~55k counts in a 16-bit field is precisely the
condition for a wrap, and the affected population is exactly the population that
should be affected. Nothing else about these rows is anomalous.

Counts of affected rows, by band, using `APE_CTS < 0`:

| band | negative rows |
|---|---|
| 1 | 20 |
| P1 | 2 |
| P2 | 5 |
| P3 | 8 |
| P4 | 4 |

**Zero of them are in our 25,454-row sample.**

---

## 3. What we are NOT certain of, and why it kills recovery

The obvious repair is exact: a single wrap displaces the value by exactly 2^16,
so `true = stored + k * 65536` for integer k, and k could be read off by
comparing against the fitted counts. **This was tried and it fails.**

Predicting `APE_CTS ~ ML_CTS * ML_EEF + APE_BKG` and testing k = 1, 2, 3:

| band | suspect rows | rows recovering to within 5 sigma at any k |
|---|---|---|
| 1 | 72 | **0** |
| P1 | 4 | **0** |
| P2 | 34 | **0** |
| P3 | 19 | **0** |
| P4 | 6 | 1 |

Post-unwrap residuals stay at 200–500 sigma. The reason is not that the wrap
hypothesis is wrong — it is that **the reference model is untrustworthy for
exactly this population**. `ML_CTS` is PSF-fitted and assumes a point source;
`APE_CTS` is a fixed-aperture sum. The brightest objects in eRASS include
galaxy clusters and piled-up sources, for which those two quantities legitimately
disagree by tens of thousands of counts. So the test cannot separate "wrapped"
from "extended", and k is not identifiable.

**Therefore we can be confident the value is corrupt and still be unable to
recover it.** Those are different claims and only the first one holds.

---

## 4. The decision, and why not the alternatives

**DROP the affected band for the affected row. Keep the row's other bands.
Never impute.**

- *Why not impute 0?* Because the measurement says these are the brightest
  sources in the sky. Writing 0 would state that the most luminous objects in
  the catalogue are non-detections — the largest possible error, on the rows
  with the most leverage.
- *Why not impute 32,767 (the int16 maximum)?* This was proposed and is
  reasonable in spirit: if we knew it wrapped once, the true value is
  `stored + 65536`, and clamping would be biased low but bounded. The problem is
  the "if". Section 3 shows k is not identifiable, and for `stored = -5,854` the
  single-wrap truth is 59,682 — clamping to 32,767 is off by 45%, not a little.
  An imputed value is a fabricated measurement, and here it would be fabricated
  precisely where we understand the source least.
- *Why not drop the whole row?* The other bands are unaffected. Discarding a
  perfectly good P2 count because P4 wrapped would throw away real data and, on
  a brightness-selected population, would itself be a selection effect.

**Second, weaker cut, kept separate on purpose.** A double wrap comes back
positive and looks ordinary; the sign test cannot see it. Flagging rows where
`(APE_CTS - ML_CTS*EEF - BKG) / sqrt(expected) < -100` catches 135 rows across
all bands, 96 of which have a *positive*, innocent-looking `APE_CTS`. But by the
argument in section 3 this test also fires on genuinely extended sources, so it
is **suspicion, not proof**. It is applied because 135 of 1,975,540 rows
(0.007%) is a negligible loss and a fabricated bright measurement is not.

---

## 5. Negative `APE_BKG` is a DIFFERENT defect — check before conflating

20 band-1 rows have `APE_BKG < 0`, range −13.498 to −0.003 against a typical
positive background of 2.52. **They share zero rows with the negative-count
population** (measured: overlap = 0), which is easy to assume and wrong.

Background is not a count — it is a smooth-map fit evaluated in the aperture,
and a fit can undershoot below zero near zero. That IS the "consistent with
approximately zero" case, so these are **clipped to 0 and the row is kept**.
Clipping is safe here for the reason imputation is unsafe above: a background
that should be ~0 and reads slightly negative is a rounding artifact, whereas a
count that should be ~55,000 and reads −32,540 has lost information.

---

## 6. What this means for Run B

`N ~ Poisson(lambda*t + B)` requires `N >= 0` and `lambda*t + B > 0`. Both cuts
above run at load time, before the likelihood sees anything, and the build
prints the affected counts so the number cannot drift silently as the sample
grows. **It grows with the sample**: zero rows are affected today, but the
merged rebuild pulls in brighter sources, which is exactly when a wrap starts to
matter.

Worth reporting upstream to the eROSITA team. It does not block us.
