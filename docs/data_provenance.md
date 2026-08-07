# Data provenance

Retrieval records for external catalogues this pipeline depends on. One section
per file. Update in place rather than appending a duplicate when re-verifying.

---

## `IronPhysProp_v1.2.fits` — DESI DR1 (Iron) CIGALE physical properties VAC

**Status: DOWNLOADED AND VERIFIED 2026-08-06.** See RETRIEVAL COMPLETED at the
end of this section for the checksum match. The header below records the
characterisation made BEFORE the transfer, which is the point of the record: the
7.32 GB file sat above the threshold set for unattended download, so it was
decided on rather than started by accident.
The file is 7.32 GB, above the 5 GB threshold set for unattended download, so
retrieval was deliberately deferred for a human decision. Every field below
except the *local* sha256 was obtained without transferring the bulk file
(S3 listing, HTTP `HEAD`, HTTP `Range` reads of the FITS header, and the
publisher's own checksum sidecar).

| Field | Value |
|---|---|
| File | `IronPhysProp_v1.2.fits` |
| Version available | **v1.2 — the only version published; no newer release exists** |
| URL (HTTPS) | `https://desidata.s3.amazonaws.com/dr1/vac/dr1/cigale/iron/v1.2/IronPhysProp_v1.2.fits` |
| URL (S3) | `s3://desidata/dr1/vac/dr1/cigale/iron/v1.2/IronPhysProp_v1.2.fits` (`--no-sign-request`) |
| Date located / verified | 2026-08-06 |
| Size | 7,322,716,800 bytes (7.32 GB / 6.82 GiB) |
| Publisher `Last-Modified` | 2025-04-23 12:50:04 GMT |
| Publisher ETag | `7dfcadb776bc105ca4b59283bd78bed8-140` (multipart — not an MD5) |
| sha256 (published) | `8752fe4e5ce357472fb22e38bc2c8a235efabf311e3e593520ced786eded2cc9` |
| sha256 (locally computed) | *pending — file not yet downloaded* |
| Rows | 17,149,172 |
| Intended local path | `/home/roccoditella/astroai/stanford_deadline/data/vac/` (dir does not exist yet) |

### Provenance of the metadata above

The publisher ships a checksum sidecar next to the data, so the sha256 is
authoritative and did not require downloading:

```
https://desidata.s3.amazonaws.com/dr1/vac/dr1/cigale/iron/v1.2/dr1_vac_dr1_cigale_iron_v1.2.sha256sum
```

Its contents (retrieved 2026-08-06):

```
701dd550766c259b88b5e8c007a8e9139f3329cc69d2cc349708c605580ceaa6  IronPhysProp_AfterBurnerv1.2.fits
8752fe4e5ce357472fb22e38bc2c8a235efabf311e3e593520ced786eded2cc9  IronPhysProp_v1.2.fits
```

On download, verify with:

```sh
sha256sum -c dr1_vac_dr1_cigale_iron_v1.2.sha256sum --ignore-missing
```

### HDU structure

Read directly from the file's FITS headers over HTTP `Range` requests.

| HDU | Type | EXTNAME | Rows | Cols | Row bytes |
|---|---|---|---|---|---|
| 0 | PRIMARY | — | — | — | — |
| 1 | BINTABLE | `DATA` | 17,149,172 | 69 | 427 |

Size check — the declared row count fully accounts for the advertised file size,
so the table is the whole file and nothing is truncated:

```
      2,880  primary header            (1 x 2880-byte block)
     17,280  HDU1 header               (6 blocks; 69 cols x ~3 cards)
7,322,696,640  data, 427 x 17,149,172 padded to a block boundary
-------------
7,322,716,800  = Content-Length exactly
```

### Columns of interest (read from the header, not guessed)

| Purpose | Column | Type | Units |
|---|---|---|---|
| Stellar mass | `LOGM` | float64 | log(solMass) |
| **Stellar mass uncertainty** | `LOGM_ERR` | float64 | log(solMass) |
| Star formation rate | `LOGSFR` | float64 | log(solMass/yr), 10 Myr average |
| SFR uncertainty | `LOGSFR_ERR` | float64 | log(solMass/yr) |
| **AGN fraction** | `AGNFRAC` | float64 | — (0 = no AGN, 1 = 100% of total IR from AGN) |
| AGN luminosity | `AGNLUM` | float64 | W |
| AGN viewing angle | `AGNPSY` | float64 | deg (~30 type 1, ~70 type 2) |
| Fit quality | `CHI2` | float64 | reduced chi2 |
| Mass PDF quality flag | `FLAG_MASSPDF` | float64 | Mbest/Mbayes; keep 1/5 <~ x <~ 5 |
| SFR PDF quality flag | `FLAG_SFRPDF` | float64 | SFRbest/SFRbayes; keep 1/5 <~ x <~ 5 |
| Join key | `TARGETID` | int64 | — |

Errors are the standard deviations of the likelihood-weighted PDF. There is **no
published posterior covariance** between `LOGM` and `LOGSFR`, even though both
come from the same fit — relevant to the sSFR error treatment in `targets.md`.

Full 69-column list, in order: `TARGETID`, `SURVEY`, `PROGRAM`, `HEALPIX`,
`SPECTYPE`, `RA`, `DEC`, `RELEASE`, `Z`, `CHI2`, `LOGM`, `LOGM_ERR`, `LOGSFR`,
`LOGSFR_ERR`, `AGNLUM`, `AGNFRAC`, `AGNPSY`, `LNU_{U,G,R,I,Z}`, `NUVR`, `RK`,
`UV`, `VJ`, `GR`, `LNU_{U,G,R,I,Z}_ERR`, `NUVR_ERR`, `RK_ERR`, `UV_ERR`,
`VJ_ERR`, `GR_ERR`, `FLAG_MASSPDF`, `FLAG_SFRPDF`, `FLAGOPTICAL`,
`FLAGINFRARED`, `FLUX_{G,R,Z,W1,W2,W3,W4}`, `FLUX_IVAR_{...}`,
`MW_TRANSMISSION_{...}`, `SNR_{R,G,Z,W1,W2,W3,W4}`.

Note: `TUNIT` for `LOGSFR`/`LOGSFR_ERR` is truncated to `log(solMass` in the FITS
header itself; the documentation gives the correct `log(solMass/yr)`.

### Companion file

`IronPhysProp_AfterBurnerv1.2.fits` — 26,455,680 bytes, 61,904 rows, the **same
69 columns**. It is the same SED fit run on *Afterburner* redshifts instead of
Redrock redshifts. It is a separate small subset, not a patch to the main file.
sha256 `701dd550766c259b88b5e8c007a8e9139f3329cc69d2cc349708c605580ceaa6`.

### Access route (and what does not work)

- **Works:** the anonymous public S3 bucket `desidata`, either via
  `aws s3 --no-sign-request` or plain `curl`/`wget` on the HTTPS endpoint.
  If `aws` errors with `Provided region_name ... doesn't match a supported
  format`, export `AWS_DEFAULT_REGION=us-west-2` first.
- **Does not work:** the documented data URL `https://data.desi.lbl.gov/public/dr1/vac/dr1/cigale/`
  404s from this workstation (that hostname serves GitHub Pages docs).
  The NOIRLab mirror 404s on `/public/` too.
- **Documentation (works):** `https://desi-data.csdc.noirlab.edu/doc/releases/dr1/vac/cigale/`
  carries the full data model, quality-cut recommendations and change log.
- **Astro Data Lab TAP** (`https://datalab.noirlab.edu/tap`) hosts the `desi_dr1`
  schema but **does not** carry the `cigale` or `fastspecfit` tables, so it
  cannot substitute for this download.
- No DESI/NERSC credentials exist on this workstation; all of the above is anonymous.

### Catalogue definition

CIGALE v22.1 (Boquien et al. 2019), delayed SFH with optional exponential burst,
BC03 SSPs, Chabrier (2003) IMF, solar metallicity, Inoue (2011) nebular emission,
Calzetti et al. (2000) attenuation, Dale et al. (2014) dust emission, Fritz et al.
(2006) AGN models. Photometry g, r, z, W1-W4. Cosmology WMAP7. The AGN and galaxy
components are fit simultaneously. Reference paper: Siudek et al. (2024) (EDR).
Contact: Malgorzata Siudek.

Sample construction (per the VAC docs): start from the `zall-pix` redshift
catalogue (28,425,963 rows); cut on `COADD_FIBERSTATUS == 0`,
`ZWARN in (0, 4)`, `ZCAT_PRIMARY == True`, `SPECTYPE in (GALAXY, QSO)`
-> 17,200,298; join to the LS Tractor photometry VAC and drop duplicates
-> **17,149,172**. (The doc names `zall-pix-fuji.fits`, which is the EDR file;
the row counts match the Iron/DR1 catalogue, so this appears to be a doc typo.)

Change log: v1.0 initial; v1.1 added `RA`/`DEC`/`RELEASE`/the four flags/photometry,
renamed `redshift` -> `Z`, fixed types and units; **v1.2 updated `FLAG_MASSPDF`
and `FLAG_SFRPDF`.**

### Consumers in this repo

`scripts/make_targets_sidecar.py` (`--cigale` argument) derives
`logmstar_cigale`, `log_sfr`, `ref_log_ssfr`, `cigale_agnfrac`, `cigale_agnlum`,
`cigale_chi2`, `cigale_flag_masspdf`, `cigale_flag_sfrpdf` and `cigale_spectype`
from this file. It is the sole source of the M*, SFR and sSFR targets.

### Known gotcha

FITS columns are big-endian; pandas raises
`ValueError: Big-endian buffer not supported on little-endian compiler` on any
sort or hash. Convert first: `a.astype(a.dtype.newbyteorder("="))`.

---

## Reproducing the checks above without downloading

```sh
export AWS_DEFAULT_REGION=us-west-2
B=https://desidata.s3.amazonaws.com/dr1/vac/dr1/cigale/iron/v1.2

# listing with exact byte sizes
aws s3 ls s3://desidata/dr1/vac/dr1/cigale/ --recursive --no-sign-request

# size + mtime without transferring the body
curl -sI "$B/IronPhysProp_v1.2.fits" | grep -i 'content-length\|last-modified\|etag'

# publisher checksums and README (a few hundred bytes)
curl -s "$B/dr1_vac_dr1_cigale_iron_v1.2.sha256sum"
curl -s "$B/README.md"

# FITS header only: rows, columns, units -- a few KB via Range request
curl -s -r 0-115199 "$B/IronPhysProp_v1.2.fits" | strings | grep -E '^(TTYPE|NAXIS2|TFIELDS)'
```

To actually fetch it (7.32 GB), single-threaded and resumable:

```sh
mkdir -p /home/roccoditella/astroai/stanford_deadline/data/vac
cd /home/roccoditella/astroai/stanford_deadline/data/vac
nohup wget -c --tries=20 --waitretry=15 \
  "https://desidata.s3.amazonaws.com/dr1/vac/dr1/cigale/iron/v1.2/IronPhysProp_v1.2.fits" \
  > wget_cigale.log 2>&1 &
```

Then verify against the published sha256 recorded above before use.

---

## RETRIEVAL COMPLETED AND VERIFIED, 2026-08-06

Downloaded to `/home/roccoditella/astroai/stanford_deadline/data/vac/IronPhysProp_v1.2.fits`.

* **sha256 MATCHES the publisher's** `dr1_vac_dr1_cigale_iron_v1.2.sha256sum`:
  `8752fe4e5ce357472fb22e38bc2c8a235efabf311e3e593520ced786eded2cc9`
* Opened with `astropy.io.fits`: 2 HDUs, **17,149,172 rows**, 69 columns, matching
  the header-only characterisation made before the download.
* Columns confirmed present: `TARGETID`, `LOGM`, `LOGM_ERR`, `LOGSFR`,
  `LOGSFR_ERR`, `AGNFRAC`.
* 17,149,170 of 17,149,172 rows have finite `LOGM` and `LOGSFR`.
  Median `LOGM` 10.147, median `LOGSFR` 0.138.

The "local sha256 pending" caveat recorded above is now discharged.

**Known limitation, not fixable by a different file:** DESI publishes no
posterior covariance between `LOGM` and `LOGSFR`, only the two marginal standard
deviations, despite both coming from a single CIGALE fit. Any statement about
the M*-SFR error correlation therefore cannot be sourced from this catalogue.
That is the direct motivation for running CIGALE on simulation mocks where the
true values are known.
