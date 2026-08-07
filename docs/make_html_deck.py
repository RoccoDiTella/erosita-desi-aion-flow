#!/usr/bin/env python
"""Interactive HTML companion to the PDF deck.

Renders the full per-target results: a target selector switches the table
between heads, showing every input combination with R2, RMSE, NLL and info
gain. Self-contained (inline CSS/JS, figures embedded as data URIs) so it
opens from disk or gets shared as one file.

Two columns of every row are the row set it was measured on: `n` is how many
sources that combination was actually scored on and `pool` is the head's whole
labelled test set. They differ, and which one you are reading changes what the
R2 beside them means, so both are printed.

    python docs/make_html_deck.py --mt-metrics multi_test_metrics.csv \
        [--sample common] [--hr-csv hr_implied_target.csv] \
        [--hr-summary-csv hr_joint_summary.csv] [--output docs/results.html]
"""

from __future__ import annotations

import argparse
import base64
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

DOCS = Path(__file__).resolve().parent
sys.path.insert(0, str(DOCS))
from deck_sample import (  # noqa: E402
    add_sample_argument, hr_note_for, sample_meaning, select_sample,
)

FIGS = DOCS / "figures"

HEAD_LABELS = {
    "log_ml_flux_1": "log flux (0.2-2.3 keV)",
    "log_lx": "log L_X",
    "log_sfr": "log SFR",
    "logmstar_cigale": "log M* (CIGALE)",
    "logmstar": "log M*",
    "log_flux_p1": "P1 flux (0.2-0.5 keV)",
    "log_flux_p2": "P2 flux (0.5-1.0 keV)",
    "log_flux_p3": "P3 flux (1.0-2.0 keV)",
    "log_flux_p4": "P4 flux (2.0-5.0 keV)",
    "log_mbh_pan25": "log M_BH (Pan25)",
    "log_mbh_vo09": "log M_BH (VO09)",
    "p2xp3_joint": "P2 x P3 joint (2-D)",
}
MODS = [("spectra", "S"), ("z", "Z"), ("wise", "W"), ("image", "I")]


def head_label(head: str, rows: pd.DataFrame) -> str:
    """Label for a head, including one this file has never heard of.

    The joint head is named by `configure_heads` and is "joint" for anything but
    a restored legacy checkpoint, so a fixed dict keyed on "p2xp3_joint" drops
    the joint row of every new run. Its dimensions come from the table itself.
    """
    if head in HEAD_LABELS:
        return HEAD_LABELS[head]
    dims = [d for d in rows.get("joint_dims", pd.Series(dtype=str)).dropna().unique()
            if str(d).strip()]
    if dims:
        parts = str(dims[0]).split("+")
        return f"{head} ({len(parts)}-D: {', '.join(parts)})"
    return head


def marker(group: str) -> str:
    parts = set(str(group).split("+"))
    return "".join(short if name in parts else "·" for name, short in MODS)


def embed(path: Path) -> str:
    if not path.exists():
        return ""
    return "data:image/png;base64," + base64.b64encode(path.read_bytes()).decode()


def _int_or_none(value) -> int | None:
    return None if value is None or pd.isna(value) else int(value)


def build_payload(metrics_csv: Path, hr_csv: Path | None,
                  compare_csv: Path | None = None, compare_label: str = "V_PAI",
                  sample: str | None = None) -> tuple[dict, str, str | None]:
    """``(payload, sample caption, hardness-absence note)``.

    The sample filter comes first and is not optional. `--sample both` writes one
    `common` and one `native` row per (head, input_group); the dict-of-rows this
    function builds would have kept whichever came last, which is `native`, i.e.
    exactly the row the flag exists to avoid publishing on its own.
    """
    raw = pd.read_csv(metrics_csv)
    t, chosen = select_sample(raw, sample, source=str(metrics_csv))
    # No aggregate row count here: heads have different labelled pools, so a
    # single "n of N" over the whole table would be a number that belongs to no
    # row. The per-row n and pool columns carry it exactly.
    caption = f"sample={chosen} · {sample_meaning(chosen)}"
    payload: dict[str, list[dict]] = {}
    # single-target baseline (same target, same test split) for a like-for-like column
    cmp_r2: dict[str, float] = {}
    if compare_csv and compare_csv.exists():
        c = pd.read_csv(compare_csv)
        cmp_r2 = {marker(r.input_group): float(r.r2) for r in c.itertuples()}
    known = [h for h in HEAD_LABELS if h in set(t["head"])]
    extra = [h for h in dict.fromkeys(t["head"]) if h not in HEAD_LABELS]
    for head in known + extra:
        rows = t[t["head"] == head]
        if not len(rows):
            continue
        label = head_label(head, rows)
        rows = rows.sort_values("r2", ascending=False, na_position="last")
        payload[label] = [
            {
                "inputs": marker(r.input_group),
                "cmp": (round(cmp_r2[marker(r.input_group)], 4)
                        if head == "log_ml_flux_1" and marker(r.input_group) in cmp_r2 else None),
                # n = rows this combination was scored on. n_test is the head's
                # whole labelled test pool and does NOT depend on the combination,
                # so publishing it as "the n" (which this file used to do)
                # overstates every row that was scored on a subset.
                "n": _int_or_none(getattr(r, "n_sample", None)),
                "pool": _int_or_none(getattr(r, "n_test", None)),
                "r2": None if pd.isna(r.r2) else round(float(r.r2), 4),
                "rmse": None if pd.isna(r.rmse_dex) else round(float(r.rmse_dex), 4),
                "nll": None if pd.isna(r.nll) else round(float(r.nll), 4),
                "ig": None if pd.isna(r.info_gain_nats) else round(float(r.info_gain_nats), 4),
                "expig": None if pd.isna(r.info_gain_nats) else round(math.exp(float(r.info_gain_nats)), 3),
            }
            for r in rows.itertuples()
        ]
    if compare_csv and compare_csv.exists():
        c = pd.read_csv(compare_csv).sort_values("r2", ascending=False)
        payload[f"{compare_label}: log flux (single-target)"] = [
            {"inputs": marker(r.input_group), "cmp": None, "n": int(r.n_test),
             "pool": int(r.n_test),
             "r2": round(float(r.r2), 4), "rmse": round(float(r.rmse), 4),
             "nll": round(-float(r.mean_posterior_log_prob_nats), 4),
             "ig": round(float(r.info_gain_nats), 4),
             "expig": round(float(r.exp_info_gain), 3)}
            for r in c.itertuples()]
    hr_note = hr_note_for(raw, hr_csv)
    if hr_csv and hr_csv.exists():
        h = pd.read_csv(hr_csv)
        ok = h[h["ok"]] if "ok" in h.columns else h

        def block(d, name):
            if len(d) < 30:
                return None
            r2 = 1 - np.sum((d.hr_meas - d.hr_p50) ** 2) / np.sum((d.hr_meas - d.hr_meas.mean()) ** 2)
            return {"inputs": name, "cmp": None, "n": int(len(d)), "pool": int(len(h)),
                    "r2": round(float(r2), 4),
                    "rmse": round(float(np.sqrt(np.mean((d.hr_meas - d.hr_p50) ** 2))), 4),
                    "nll": round(float(-d.log_post.mean()), 4),
                    "ig": round(float(d.info_gain.mean()), 4),
                    "expig": round(float(np.exp(d.info_gain.mean())), 3)}
        rows = [b for b in (block(h, "all measured"), block(ok, "well measured")) if b]
        if rows:
            payload["HR32 (implied, marginalized)"] = rows
    return payload, caption, hr_note


def hr_summary_html(path: Path | None) -> str:
    """`hr_joint_summary.csv` as a table: the joint against independent bands.

    This file had no consumer at all until now, which is why it is here: it is
    the only place the eval writes the joint-vs-independent comparison, and that
    comparison IS Run B's headline question. Written by eval_multitarget only
    when --hr-ref-csv is given.
    """
    if path is None or not path.exists():
        return ""
    d = pd.read_csv(path)
    if not len(d):
        return ""
    cols = [c for c in ("subset", "n", "corr_joint", "r2_joint", "halfwidth_joint",
                        "corr_independent", "r2_independent", "halfwidth_independent")
            if c in d.columns]
    def cell(col: str, v) -> str:
        if isinstance(v, str):
            return f"<td>{v}</td>"
        if pd.isna(v):
            return "<td>—</td>"
        if col == "n":
            return f"<td>{int(v):,}</td>"
        # a posterior half-width is a magnitude, a correlation is signed
        return f"<td>{v:.3f}</td>" if col.startswith("halfwidth") else f"<td>{v:+.3f}</td>"

    head = "".join(f"<th>{c}</th>" for c in cols)
    body = ""
    for row in d[cols].itertuples(index=False):
        body += "<tr>" + "".join(cell(c, v) for c, v in zip(cols, row)) + "</tr>"
    have_ind = "corr_independent" in cols
    note = ("Same sources, same per-band flux/rate constants, both drawn from this run: the "
            "joint rows and the independent rows differ only in whether the two bands were "
            "sampled together. That difference is the answer to whether the joint is worth "
            "having." if have_ind else
            "This run has no independent-band baseline (both marginal band heads are needed), "
            "so the joint numbers stand alone.")
    return ("<h2>Hardness ratio: joint vs independent bands</h2>"
            f"<div class=\"wrap\"><table><thead><tr>{head}</tr></thead>"
            f"<tbody>{body}</tbody></table></div>"
            f"<p class=\"note\">{note}</p>")


HTML = """<!doctype html>
<meta charset="utf-8">
<title>AION-flow: full results</title>
<style>
 :root {{ --ink:#1a1a1a; --accent:#0072B2; --muted:#6a6a6a; --line:#e3e3e3; --bg:#fff; }}
 @media (prefers-color-scheme: dark) {{
   :root {{ --ink:#eaeaea; --accent:#4dabdc; --muted:#9a9a9a; --line:#333; --bg:#141414; }}
 }}
 body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Helvetica, Arial, sans-serif;
        margin:0; padding:2.2rem 1.4rem 4rem; color:var(--ink); background:var(--bg);
        max-width:1100px; margin-inline:auto; line-height:1.5; }}
 h1 {{ font-size:1.7rem; margin:0 0 .2rem; }}
 h2 {{ font-size:1.15rem; margin:2.4rem 0 .6rem; border-bottom:2px solid var(--accent);
       padding-bottom:.3rem; }}
 .sub {{ color:var(--muted); margin-bottom:1.6rem; }}
 .tabs {{ display:flex; flex-wrap:wrap; gap:.4rem; margin:.8rem 0 1rem; }}
 .tab {{ padding:.42rem .8rem; border:1px solid var(--line); border-radius:999px;
        background:transparent; color:var(--ink); cursor:pointer; font-size:.9rem; }}
 .tab:hover {{ border-color:var(--accent); }}
 .tab[aria-selected="true"] {{ background:var(--accent); border-color:var(--accent); color:#fff; }}
 table {{ border-collapse:collapse; width:100%; font-size:.92rem; }}
 th, td {{ padding:.45rem .6rem; text-align:right; border-bottom:1px solid var(--line); }}
 th:first-child, td:first-child {{ text-align:left; font-family:ui-monospace, Menlo, Consolas, monospace;
                                   letter-spacing:.18em; }}
 th {{ background:var(--accent); color:#fff; position:sticky; top:0; }}
 tbody tr:hover {{ background:rgba(0,114,178,.07); }}
 .best td {{ font-weight:700; }}
 td.up {{ color:#009E73; font-weight:600; }}
 td.down {{ color:#D55E00; }}
 .wrap {{ overflow-x:auto; }}
 figure {{ margin:1.2rem 0; }}
 figure img {{ width:100%; max-width:100%; height:auto; }}
 figcaption {{ color:var(--muted); font-size:.86rem; margin-top:.3rem; }}
 .note {{ color:var(--muted); font-size:.86rem; }}
</style>
<h1>AION-flow: full results</h1>
<div class="sub">Multi-target, test set. Pick a target to see every input combination.
Inputs: <code>S</code> spectra, <code>Z</code> redshift, <code>W</code> WISE, <code>I</code> image.
The V_PAI column is the paper-head model trained on flux alone, same cleaned data and same test split.<br>
<b>{sample_note}</b><br>
<code>n</code> is the sources this combination was scored on; <code>pool</code> is the head's whole
labelled test set and does not depend on the combination.</div>

<h2>Per-target results</h2>
<div class="tabs" id="tabs"></div>
<div class="wrap"><table>
 <thead><tr><th>inputs</th><th>n</th><th>pool</th><th>R²</th><th>V_PAI R²</th><th>Δ</th><th>RMSE (dex)</th><th>NLL</th><th>IG (nats)</th><th>exp(IG)</th></tr></thead>
 <tbody id="rows"></tbody>
</table></div>
<p class="note" id="foot"></p>
{hr_block}
{hr_summary}

<h2>Summary</h2>
<figure><img src="{results_img}" alt="per-target results"><figcaption>
Point accuracy and posterior sharpness for every head. HR32 is implied: marginalized out of the joint (P2,P3) posterior, never trained.
</figcaption></figure>

<h2>Training diagnostics</h2>
<figure><img src="{curves_img}" alt="per-head loss curves"><figcaption>
Per-head negative log-likelihood. Train is on injected targets (harder by construction); validation is the plain likelihood.
</figcaption></figure>

<script>
const DATA = {payload};
const tabs = document.getElementById('tabs');
const rows = document.getElementById('rows');
const foot = document.getElementById('foot');
const fmt = (v, d=3) => v === null || v === undefined ? '—' : v.toFixed(d);
function render(name) {{
  const d = DATA[name] || [];
  rows.innerHTML = d.map((r, i) => {{
    const dlt = (r.cmp === null || r.cmp === undefined || r.r2 === null) ? null : r.r2 - r.cmp;
    const dcls = dlt === null ? '' : (dlt >= 0 ? 'up' : 'down');
    return `<tr class="${{i === 0 ? 'best' : ''}}">
      <td>${{r.inputs}}</td><td>${{r.n === null || r.n === undefined ? '—' : r.n}}</td>
      <td>${{r.pool === null || r.pool === undefined ? '—' : r.pool}}</td><td>${{fmt(r.r2)}}</td>
      <td>${{fmt(r.cmp)}}</td><td class="${{dcls}}">${{dlt === null ? '—' : (dlt >= 0 ? '+' : '') + dlt.toFixed(3)}}</td>
      <td>${{fmt(r.rmse)}}</td><td>${{fmt(r.nll)}}</td>
      <td>${{fmt(r.ig)}}</td><td>${{r.expig === null ? '—' : fmt(r.expig, 2) + '×'}}</td></tr>`;
  }}).join('');
  foot.textContent = d.length ? `${{d.length}} rows, best first.` : 'no rows for this target.';
  [...tabs.children].forEach(b => b.setAttribute('aria-selected', String(b.textContent === name)));
}}
Object.keys(DATA).forEach((name, i) => {{
  const b = document.createElement('button');
  b.className = 'tab'; b.textContent = name; b.onclick = () => render(name);
  tabs.appendChild(b);
  if (i === 0) render(name);
}});
</script>
"""


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--mt-metrics", type=Path, required=True)
    ap.add_argument("--hr-csv", type=Path, default=None)
    ap.add_argument("--hr-summary-csv", type=Path, default=None,
                    help="hr_joint_summary.csv from the eval: the joint against the "
                         "independent-band baseline, on the same rows")
    ap.add_argument("--compare-metrics", type=Path, default=None,
                    help="Single-target baseline table (V_PAI on flux) for the comparison column.")
    ap.add_argument("--output", type=Path, default=DOCS / "results.html")
    add_sample_argument(ap)
    args = ap.parse_args()

    payload, caption, hr_note = build_payload(
        args.mt_metrics, args.hr_csv, args.compare_metrics, sample=args.sample)
    # An absent hardness panel is stated, not skipped: both consumers used to
    # guard on .exists() and drop it without a word, so an unproducible file and
    # a file nobody asked for looked identical on the page.
    hr_block = "" if hr_note is None else (
        "<h2>Hardness ratio</h2><p class=\"note\">" + hr_note + "</p>")
    html = HTML.format(
        payload=json.dumps(payload),
        sample_note=caption,
        hr_block=hr_block,
        hr_summary=hr_summary_html(args.hr_summary_csv),
        results_img=embed(FIGS / "fig_v3b_results.png"),
        curves_img=embed(FIGS / "fig_v3b_curves.png"),
    )
    args.output.write_text(html)
    kb = len(html) / 1024
    print(f"wrote {args.output} ({kb:.0f} KB, {len(payload)} targets, {caption.split(' · ')[0]})")


if __name__ == "__main__":
    main()
