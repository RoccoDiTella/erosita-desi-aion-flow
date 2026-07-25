#!/usr/bin/env python
"""Interactive HTML companion to the PDF deck.

Renders the full per-target results: a target selector switches the table
between heads, showing every input combination with R2, RMSE, NLL and info
gain. Self-contained (inline CSS/JS, figures embedded as data URIs) so it
opens from disk or gets shared as one file.

    python docs/make_html_deck.py --mt-metrics multi_test_metrics.csv \
        [--hr-csv hr_implied_target.csv] [--output docs/results.html]
"""

from __future__ import annotations

import argparse
import base64
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

DOCS = Path(__file__).resolve().parent
FIGS = DOCS / "figures"

HEAD_LABELS = {
    "log_ml_flux_1": "log flux (0.2-2.3 keV)",
    "log_lx": "log L_X",
    "logmstar": "log M*",
    "log_flux_p1": "P1 flux (0.2-0.6 keV)",
    "log_flux_p2": "P2 flux (0.6-2.3 keV)",
    "log_flux_p3": "P3 flux (2.3-5.0 keV)",
    "log_flux_p4": "P4 flux (5.0-8.0 keV)",
    "p2xp3_joint": "P2 x P3 joint (2-D)",
}
MODS = [("spectra", "S"), ("z", "Z"), ("wise", "W"), ("image", "I")]


def marker(group: str) -> str:
    parts = set(str(group).split("+"))
    return "".join(short if name in parts else "·" for name, short in MODS)


def embed(path: Path) -> str:
    if not path.exists():
        return ""
    return "data:image/png;base64," + base64.b64encode(path.read_bytes()).decode()


def build_payload(metrics_csv: Path, hr_csv: Path | None) -> dict:
    t = pd.read_csv(metrics_csv)
    payload: dict[str, list[dict]] = {}
    for head, label in HEAD_LABELS.items():
        rows = t[t["head"] == head]
        if not len(rows):
            continue
        rows = rows.sort_values("r2", ascending=False, na_position="last")
        payload[label] = [
            {
                "inputs": marker(r.input_group),
                "n": int(r.n_test),
                "r2": None if pd.isna(r.r2) else round(float(r.r2), 4),
                "rmse": None if pd.isna(r.rmse_dex) else round(float(r.rmse_dex), 4),
                "nll": round(float(r.nll), 4),
                "ig": None if pd.isna(r.info_gain_nats) else round(float(r.info_gain_nats), 4),
                "expig": None if pd.isna(r.info_gain_nats) else round(math.exp(float(r.info_gain_nats)), 3),
            }
            for r in rows.itertuples()
        ]
    if hr_csv and hr_csv.exists():
        h = pd.read_csv(hr_csv)
        ok = h[h["ok"]] if "ok" in h.columns else h

        def block(d, name):
            if len(d) < 30:
                return None
            r2 = 1 - np.sum((d.hr_meas - d.hr_p50) ** 2) / np.sum((d.hr_meas - d.hr_meas.mean()) ** 2)
            return {"inputs": name, "n": int(len(d)),
                    "r2": round(float(r2), 4),
                    "rmse": round(float(np.sqrt(np.mean((d.hr_meas - d.hr_p50) ** 2))), 4),
                    "nll": round(float(-d.log_post.mean()), 4),
                    "ig": round(float(d.info_gain.mean()), 4),
                    "expig": round(float(np.exp(d.info_gain.mean())), 3)}
        rows = [b for b in (block(h, "all measured"), block(ok, "well measured")) if b]
        if rows:
            payload["HR32 (implied, marginalized)"] = rows
    return payload


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
 .wrap {{ overflow-x:auto; }}
 figure {{ margin:1.2rem 0; }}
 figure img {{ width:100%; max-width:100%; height:auto; }}
 figcaption {{ color:var(--muted); font-size:.86rem; margin-top:.3rem; }}
 .note {{ color:var(--muted); font-size:.86rem; }}
</style>
<h1>AION-flow: full results</h1>
<div class="sub">V3b multi-target, test set. Pick a target to see every input combination.
Inputs: <code>S</code> spectra, <code>Z</code> redshift, <code>W</code> WISE, <code>I</code> image.</div>

<h2>Per-target results</h2>
<div class="tabs" id="tabs"></div>
<div class="wrap"><table>
 <thead><tr><th>inputs</th><th>n</th><th>R²</th><th>RMSE (dex)</th><th>NLL</th><th>IG (nats)</th><th>exp(IG)</th></tr></thead>
 <tbody id="rows"></tbody>
</table></div>
<p class="note" id="foot"></p>

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
  rows.innerHTML = d.map((r, i) => `<tr class="${{i === 0 ? 'best' : ''}}">
      <td>${{r.inputs}}</td><td>${{r.n}}</td><td>${{fmt(r.r2)}}</td>
      <td>${{fmt(r.rmse)}}</td><td>${{fmt(r.nll)}}</td>
      <td>${{fmt(r.ig)}}</td><td>${{r.expig === null ? '—' : fmt(r.expig, 2) + '×'}}</td></tr>`).join('');
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
    ap.add_argument("--output", type=Path, default=DOCS / "results.html")
    args = ap.parse_args()

    payload = build_payload(args.mt_metrics, args.hr_csv)
    html = HTML.format(
        payload=json.dumps(payload),
        results_img=embed(FIGS / "fig_v3b_results.png"),
        curves_img=embed(FIGS / "fig_v3b_curves.png"),
    )
    args.output.write_text(html)
    kb = len(html) / 1024
    print(f"wrote {args.output} ({kb:.0f} KB, {len(payload)} targets)")


if __name__ == "__main__":
    main()
