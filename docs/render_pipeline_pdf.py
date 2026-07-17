#!/usr/bin/env python3
"""Render docs/pipeline.md -> docs/pipeline.pdf (living experiment-pipeline doc).

Keep pipeline.md as the source of truth and re-run this after any pipeline change.
Deps (pure-python): markdown, xhtml2pdf. Install once:
    uv pip install markdown xhtml2pdf   (or: pip install markdown xhtml2pdf)
Usage:
    python docs/render_pipeline_pdf.py
"""
from datetime import datetime, timezone
from pathlib import Path

import markdown
from xhtml2pdf import pisa

HERE = Path(__file__).resolve().parent
SRC = HERE / "pipeline.md"
OUT = HERE / "pipeline.pdf"

CSS = """
<style>
  @page { size: A4; margin: 1.6cm 1.5cm; }
  body { font-family: Helvetica, Arial, sans-serif; font-size: 9.5pt; color: #1a1a1a; line-height: 1.35; }
  h1 { font-size: 17pt; color: #0b2e59; border-bottom: 2px solid #0b2e59; padding-bottom: 3px; }
  h2 { font-size: 12.5pt; color: #0b2e59; margin-top: 14px; border-bottom: 1px solid #c8d4e3; padding-bottom: 2px; }
  h3 { font-size: 10.5pt; color: #244; margin-top: 10px; }
  code, pre { font-family: Courier, monospace; font-size: 8.5pt; background: #f3f4f6; }
  pre { padding: 6px; border: 1px solid #e0e0e0; }
  table { border-collapse: collapse; width: 100%; margin: 6px 0; }
  th, td { border: 1px solid #b8c2cc; padding: 3px 5px; text-align: left; vertical-align: top; font-size: 8.7pt; }
  th { background: #e8eef6; }
  a { color: #0b57d0; text-decoration: none; }
  .footer { color: #888; font-size: 7.5pt; text-align: center; }
</style>
"""


def _link_callback(uri: str, rel: str) -> str:
    """Resolve relative image/asset paths (e.g. figures/foo.png) to absolute."""
    path = (HERE / uri).resolve()
    return str(path)


def main() -> None:
    md_text = SRC.read_text()
    html_body = markdown.markdown(md_text, extensions=["tables", "fenced_code", "sane_lists"])
    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    html = f"{CSS}{html_body}<div class='footer'>Rendered {stamp} from docs/pipeline.md</div>"
    with OUT.open("wb") as f:
        status = pisa.CreatePDF(html, dest=f, link_callback=_link_callback)
    if status.err:
        raise SystemExit(f"PDF render failed with {status.err} error(s)")
    print(f"wrote {OUT} ({OUT.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
