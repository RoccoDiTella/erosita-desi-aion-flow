#!/usr/bin/env python
"""Write the clean-split sidecar CSV (targetid -> split) for the runtime view.

The canonical staged data is the paper superset (original manifest, no clean
filter). The cleaned configuration is a row-selection over it: NWAY keep +
dedup + re-split, exactly ``build_clean_manifest`` — this script just persists
that assignment so dataloaders can apply it as a view instead of us keeping a
second 11 GB staged copy.

    python scripts/make_clean_split.py \
        --split-manifest-csv .../aion_tvsplit_manifest.csv \
        --match-quality-csv .../match_quality.csv \
        --output-csv .../clean_split.csv
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from shareable_aion_flow.data_to_aion_embeddings import build_clean_manifest  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split-manifest-csv", type=Path, required=True)
    parser.add_argument("--match-quality-csv", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    manifest = pd.read_csv(args.split_manifest_csv)
    cleaned = build_clean_manifest(manifest, args.match_quality_csv, seed=args.seed)
    out = cleaned[["targetid", "split"]].copy()
    out.to_csv(args.output_csv, index=False)
    counts = out["split"].value_counts().to_dict()
    print(f"wrote {args.output_csv}: {len(out)} targetids, splits={counts}")


if __name__ == "__main__":
    main()
