#!/usr/bin/env python
"""Per-group learning-rate report from a run's history.jsonl.

Reads `move/<group>` = |w_t - w_{t-k}| / |w_t|, the update-to-weight ratio that
`group_metrics` logs for cls_tokens, adapters_low/mid/high, shared_mlp and each
flow. Groups moving orders of magnitude faster or slower than the rest are the
ones that want a different learning rate. This is the evidence that produced the
adapter split in the first place: zero-initialised adapters (|w| ~3) moved ~30x
faster than standard-initialised flows (|w| ~40) on a shared LR, which left the
flows near their initialisation for a whole run.

Two things this deliberately does NOT do:
  * read anything before --skip-steps. `move` is meaningless while |w| is still
    ~0 for the zero-init adapters, and warmup makes the early steps unlike the
    rest of the run.
  * claim precision. The suggested LR assumes the update-to-weight ratio scales
    linearly with LR, which holds for Adam only in the small-update regime. Use
    it to choose the next probe, not as an answer.

    python scripts/lr_report.py <run_dir> [--target 1e-3] [--skip-steps 300]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

# which CLI learning rate drives each diagnostic group
LR_OF_GROUP = {
    "cls_tokens": "lr",
    "shared_mlp": "lr",
    "adapters_low": "adapter_lr",
    "adapters_mid": "adapter_lr",
    "adapters_high": "adapter_lr",
}


def lr_for(group: str, config: dict) -> tuple[str, float | None]:
    key = LR_OF_GROUP.get(group, "head_lr" if group.startswith("flow_") else "lr")
    val = config.get(key)
    if val is None and key == "head_lr":          # empty head_lr means "same as lr"
        key, val = "lr", config.get("lr")
    return key, (float(val) if val is not None else None)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir", type=Path)
    ap.add_argument("--target", type=float, default=1e-3,
                    help="Update-to-weight ratio to aim for (default 1e-3).")
    ap.add_argument("--skip-steps", type=int, default=300,
                    help="Ignore steps before this: warmup, and |w|~0 for zero-init groups.")
    args = ap.parse_args()

    hist = args.run_dir / "history.jsonl"
    if not hist.exists():
        raise SystemExit(f"no history.jsonl in {args.run_dir} "
                         "(runs before the tracking mirror landed only have wandb)")
    rows = [json.loads(line) for line in hist.read_text().splitlines() if line.strip()]
    config = json.loads((args.run_dir / "config.json").read_text()) \
        if (args.run_dir / "config.json").exists() else {}

    groups: dict[str, list[float]] = {}
    for r in rows:
        if (r.get("_step") or 0) < args.skip_steps:
            continue
        for k, v in r.items():
            if k.startswith("move/") and isinstance(v, (int, float)) and np.isfinite(v):
                groups.setdefault(k[len("move/"):], []).append(float(v))

    if not groups:
        raise SystemExit(f"no move/ entries after step {args.skip_steps}; "
                         f"the run logged {len(rows)} rows. Lower --skip-steps.")

    print(f"run: {args.run_dir}")
    print(f"steps used: > {args.skip_steps}   target move: {args.target:.1e}\n")
    print(f"{'group':22s} {'n':>4s} {'median move':>12s} {'spread':>16s} "
          f"{'lr now':>10s} {'suggested':>11s}")
    med = {g: float(np.median(v)) for g, v in groups.items()}
    overall = float(np.median(list(med.values())))
    for g in sorted(med, key=lambda x: -med[x]):
        v = np.array(groups[g])
        key, lr = lr_for(g, config)
        sugg = f"{lr * args.target / med[g]:.2e}" if lr and med[g] > 0 else "n/a"
        print(f"{g:22s} {len(v):>4d} {med[g]:>12.2e} "
              f"{np.percentile(v, 16):>7.1e}-{np.percentile(v, 84):<8.1e} "
              f"{(f'{lr:.1e}' if lr else '?'):>10s} {sugg:>11s}  ({key})")

    # Judge against the TARGET, not the cross-group median. The adapters are
    # split into three depth sub-groups, so they outvote every other part and
    # would drag a median-based reference up until the healthy groups looked
    # frozen. The target is an absolute band and does not care how many
    # sub-groups a part happens to have.
    # A group at EXACTLY zero is not mis-tuned, it is switched off: under
    # --joint-only the marginal flows get no loss term by design. Calling them
    # "frozen, raise the LR" inverts the intent, and dividing by zero movement
    # turns the imbalance ratio into astronomy.
    off = sorted(g for g in med if med[g] == 0.0)
    live = {g: m for g, m in med.items() if m > 0.0}
    if off:
        print(f"\nnot trained (no loss term, as configured): {', '.join(off)}")
    if not live:
        raise SystemExit("every group is frozen; nothing to tune")
    collapsed = {("adapters" if g.startswith("adapters") else g): m for g, m in live.items()}
    print(f"median across live parts (adapters collapsed): {np.median(list(collapsed.values())):.2e}")
    hi = [g for g in live if live[g] > 10 * args.target]
    lo = [g for g in live if live[g] < 0.1 * args.target]
    if hi:
        print(f"  MOVING FAST (>10x target), lower their LR: {', '.join(sorted(hi))}")
    if lo:
        print(f"  NEARLY FROZEN (<0.1x target), raise their LR: {', '.join(sorted(lo))}")
    if not hi and not lo:
        print("  every group within a decade of target: no per-group LR change indicated")
    spread = max(live.values()) / max(min(live.values()), 1e-12)
    print(f"  fastest/slowest ratio: {spread:.0f}x"
          + ("  <- groups are badly out of balance" if spread > 30 else ""))


if __name__ == "__main__":
    main()
