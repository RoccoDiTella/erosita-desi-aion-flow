#!/usr/bin/env python
"""Compare phase-2 refits across phase-1 bodies, and pick one.

Phase 1 writes a body snapshot every few epochs; phase 2 refits every head on
each. The body should be chosen by POST-REFIT validation, which is not the epoch
of the joint's own val minimum: the joint's curve is contaminated by a flow that
phase 2 throws away and refits, so it answers the wrong question.

Two things this prints:
  * per-head val NLL against body epoch, and where each head's own optimum sits.
    Heads peaking at different BODIES would say the trunk is being pulled in
    incompatible directions.
  * the joint's own val NLL against the SUM of refit marginals over the same
    dimensions. The joint exists to capture dependence; if the sum is lower, it
    captured none worth having and the marginals should just be used directly.

    python scripts/refit_compare.py <run_dir>
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir", type=Path)
    ap.add_argument("--history", type=Path, default=None,
                    help="history.jsonl for the joint's own curve (default <run_dir>/history.jsonl)")
    args = ap.parse_args()

    reports = sorted(args.run_dir.glob("refit/epoch_*/refit_report.json"))
    if not reports:
        raise SystemExit(f"no refit reports under {args.run_dir}/refit/")

    bodies: dict[int, dict[str, dict]] = {}
    for path in reports:
        m = re.search(r"epoch_(\d+)", str(path.parent))
        ep = int(m.group(1)) if m else -1
        bodies[ep] = {r["name"]: r for r in json.loads(path.read_text())}

    eps = sorted(bodies)
    heads = sorted({n for b in bodies.values() for n in b} - {"joint_refit"})

    print(f"bodies: {eps}\n")
    w = max(len(h) for h in heads) + 2
    print("val NLL after refit, by BODY epoch")
    print(" " * w + "".join(f"{('ep' + str(e)):>9s}" for e in eps) + "   best body")
    for h in heads:
        vals = [bodies[e].get(h, {}).get("val_nll") for e in eps]
        cells = "".join(f"{(v if v is not None else float('nan')):>9.3f}" for v in vals)
        fin = [(e, v) for e, v in zip(eps, vals) if v is not None and np.isfinite(v)]
        best = min(fin, key=lambda t: t[1])[0] if fin else -1
        print(f"{h:<{w}}{cells}   {('ep' + str(best)) if best > 0 else '-'}")

    # per-head stopping epochs on the best body: the thing phase 2 buys us
    fin_bodies = [e for e in eps if bodies[e]]
    tot = {e: sum(r["val_nll"] for n, r in bodies[e].items() if n != "joint_refit")
           for e in fin_bodies}
    pick = min(tot, key=lambda e: tot[e])
    print(f"\nsummed val NLL over all heads by body: "
          + ", ".join(f"ep{e}={tot[e]:.3f}" for e in fin_bodies))
    print(f"-> best body by POST-REFIT validation: epoch {pick}")
    print(f"\nper-head stopping epoch on body ep{pick} "
          f"(this spread is what a single shared checkpoint cannot serve):")
    for h in sorted((n for n in bodies[pick] if n != "joint_refit"),
                    key=lambda n: bodies[pick][n]["best_epoch"]):
        r = bodies[pick][h]
        print(f"   {h:<22s} stops at {r['best_epoch']:>4d}/{r['epochs_run']:<4d} "
              f"val {r['val_nll']:.4f}  (n_train {r['n_train']:,})")

    # does the joint beat independent marginals on its own dimensions?
    hist = args.history or (args.run_dir / "history.jsonl")
    joint_by_epoch = {}
    if hist.exists():
        for line in hist.read_text().splitlines():
            row = json.loads(line)
            if "val/nll_joint" in row and "epoch" in row:
                joint_by_epoch[int(row["epoch"])] = float(row["val/nll_joint"])
    cfg = args.run_dir / "config.json"
    joint_dims = json.loads(cfg.read_text()).get("joint_pair") if cfg.exists() else None
    joint_dims = joint_dims or ["logmstar_cigale", "log_sfr", "log_lx", "log_flux_p3"]

    print("\njoint vs independent marginals, on the joint's own dimensions")
    print(f"  dims: {', '.join(joint_dims)}")
    print(f"{'body':>6s} {'joint':>9s} {'source':>9s} {'sum marg':>10s} {'dependence':>12s}")
    for e in fin_bodies:
        have = [bodies[e][d]["val_nll"] for d in joint_dims if d in bodies[e]]
        if len(have) != len(joint_dims):
            continue
        # Prefer the REFIT joint: it was fit on the same frozen body under the
        # same protocol as the marginals. The phase-1 curve is whatever training
        # happened to leave, so comparing it measures fitting effort, not
        # dependence, and understates the joint.
        refit = bodies[e].get("joint_refit", {}).get("val_nll")
        j, src = (refit, "refit") if refit is not None else (joint_by_epoch.get(e), "phase1")
        if j is None:
            continue
        print(f"{('ep' + str(e)):>6s} {j:>9.3f} {src:>9s} {sum(have):>10.3f} {sum(have) - j:>+12.3f}")
    print("\n  dependence > 0 means the joint beats independent marginals by that many nats,")
    print("  which is the only reason to carry a joint head at all.")
    print("  source=phase1 rows are NOT like for like: that joint got less fitting than")
    print("  the marginals it is being compared against.")


if __name__ == "__main__":
    main()
