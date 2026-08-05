#!/usr/bin/env python
"""How the two-stage run went, and what each head cost.

Two figures:
  fig_two_stage_heads.png   per-head loss, phase 1 beside phase 2
  fig_two_stage_select.png  why a snapshot was chosen, and what the joint bought

The point the first figure has to make is easy to miss from a table. Phase 1
trains ONLY the joint head (--joint-only), so the marginal flows receive no
gradient at all. Their validation loss still drifts, because the body beneath
them keeps moving, but they are passengers, not learners. Phase 2 freezes that
body and refits each head on cached contexts, and the marginals then drop by
factors of several. That gap between "riding a moving body" and "refit on a
frozen one" is the entire argument for splitting training in two.

The second figure is the selection record. Phase 1's own validation minimum is
NOT the right body: what matters is how well the heads do AFTER refitting, and
that peaks at a different epoch. Dependence is the sum of the four marginal
val NLLs minus the joint's, i.e. how many nats the joint buys over assuming
the targets are independent.

    python docs/make_two_stage_figures.py --run-dir results/dr2_37257713
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

INK, MUTED, GRID = "#1a1a1a", "#6a6a6a", "#d5d5d5"
P1_C, P2_C, JOINT_C = "#E69F00", "#0072B2", "#009E73"
PICK_C = "#D55E00"

PRETTY = {
    "log_lx": r"log $L_X$", "log_sfr": "log SFR", "logmstar_cigale": r"log $M_*$",
    "log_flux_p3": "P3 flux", "joint": "joint (4-D)", "joint_refit": "joint (4-D)",
    "log_ml_flux_1": "log flux", "logmstar": r"log $M_*$ (FSF)",
    "log_flux_p1": "P1", "log_flux_p2": "P2", "log_flux_p4": "P4",
    "log_mbh_pan25": r"log $M_{BH}$ Pan25", "log_mbh_vo09": r"log $M_{BH}$ VO09",
    "p2xp3_refit": r"P2$\times$P3",
}
# The four dimensions the joint head models, in flow order.
JOINT_DIMS = ["logmstar_cigale", "log_sfr", "log_lx", "log_flux_p3"]


def pretty(n: str) -> str:
    return PRETTY.get(n, n.replace("_", " "))


def style(ax) -> None:
    ax.grid(True, color=GRID, lw=0.6, alpha=0.7)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(MUTED)
    ax.tick_params(colors=MUTED, labelsize=9)


def load_history(run_dir: Path) -> list[dict]:
    p = run_dir / "history.jsonl"
    rows = [json.loads(l) for l in p.read_text().splitlines() if l.strip()]
    return [r for r in rows if "val/nll_joint" in r]


def load_refits(run_dir: Path) -> dict[int, dict[str, dict]]:
    """{snapshot epoch: {head name: report}}"""
    out = {}
    for p in sorted(run_dir.glob("refit_epoch*.json")):
        ep = int(p.stem.replace("refit_epoch", ""))
        out[ep] = {r["name"]: r for r in json.loads(p.read_text())}
    return out


def draw_heads(hist: list[dict], refits: dict, out: Path, pick: int) -> None:
    """Phase 1 (untrained marginals drifting) beside phase 2 (refit on frozen body)."""
    fig, axes = plt.subplots(1, 2, figsize=(12.4, 5.0))
    heads = JOINT_DIMS + ["joint"]
    colors = plt.cm.viridis(np.linspace(0.05, 0.8, len(JOINT_DIMS))).tolist() + [JOINT_C]

    ax = axes[0]
    for h, c in zip(heads, colors):
        xs = [r.get("epoch", i) for i, r in enumerate(hist) if f"val/nll_{h}" in r]
        ys = [r[f"val/nll_{h}"] for r in hist if f"val/nll_{h}" in r]
        if not ys:
            continue
        ax.plot(xs, ys, color=c, lw=2.4 if h == "joint" else 1.6,
                label=pretty(h), zorder=3 if h == "joint" else 2)
    ax.axvline(pick, color=PICK_C, lw=1.2, ls="--", zorder=1)
    ax.text(pick, ax.get_ylim()[1], f" chosen body: epoch {pick}", color=PICK_C,
            fontsize=9, va="top", ha="left")
    ax.set_title("Phase 1: only the joint head is trained", color=INK, fontsize=12)
    ax.set_xlabel("epoch", color=MUTED)
    ax.set_ylabel("validation NLL (nats)", color=MUTED)
    ax.legend(frameon=False, fontsize=9, ncol=2)
    style(ax)

    ax = axes[1]
    rep = refits[pick]
    names = [n for n in rep if n not in ("p2xp3_refit",)]
    names.sort(key=lambda n: rep[n]["val_nll"])
    ys = np.arange(len(names))
    before = []
    for n in names:
        key = f"val/nll_{'joint' if n == 'joint_refit' else n}"
        v = [r[key] for r in hist if key in r]
        before.append(v[-1] if v else np.nan)
    after = [rep[n]["val_nll"] for n in names]
    for y, b, a in zip(ys, before, after):
        if np.isfinite(b):
            ax.plot([a, b], [y, y], color=GRID, lw=1.4, zorder=1)
            ax.scatter([b], [y], s=34, color=P1_C, zorder=2)
        ax.scatter([a], [y], s=34, color=P2_C, zorder=3)
    ax.set_yticks(ys)
    ax.set_yticklabels([pretty(n) for n in names], fontsize=9)
    ax.scatter([], [], s=34, color=P1_C, label="end of phase 1 (not trained)")
    ax.scatter([], [], s=34, color=P2_C, label="after phase 2 refit")
    ax.set_title(f"Phase 2: each head refit on the frozen epoch-{pick} body",
                 color=INK, fontsize=12)
    ax.set_xlabel("validation NLL (nats)", color=MUTED)
    ax.legend(frameon=False, fontsize=9, loc="upper right")
    style(ax)

    fig.tight_layout()
    fig.savefig(out, dpi=190, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"wrote {out}")


def draw_selection(hist: list[dict], refits: dict, out: Path, pick: int) -> None:
    """Phase-1 curve with its gap, and the post-refit numbers that chose the body."""
    fig, axes = plt.subplots(1, 3, figsize=(14.6, 4.6))
    eps = sorted(refits)

    ax = axes[0]
    xs = [r.get("epoch", i) for i, r in enumerate(hist)]
    ax.plot(xs, [r["val/nll_joint"] for r in hist], color=P2_C, lw=2.2, label="validation")
    pr = [(r.get("epoch", i), r["probe/nll_joint"])
          for i, r in enumerate(hist) if "probe/nll_joint" in r]
    if pr:
        ax.plot([p[0] for p in pr], [p[1] for p in pr], color=P1_C, lw=1.8,
                label="train probe")
    for e in eps:
        ax.axvline(e, color=GRID, lw=0.8, zorder=0)
    ax.axvline(pick, color=PICK_C, lw=1.3, ls="--", zorder=1)
    ax.set_title("Phase 1: the joint head", color=INK, fontsize=12)
    ax.set_xlabel("epoch", color=MUTED)
    ax.set_ylabel("NLL (nats)", color=MUTED)
    ax.legend(frameon=False, fontsize=9)
    style(ax)

    ax = axes[1]
    jr = [refits[e]["joint_refit"]["val_nll"] for e in eps]
    sm = [sum(refits[e][d]["val_nll"] for d in JOINT_DIMS) for e in eps]
    ax.plot(eps, jr, "o-", color=JOINT_C, lw=2.0, ms=6, label="joint, refit")
    ax.plot(eps, sm, "s-", color=MUTED, lw=1.6, ms=5, label="sum of 4 marginals")
    best = eps[int(np.argmin(jr))]
    ax.scatter([best], [min(jr)], s=150, facecolor="none", edgecolor=PICK_C,
               lw=2.0, zorder=5)
    ax.set_title("Phase 2 decides the body", color=INK, fontsize=12)
    ax.set_xlabel("snapshot epoch", color=MUTED)
    ax.set_ylabel("post-refit validation NLL", color=MUTED)
    ax.legend(frameon=False, fontsize=9)
    style(ax)

    ax = axes[2]
    dep = [s - j for s, j in zip(sm, jr)]
    ax.plot(eps, dep, "o-", color=JOINT_C, lw=2.2, ms=6)
    ax.axhline(0, color=MUTED, lw=1.0, ls=":")
    ax.fill_between(eps, 0, dep, color=JOINT_C, alpha=0.12)
    ax.set_ylim(0, max(dep) * 1.25)
    ax.set_title("What the joint head buys", color=INK, fontsize=12)
    ax.set_xlabel("snapshot epoch", color=MUTED)
    ax.set_ylabel("dependence (nats)", color=MUTED)
    ax.text(eps[len(eps) // 2], max(dep) * 0.45,
            f"mean {np.mean(dep):.2f} nats\nover independent marginals",
            color=INK, fontsize=10, ha="center")
    style(ax)

    fig.tight_layout()
    fig.savefig(out, dpi=190, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"wrote {out}")


def write_table(refits: dict, out: Path, pick: int) -> None:
    rep = refits[pick]
    lines = ["head,val_nll,best_epoch,epochs_run,n_train,n_val"]
    for n, r in sorted(rep.items(), key=lambda kv: kv[1]["val_nll"]):
        lines.append(f"{n},{r['val_nll']:.4f},{r['best_epoch']},{r['epochs_run']},"
                     f"{r['n_train']},{r['n_val']}")
    out.write_text("\n".join(lines) + "\n")
    print(f"wrote {out}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run-dir", type=Path, required=True,
                    help="dir with history.jsonl and refit_epoch*.json")
    ap.add_argument("--figdir", type=Path, default=Path(__file__).parent / "figures")
    ap.add_argument("--pick", type=int, default=None,
                    help="chosen snapshot; default is the post-refit joint minimum")
    args = ap.parse_args()

    hist = load_history(args.run_dir)
    refits = load_refits(args.run_dir)
    if not refits:
        raise SystemExit(f"no refit_epoch*.json in {args.run_dir}")
    eps = sorted(refits)
    pick = args.pick if args.pick is not None else min(
        eps, key=lambda e: refits[e]["joint_refit"]["val_nll"])
    print(f"[two-stage] snapshots {eps}, chosen body epoch {pick}")

    args.figdir.mkdir(parents=True, exist_ok=True)
    draw_heads(hist, refits, args.figdir / "fig_two_stage_heads.png", pick)
    draw_selection(hist, refits, args.figdir / "fig_two_stage_select.png", pick)
    write_table(refits, args.figdir / "two_stage_heads.csv", pick)


if __name__ == "__main__":
    main()
