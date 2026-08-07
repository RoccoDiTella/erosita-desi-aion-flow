#!/usr/bin/env python
"""Test-set evaluation of a train-multi checkpoint: per-head tables + joint HR.

Per modality combo (all 15): one encoder pass serves every head. Scalar heads
get plain-LL NLL, posterior-mean R2/RMSE, and IG vs a per-target KDE prior fit
on the train view. The joint head is scored on its fully observed rows and, when
both band fluxes are joint dimensions, sampled to produce per-source HR
posteriors via the exact per-band flux/rate constants -- compared against
measured HR32 and against the independent-bands baseline, which is recomputed
in-run on the same rows rather than quoted from a previous one.

Every table row carries its sample declaration (sample/n_test/n_common/
n_sample/frac_of_test). ``--sample common`` fixes the row set across all 15
combos, which is the only setting in which an IG, R2 or RMSE difference between
two combos is a statement about inputs rather than about row sets. The resolved
sample is printed before the run, so the log says which row set every number
below it was measured on. ``sbatch/eval_multi.sbatch`` passes ``--sample both``,
which writes one ``common`` and one ``native`` row per (head, input_group); the
deck then filters on that column and refuses to guess (docs/deck_sample.py).

Outputs, all under ``<checkpoint dir>/eval/``:
  multi_test_metrics.csv    per (head, input_group, sample)
  sfr_vs_mass_baseline.csv  the SFR head against a mass-only predictor
  hr_joint_posteriors.csv   per-source HR from the joint (+ the independent-band
                            baseline, recomputed on the same rows)
  hr_joint_summary.csv      those two against the measured HR32; needs --hr-ref-csv.
                            Rendered by docs/make_slides.py and docs/make_html_deck.py.

    python scripts/eval_multitarget.py --checkpoint .../best.pt \
        --staged-dir ... --clean-split-csv ... --extra-targets-csv ... \
        [--hr-ref-csv targets_extra.csv] [--sample both] [--batch-size 224]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from shareable_aion_flow.attention_pooling_head import MODALITIES, combo_name  # noqa: E402
from shareable_aion_flow.data_to_aion_embeddings import build_dataloaders  # noqa: E402
from shareable_aion_flow.eval_core import (  # noqa: E402
    SAMPLE_CHOICES, build_priors, hr_summary_rows, resolve_samples, run_eval,
)
from shareable_aion_flow.multitarget import (  # noqa: E402
    MultiTargetFlows, MultiTargetLookup, SharedCLSHead,
)
from shareable_aion_flow.normalizing_flow import TargetStandardizer  # noqa: E402
from shareable_aion_flow.stub_encoder import build_encoder  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--checkpoint", type=Path, required=True)
    ap.add_argument("--staged-dir", type=Path, required=True)
    ap.add_argument("--clean-split-csv", type=Path, required=True)
    ap.add_argument("--extra-targets-csv", type=Path, required=True)
    ap.add_argument("--hr-ref-csv", type=Path, default=None)
    ap.add_argument("--batch-size", type=int, default=224)
    ap.add_argument("--num-samples", type=int, default=128)
    ap.add_argument("--joint-samples", type=int, default=256)
    ap.add_argument("--num-workers", type=int, default=8)
    ap.add_argument("--device", default=None,
                    help="torch device; default cuda when available. Pass cpu to run "
                         "this script on a workstation, which together with "
                         "AIONFLOW_STUB_ENCODER=1 makes the eval path smoke-testable "
                         "off the cluster (its numbers are then meaningless).")
    ap.add_argument("--sample", choices=SAMPLE_CHOICES, default="native",
                    help="row set per combo: native = rows supporting that combo, "
                         "common = rows supporting every modality (required for cross-combo "
                         "deltas), both = one row of each. sbatch/eval_multi.sbatch passes "
                         "both; this default is kept at native so a hand-run command that "
                         "names no sample cannot be mistaken for a deck-grade table.")
    args = ap.parse_args()

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    ckpt = torch.load(args.checkpoint, map_location=device)
    # The checkpoint may have been trained with heads dropped, and with a
    # different joint. Rebind the head configuration before building anything:
    # module-level imports captured the default set at import time, which is why
    # eval_core reads HEAD_NAMES/joint_dims through the module on every call.
    import shareable_aion_flow.multitarget as _mt
    _mt.configure_heads_from_config(ckpt.get("config", {}))
    print(f"[eval] heads: {_mt.HEAD_NAMES}", flush=True)
    print(f"[eval] joint dimensions (flow column order): {_mt.joint_dims()}", flush=True)
    encoder = build_encoder(num_cls=_mt.N_HEADS, device=device, tag="eval")
    missing, unexpected = encoder.load_state_dict(ckpt["encoder_trainable_state_dict"], strict=False)
    assert not unexpected, f"unexpected encoder keys: {unexpected[:4]}"
    head = SharedCLSHead().to(device)
    head.load_state_dict(ckpt["head_state_dict"])
    flows = MultiTargetFlows().to(device)
    flows.load_state_dict(ckpt["flows_state_dict"])
    standardizers = [TargetStandardizer.from_state_dict(s) for s in ckpt["standardizers"]]
    encoder.eval(); head.eval(); flows.eval()

    _, _, test_loader = build_dataloaders(
        staged_dir=args.staged_dir, target_name=None,
        batch_size=args.batch_size, eval_batch_size=args.batch_size,
        num_workers=args.num_workers, clean_split_csv=args.clean_split_csv,
    )
    lookup = MultiTargetLookup(args.staged_dir, args.extra_targets_csv)
    assign = pd.read_csv(args.clean_split_csv)
    train_tids = assign.loc[assign.split == "train", "targetid"].to_numpy(np.int64)
    train_y = lookup.values_for(train_tids)
    priors = build_priors(train_y, standardizers)
    for j, prior in enumerate(priors):
        if prior is None:
            print(f"[eval] head {_mt.HEAD_NAMES[j]}: no train labels, information gain unavailable",
                  flush=True)

    out_dir = args.checkpoint.parent / "eval"
    out_dir.mkdir(exist_ok=True)

    def encode(batch, combo):
        cls_seq, _ = encoder.encode_tokens(batch, combo)
        return head(cls_seq)

    # The resolved sample goes in the log BEFORE the run: every number below is
    # conditional on it, and a table whose row set is not stated in the log is a
    # table nobody can check afterwards.
    resolved = resolve_samples(args.sample)
    print(f"[eval] sample: --sample {args.sample} -> every row stamped {list(resolved)}", flush=True)
    print("[eval]   common = sources carrying every modality, identical row set for all 15 "
          "combos (the only sample in which an IG/R2/RMSE difference between combos is an "
          "input effect)", flush=True)
    print("[eval]   native = sources carrying the modalities of that combo", flush=True)

    result = run_eval(
        loader=test_loader, lookup=lookup, encode=encode, flows=flows,
        standardizers=standardizers, train_y=train_y, priors=priors, device=device,
        sample=args.sample, num_samples=args.num_samples, joint_samples=args.joint_samples,
        log=lambda msg: print(msg, flush=True),
    )
    for note in result.notes:
        print(note, flush=True)

    table = pd.DataFrame(result.table)
    table.to_csv(out_dir / "multi_test_metrics.csv", index=False)
    print(f"[eval] wrote {len(table)} rows, "
          + ", ".join(f"{int(n)} with sample={s}"
                      for s, n in table["sample"].value_counts().items()), flush=True)
    print(f"[eval] the reference-combo products (SFR-vs-mass baseline, HR posteriors) are from "
          f"sample={result.hr_sample}; for the all-modality combo the two row sets coincide",
          flush=True)
    allin = table[table.input_group == combo_name(MODALITIES)]
    cols = [c for c in ("sample", "head", "n_sample", "n_test", "nll", "r2", "rmse_dex",
                        "info_gain_nats", "n_joint_full", "n_joint_marginalised")
            if c in allin.columns]
    for s in result.samples:
        print(f"\n[eval] all inputs, sample={s}", flush=True)
        print(allin[allin["sample"] == s][cols].to_string(index=False), flush=True)

    if result.sfr_baseline is not None:
        b = result.sfr_baseline
        print(f"\n[SFR vs mass-only baselines]  n={b['n']}  mass head: {b['mass_head']}  "
              f"main sequence: logSFR = {b['ms_slope']:+.3f}*logM* {b['ms_intercept']:+.3f} "
              f"(train fit, n={b['n_train_pairs']})")
        print(f"  SFR head                       R2 = {b['r2_sfr_head']:+.3f}")
        print(f"  baseline, TRUE {b['mass_head']:<16s} R2 = {b['r2_true_mstar']:+.3f}   "
              f"<- ceiling for any mass-only predictor")
        print(f"  baseline, PREDICTED mass       R2 = {b['r2_pred_mstar']:+.3f}")
        print(f"  verdict: {b['verdict']} (head - best baseline = {b['margin']:+.3f})")
        pd.DataFrame([b]).to_csv(out_dir / "sfr_vs_mass_baseline.csv", index=False)

    hr_df = pd.DataFrame(result.hr_records)
    hr_df.to_csv(out_dir / "hr_joint_posteriors.csv", index=False)
    if len(hr_df):
        width = float(np.median(0.5 * (hr_df.hr_p84.to_numpy() - hr_df.hr_p16.to_numpy())))
        baseline = result.hr_baseline_width
        base_txt = ("independent-bands baseline: unavailable" if baseline is None
                    else f"independent-bands baseline: {baseline:.3f} (recomputed on these rows)")
        print(f"\n[HR-from-JOINT] n={len(hr_df)} (sample={result.hr_sample})  "
              f"median 68% half-width {width:.3f}  {base_txt}")
    if args.hr_ref_csv is not None and len(hr_df):
        ref = pd.read_csv(args.hr_ref_csv).drop_duplicates("targetid").set_index("targetid")
        rows = hr_summary_rows(result.hr_records, ref)
        for row in rows:
            extra = ("" if "corr_independent" not in row else
                     f"  [independent bands corr={row['corr_independent']:+.3f} "
                     f"R2={row['r2_independent']:+.3f}]")
            print(f"  {row['subset']:12s} n={row['n']:5d} corr={row['corr_joint']:+.3f}  "
                  f"R2={row['r2_joint']:+.3f}{extra}")
        pd.DataFrame(rows).to_csv(out_dir / "hr_joint_summary.csv", index=False)
        qs = np.percentile(hr_df.hr_p50.to_numpy(), [10, 25, 50, 75, 90])
        print("  population HR quantiles p10..p90: " + " ".join(f"{q:+.3f}" for q in qs))
    print(f"written to {out_dir}")


if __name__ == "__main__":
    main()
