"""
block_ablation.py -- which single CLIP block earns its AdaptFormer?

Trains one model per block with AdaptFormer attached to **that block only**, then
aggregates the runs into a per-block importance curve. Answers "where in the vision tower
does adaptation actually pay off", which the all-blocks run cannot tell you.

    block 0 only  ->  test macro F1 ?
    block 1 only  ->  test macro F1 ?
    ...
    block N-1 only -> test macro F1 ?

Two reference points are included by default:

    head-only   no adapters at all, only the classifier head trains (the floor: what you
                get from the frozen CLIP features alone)
    all-blocks  every block adapted (the ceiling from the previous experiments)

Each child run is a normal `clip_adaptformer.py` run in its own `Runs/` folder with the
full artefact set, launched as a separate process so one crash cannot take down the sweep
and GPU memory is released between runs. The sweep folder holds the aggregate.

Usage
-----
    python block_ablation.py                                  # ViT-B/32, 12 blocks, 10 epochs
    python block_ablation.py --backbone ViT-L/14 --epochs 3 --batch_size 32
    python block_ablation.py --blocks 0,4,8,11 --seeds 0,1,2   # a few blocks, 3 seeds each
    python block_ablation.py --aggregate_only Runs/<sweep-dir> # re-aggregate without training

Note on noise: one seed per block does not separate blocks that differ by a point or two of
F1. Pass `--seeds 0,1,2` for error bars before drawing conclusions from small gaps; the
aggregate reports the spread whenever more than one seed is present.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import subprocess
import sys
import time
from datetime import datetime

SCRIPTS_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPTS_DIR not in sys.path:
    sys.path.insert(0, SCRIPTS_DIR)

import experiment_logger as el
from experiment_logger import CLASS_SHORT, ROOT_DIR

HEAD_ONLY = "head_only"
ALL_BLOCKS = "all"


# ===========================================================================
# Running one child job
# ===========================================================================
def child_command(spec, seed, args, run_name):
    """`spec` is a block index, HEAD_ONLY, or ALL_BLOCKS."""
    if spec == HEAD_ONLY:
        blocks_arg = ""          # empty list -> no adapters injected, head still trains
    elif spec == ALL_BLOCKS:
        blocks_arg = "all"
    else:
        blocks_arg = str(spec)
    cmd = [
        sys.executable, os.path.join(SCRIPTS_DIR, "clip_adaptformer.py"),
        "--backbone", args.backbone,
        "--blocks", blocks_arg,
        "--bottleneck", str(args.bottleneck),
        "--epochs", str(args.epochs),
        "--batch_size", str(args.batch_size),
        "--lr", str(args.lr),
        "--seed", str(seed),
        "--head", args.head,
        "--select_on", args.select_on,
        "--run_name", run_name,
        "--log_dir", args.log_dir,
        "--num_workers", str(args.num_workers),
    ]
    if args.no_amp:
        cmd.append("--no_amp")
    return cmd


def launch(spec, seed, args, logger, sweep_dir):
    tag = spec if isinstance(spec, str) else f"b{spec:02d}"
    run_name = f"{args.prefix}_{tag}_s{seed}"
    cmd = child_command(spec, seed, args, run_name)

    logger.log(f"--- {tag} (seed {seed}) : launching child run '{run_name}'")
    t0 = time.time()
    child_log = os.path.join(sweep_dir, "child_logs", f"{run_name}.log")
    os.makedirs(os.path.dirname(child_log), exist_ok=True)
    with open(child_log, "w", encoding="utf-8") as fh:
        proc = subprocess.run(cmd, stdout=fh, stderr=subprocess.STDOUT, text=True)
    elapsed = time.time() - t0

    log_root = args.log_dir if os.path.isabs(args.log_dir) else os.path.join(ROOT_DIR, args.log_dir)
    matches = sorted(glob.glob(os.path.join(log_root, f"*_{run_name}")))
    run_dir = matches[-1] if matches else None

    if proc.returncode != 0 or run_dir is None:
        logger.log(f"    FAILED (exit {proc.returncode}); child log: {child_log}", "ERROR")
        return {"spec": tag, "seed": seed, "status": "failed",
                "returncode": proc.returncode, "child_log": child_log, "run_dir": run_dir}

    summary_path = os.path.join(run_dir, "summary.json")
    if not os.path.exists(summary_path):
        logger.log(f"    no summary.json in {run_dir}", "ERROR")
        return {"spec": tag, "seed": seed, "status": "no_summary", "run_dir": run_dir}

    with open(summary_path, encoding="utf-8") as fh:
        summary = json.load(fh)
    test = summary["test"]
    best = summary.get("best_validation") or {}
    record = {
        "spec": tag,
        "block": None if isinstance(spec, str) else spec,
        "seed": seed,
        "status": "ok",
        "run_dir": run_dir,
        "trainable_params": summary["trainable"]["trainable_params"],
        "adapter_params": summary["adaptformer"]["adapter_params"],
        "blocks_adapted": summary["adaptformer"]["blocks_adapted"],
        "best_val_epoch": best.get("epoch"),
        "best_val_macro_f1": best.get("val_macro_f1"),
        "test_accuracy": test["accuracy"],
        "test_weighted_f1": test["weighted_f1"],
        "test_macro_f1": test["macro_f1"],
        "test_mmae": test.get("mmae"),
        "runtime_s": round(elapsed, 1),
    }
    for i, name in enumerate(CLASS_SHORT):
        record[f"f1_{name}"] = summary["test_per_class"][name]["f1-score"]
    logger.log(f"    {tag} seed {seed}: test macroF1={test['macro_f1']:.4f} "
               f"acc={test['accuracy'] * 100:.2f}% "
               f"({record['trainable_params']:,} trainable, {elapsed:.0f}s)")
    logger.event("child_run", **record)
    return record


# ===========================================================================
# Aggregation
# ===========================================================================
def aggregate(records, sweep_dir, args, logger):
    import numpy as np
    import pandas as pd

    ok = [r for r in records if r.get("status") == "ok"]
    if not ok:
        logger.log("no successful runs to aggregate", "ERROR")
        return None

    df = pd.DataFrame(ok)
    df.to_csv(os.path.join(sweep_dir, "block_ablation_runs.csv"), index=False)

    metric = "test_macro_f1"
    grouped = (df.groupby("spec")
                 .agg(n_seeds=("seed", "count"),
                      block=("block", "first"),
                      trainable_params=("trainable_params", "first"),
                      macro_f1_mean=(metric, "mean"),
                      macro_f1_std=(metric, "std"),
                      macro_f1_min=(metric, "min"),
                      macro_f1_max=(metric, "max"),
                      acc_mean=("test_accuracy", "mean"),
                      wf1_mean=("test_weighted_f1", "mean"))
                 .reset_index())
    grouped["macro_f1_std"] = grouped["macro_f1_std"].fillna(0.0)
    grouped.to_csv(os.path.join(sweep_dir, "block_ablation.csv"), index=False)

    per_block = grouped[grouped["block"].notna()].sort_values("block")
    floor = grouped[grouped["spec"] == HEAD_ONLY]["macro_f1_mean"]
    ceiling = grouped[grouped["spec"] == ALL_BLOCKS]["macro_f1_mean"]
    floor = float(floor.iloc[0]) if len(floor) else None
    ceiling = float(ceiling.iloc[0]) if len(ceiling) else None

    lines = [
        "AdaptFormer per-block ablation -- one block adapted at a time",
        f"backbone {args.backbone} | r={args.bottleneck} | {args.epochs} epochs | "
        f"seeds {args.seeds} | head {args.head}",
        "",
        f"{'block':<12}{'trainable':>12}{'macro F1':>11}{'+/-':>8}{'n':>3}"
        f"{'acc':>9}{'weighted F1':>13}",
        "-" * 68,
    ]
    for _, row in per_block.iterrows():
        lines.append(f"{int(row['block']):<12}{int(row['trainable_params']):>12,}"
                     f"{row['macro_f1_mean']:>11.4f}{row['macro_f1_std']:>8.4f}"
                     f"{int(row['n_seeds']):>3}"
                     f"{row['acc_mean']:>9.4f}{row['wf1_mean']:>13.4f}")
    lines.append("-" * 68)
    for spec, label in ((HEAD_ONLY, "head only"), (ALL_BLOCKS, "all blocks")):
        sub = grouped[grouped["spec"] == spec]
        if len(sub):
            row = sub.iloc[0]
            lines.append(f"{label:<12}{int(row['trainable_params']):>12,}"
                         f"{row['macro_f1_mean']:>11.4f}{row['macro_f1_std']:>8.4f}"
                         f"{int(row['n_seeds']):>3}"
                         f"{row['acc_mean']:>9.4f}{row['wf1_mean']:>13.4f}")
    lines.append("  n = seeds averaged; '+/-' is 0 wherever n = 1 and means nothing there.")

    if len(per_block):
        best = per_block.loc[per_block["macro_f1_mean"].idxmax()]
        worst = per_block.loc[per_block["macro_f1_mean"].idxmin()]
        spread = float(per_block["macro_f1_mean"].max() - per_block["macro_f1_mean"].min())
        lines += ["", "Findings",
                  f"  best single block : {int(best['block'])} "
                  f"(macro F1 {best['macro_f1_mean']:.4f})",
                  f"  worst single block: {int(worst['block'])} "
                  f"(macro F1 {worst['macro_f1_mean']:.4f})",
                  f"  spread across blocks: {spread:.4f} macro F1"]
        if floor is not None:
            lines.append(f"  best single block over head-only floor: "
                         f"{best['macro_f1_mean'] - floor:+.4f}")
        # noise must come only from configs that actually have repeated seeds
        repeated = grouped[grouped["n_seeds"] > 1]
        noise = float(repeated["macro_f1_std"].mean()) if len(repeated) else None
        band = 2 * noise if noise is not None else None

        if floor is not None and ceiling is not None:
            gain = ceiling - floor
            if best["macro_f1_mean"] <= floor:
                lines.append("  no single block beat the head-only floor -- on this budget "
                             "the gain comes from adapting many blocks together, not from "
                             "any one of them")
            elif best["macro_f1_mean"] > ceiling:
                delta = best["macro_f1_mean"] - ceiling
                if band is not None and delta < band:
                    lines.append(f"  block {int(best['block'])} alone MATCHES all-blocks "
                                 f"({best['macro_f1_mean']:.4f} vs {ceiling:.4f}; the "
                                 f"{delta:+.4f} gap is inside the +/-{band:.4f} noise band) "
                                 f"-- same result for 1/{len(per_block)} of the adapter "
                                 f"budget, so adapting every block buys nothing here")
                else:
                    lines.append(f"  block {int(best['block'])} alone BEATS all-blocks "
                                 f"({best['macro_f1_mean']:.4f} vs {ceiling:.4f}, "
                                 f"{delta:+.4f} > the {band:.4f} noise band) using "
                                 f"1/{len(per_block)} of the adapter budget")
            elif gain > 1e-6:
                share = (best["macro_f1_mean"] - floor) / gain
                lines.append(f"  best single block recovers {100 * share:.1f}% of the "
                             f"head-only -> all-blocks gain, using "
                             f"1/{len(per_block)} of the adapter budget")
        if noise is not None:
            n_rep = len(repeated)
            lines.append(f"  seed-to-seed std, over the {n_rep} config(s) with repeats: "
                         f"{noise:.4f} -- treat gaps smaller than ~{band:.4f} as noise")
            single = per_block[per_block["n_seeds"] == 1]
            if len(single):
                lines.append(f"  {len(single)} block(s) have a single seed "
                             f"({', '.join(str(int(b)) for b in single['block'])}); their "
                             f"individual values carry that same ~{band:.4f} uncertainty")
        else:
            lines.append("  single seed everywhere: gaps of a point or two are NOT "
                         "separable from run-to-run noise. Re-run with --seeds 0,1,2.")

    text = "\n".join(lines)
    with open(os.path.join(sweep_dir, "block_ablation.txt"), "w", encoding="utf-8") as fh:
        fh.write(text + "\n")
    print("\n" + text + "\n")

    # ---- plot ---------------------------------------------------------
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(9, 5))
        x = per_block["block"].to_numpy(dtype=int)
        y = per_block["macro_f1_mean"].to_numpy()
        err = per_block["macro_f1_std"].to_numpy()
        if err.any():
            ax.errorbar(x, y, yerr=err, marker="o", capsize=4, linewidth=1.8, label="single block")
        else:
            ax.plot(x, y, marker="o", linewidth=1.8, label="single block")
        if floor is not None:
            ax.axhline(floor, linestyle="--", linewidth=1.3, color="#888",
                       label=f"head only ({floor:.3f})")
        if ceiling is not None:
            ax.axhline(ceiling, linestyle="-.", linewidth=1.3, color="#444",
                       label=f"all blocks ({ceiling:.3f})")
        ax.set_xlabel("CLIP vision block index (0 = closest to the patch embedding)")
        ax.set_ylabel("test macro F1")
        ax.set_title(f"AdaptFormer in one block at a time -- {args.backbone}, r={args.bottleneck}")
        ax.set_xticks(x)
        ax.grid(alpha=0.3)
        ax.legend()
        fig.tight_layout()
        fig.savefig(os.path.join(sweep_dir, "block_ablation.png"), dpi=150)
        plt.close(fig)
    except Exception as exc:
        logger.log(f"plot skipped: {exc}", "WARN")

    return {"per_block": per_block.to_dict("records"), "floor": floor, "ceiling": ceiling}


# ===========================================================================
# Entry point
# ===========================================================================
def n_blocks_of(backbone):
    """Block count without paying for a full model load on the GPU."""
    import clip
    import torch
    model, _ = clip.load(backbone, device="cpu")
    n = len(model.visual.transformer.resblocks)
    del model
    torch.cuda.empty_cache()
    return n


def build_parser():
    p = argparse.ArgumentParser(
        description="Per-block AdaptFormer ablation on the CLIP vision tower.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--backbone", type=str, default="ViT-B/32")
    p.add_argument("--blocks", type=str, default="all",
                   help="'all' or a comma list of block indices to test individually")
    p.add_argument("--seeds", type=str, default="42", help="comma list, e.g. 0,1,2")
    p.add_argument("--bottleneck", type=int, default=64)
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--head", type=str, default="zeroshot", choices=["zeroshot", "linear"])
    p.add_argument("--select_on", type=str, default="macro_f1", choices=["macro_f1", "accuracy"])
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--no_amp", action="store_true")
    p.add_argument("--no_baselines", action="store_true",
                   help="skip the head-only and all-blocks reference runs")
    p.add_argument("--prefix", type=str, default="abl", help="prefix for child run names")
    p.add_argument("--log_dir", type=str, default="Runs")
    p.add_argument("--sweep_name", type=str, default="block_ablation")
    p.add_argument("--aggregate_only", type=str, default=None,
                   help="path to an existing sweep folder: re-aggregate, train nothing")
    return p


def main(argv=None):
    args = build_parser().parse_args(argv)
    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]

    # ---- re-aggregate an existing sweep --------------------------------
    if args.aggregate_only:
        sweep_dir = args.aggregate_only if os.path.isabs(args.aggregate_only) \
            else os.path.join(ROOT_DIR, args.aggregate_only)
        with open(os.path.join(sweep_dir, "records.json"), encoding="utf-8") as fh:
            records = json.load(fh)
        logger = el.RunLogger(sweep_dir, training_csvs=False)
        try:
            aggregate(records, sweep_dir, args, logger)
        finally:
            logger.close()
        return 0

    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    log_root = args.log_dir if os.path.isabs(args.log_dir) else os.path.join(ROOT_DIR, args.log_dir)
    sweep_dir = os.path.join(log_root, f"{stamp}_{args.sweep_name}")
    os.makedirs(sweep_dir, exist_ok=True)

    logger = el.RunLogger(sweep_dir, training_csvs=False)
    exit_code = 0
    try:
        total_blocks = n_blocks_of(args.backbone)
        if args.blocks.strip().lower() == "all":
            block_list = list(range(total_blocks))
        else:
            block_list = [int(b) for b in args.blocks.split(",") if b.strip()]
        for b in block_list:
            if not 0 <= b < total_blocks:
                raise ValueError(f"block {b} out of range for {args.backbone} "
                                 f"({total_blocks} blocks)")

        specs = list(block_list)
        if not args.no_baselines:
            specs = [HEAD_ONLY] + specs + [ALL_BLOCKS]

        jobs = [(spec, seed) for spec in specs for seed in seeds]
        cfg = {
            "backbone": args.backbone, "total_blocks": total_blocks,
            "blocks_tested": block_list, "seeds": seeds, "baselines": not args.no_baselines,
            "bottleneck": args.bottleneck, "epochs": args.epochs,
            "batch_size": args.batch_size, "lr": args.lr, "head": args.head,
            "n_child_runs": len(jobs), "sweep_dir": sweep_dir,
            "command": " ".join([sys.executable] + sys.argv),
        }
        logger.dump("config.json", cfg)
        logger.log(f"{args.backbone}: {total_blocks} blocks. "
                   f"{len(jobs)} child runs = {len(specs)} configs x {len(seeds)} seed(s), "
                   f"sequentially (one GPU job at a time).")
        logger.log("Reference points: head-only (floor) and all-blocks (ceiling)."
                   if not args.no_baselines else "Baselines skipped.")

        records, t0 = [], time.time()
        for i, (spec, seed) in enumerate(jobs, start=1):
            done = time.time() - t0
            eta = (done / max(i - 1, 1)) * (len(jobs) - i + 1) if i > 1 else 0
            logger.log(f"[{i}/{len(jobs)}] elapsed {done / 60:.1f} min"
                       + (f", eta {eta / 60:.1f} min" if i > 1 else ""))
            records.append(launch(spec, seed, args, logger, sweep_dir))
            with open(os.path.join(sweep_dir, "records.json"), "w", encoding="utf-8") as fh:
                json.dump(records, fh, indent=2)

        failed = [r for r in records if r.get("status") != "ok"]
        if failed:
            logger.log(f"{len(failed)} of {len(records)} child runs failed: "
                       f"{[r['spec'] for r in failed]}", "WARN")
        aggregate(records, sweep_dir, args, logger)
        logger.timings["sweep_s"] = round(time.time() - t0, 1)
        logger.log(f"Sweep finished in {logger.timings['sweep_s'] / 60:.1f} min")

    except KeyboardInterrupt:
        exit_code = 130
        logger.log("Interrupted by user", "ERROR")
    except Exception:
        import traceback
        exit_code = 1
        tb = traceback.format_exc()
        logger.log(f"SWEEP FAILED\n{tb}", "ERROR")
        with open(os.path.join(sweep_dir, "traceback.txt"), "w", encoding="utf-8") as fh:
            fh.write(tb)
    finally:
        logger.event("sweep_end", exit_code=exit_code)
        logger.close()
        print(f"\nSweep artefacts: {sweep_dir}")
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
