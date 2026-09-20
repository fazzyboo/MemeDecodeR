"""
OCR error rates against the gold captions, measured on the same 4,848 MIMOSA memes.

Requires Dataset/caption_error_rates.csv from make_ocr_splits.py.

This is the direct measurement the caption argument rests on: both the machine and the
hand-corrected text exist for every meme, so the difference between them is an edit
distance rather than an impression.

Once the OCR-caption MAF run exists, `--predictions` joins its per-meme correctness onto
these rates and adds the accuracy-versus-CER panel, which is the figure that establishes
the link rather than merely asserting it.

Usage:
    python analysis_cer.py
    python analysis_cer.py --predictions "../OCR MIMOSA on MAF/predictions_maf_fixed_sched.csv" \
                           --gold_predictions "../MIMOSA on MAF/predictions_maf_fixed_sched.csv"
"""
import argparse
import os

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BLUE, ORANGE, AQUA, MUTED = "#2a78d6", "#eb6834", "#1baf7a", "#9a9892"
INK, INK2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e8e7e3", "#fcfcfb"

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "font.family": "DejaVu Sans", "font.size": 9,
    "axes.edgecolor": GRID, "axes.labelcolor": INK2, "axes.titlecolor": INK,
    "axes.titlesize": 10.5, "axes.titleweight": "bold", "axes.titlelocation": "left",
    "axes.titlepad": 10, "grid.color": GRID, "grid.linewidth": 0.7,
    "xtick.color": INK2, "ytick.color": INK2, "xtick.labelsize": 8.5, "ytick.labelsize": 8.5,
    "axes.spines.top": False, "axes.spines.right": False,
    "legend.frameon": False, "legend.fontsize": 8.5,
})

BINS = [0, 0.25, 0.5, 0.75, 1.0, np.inf]
BIN_LABELS = ["0-0.25\nnear-clean", "0.25-0.5", "0.5-0.75", "0.75-1.0", ">1.0\nmore noise\nthan text"]


def fig_cer_distribution(err, out):
    """Single-series distribution, so one hue and no legend - the title names it."""
    fig, ax = plt.subplots(figsize=(7.4, 3.2))
    capped = np.clip(err["cer"], 0, 2.5)
    ax.hist(capped, bins=np.linspace(0, 2.5, 51),
            weights=np.ones(len(capped)) / len(capped), color=BLUE, zorder=3)

    median = err["cer"].median()
    ax.axvline(median, color=ORANGE, linestyle="--", linewidth=1.8, zorder=4)
    ax.annotate("median CER %.2f" % median, xy=(median, 0.052), xytext=(median + 0.28, 0.062),
                fontsize=8.6, color=INK,
                arrowprops=dict(arrowstyle="->", color=INK2, linewidth=1))
    ax.axvline(1.0, color=MUTED, linestyle=":", linewidth=1.5, zorder=4)
    ax.annotate("%.0f%% of memes exceed CER 1.0 -" % (100 * (err["cer"] > 1).mean()) + chr(10)
                + "more edits than the gold caption has characters",
                xy=(1.0, 0.030), xytext=(1.30, 0.050), fontsize=8, color=INK,
                arrowprops=dict(arrowstyle="->", color=INK2, linewidth=1))

    ax.set_xlabel("character error rate against the gold caption (final bin is 2.5+)")
    ax.set_ylabel("share of memes")
    ax.set_title("Bengali OCR error on MIMOSA, measured against hand-corrected text")
    ax.set_axisbelow(True); ax.grid(axis="y", alpha=0.9); ax.grid(axis="x", visible=False)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "fig11_cer_distribution.png"), dpi=200)
    plt.close(fig)


def fig_accuracy_vs_cer(err, ocr_preds, gold_preds, out):
    """The causal link: accuracy against OCR damage, with the gold run as the control.

    If the OCR line falls as CER rises while the gold line stays flat on the SAME memes,
    caption noise is doing the damage - not some property of those memes being harder.
    """
    merged = err[err["split"] == "testing_set"].merge(
        ocr_preds[["image_name", "true_id", "pred_id"]], on="image_name", how="inner")
    merged["ocr_correct"] = merged["true_id"] == merged["pred_id"]
    if gold_preds is not None:
        g = gold_preds[["image_name", "true_id", "pred_id"]].copy()
        g["gold_correct"] = g["true_id"] == g["pred_id"]
        merged = merged.merge(g[["image_name", "gold_correct"]], on="image_name", how="left")

    merged["bin"] = pd.cut(merged["cer"], bins=BINS, labels=BIN_LABELS, include_lowest=True)
    grouped = merged.groupby("bin", observed=False)
    counts = grouped.size()

    fig, ax = plt.subplots(figsize=(7.6, 3.4))
    x = np.arange(len(BIN_LABELS))
    series = [("ocr_correct", ORANGE, "MAF, machine OCR captions")]
    if "gold_correct" in merged.columns:
        series.insert(0, ("gold_correct", BLUE, "MAF, hand-corrected captions"))

    for col, colour, label in series:
        vals = grouped[col].mean().reindex(BIN_LABELS).to_numpy()
        ax.plot(x, vals, color=colour, linewidth=2.2, marker="o", markersize=8,
                markeredgecolor=SURFACE, markeredgewidth=1.6, label=label, zorder=4)
        for xx, v in zip(x, vals):
            if not np.isnan(v):
                ax.annotate("%.2f" % v, (xx, v), textcoords="offset points",
                            xytext=(0, 10), ha="center", fontsize=8, color=INK)

    ax.set_xticks(x, ["%s\nn=%d" % (l, counts.reindex(BIN_LABELS)[l]) for l in BIN_LABELS],
                  fontsize=8)
    ax.set_ylim(0, 1.0)
    ax.set_ylabel("accuracy on the test split")
    ax.set_xlabel("OCR damage to that meme's caption")
    ax.set_title("Accuracy falls as OCR error rises; the same memes stay learnable from gold text")
    ax.legend(loc="lower left")
    ax.set_axisbelow(True); ax.grid(axis="y", alpha=0.9); ax.grid(axis="x", visible=False)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "fig12_accuracy_vs_cer.png"), dpi=200)
    plt.close(fig)

    print("\nACCURACY BY OCR DAMAGE (test split)")
    for label in BIN_LABELS:
        row = merged[merged["bin"] == label]
        if not len(row):
            continue
        line = "  %-26s n=%3d   OCR %.3f" % (label.replace(chr(10), " "), len(row),
                                             row["ocr_correct"].mean())
        if "gold_correct" in row.columns:
            line += "   gold %.3f   delta %+.3f" % (
                row["gold_correct"].mean(), row["ocr_correct"].mean() - row["gold_correct"].mean())
        print(line)


def main(args):
    here = os.path.dirname(os.path.abspath(__file__))
    rs = lambda p: p if os.path.isabs(p) else os.path.abspath(os.path.join(here, p))
    err_path, out = rs(args.errors), rs(args.out)
    os.makedirs(out, exist_ok=True)

    if not os.path.isfile(err_path):
        raise SystemExit("Missing {}. Run make_ocr_splits.py first.".format(err_path))
    err = pd.read_csv(err_path)

    print("OCR ERROR SUMMARY  (n=%d memes)" % len(err))
    for name, series in [("CER", err["cer"]), ("WER", err["wer"])]:
        print("  %-4s mean %.3f  median %.3f  p90 %.3f"
              % (name, series.mean(), series.median(), series.quantile(0.9)))
    print("  memes with CER > 0.5 : %.1f%%" % (100 * (err["cer"] > 0.5).mean()))
    print("  OCR/gold length      : %.2fx" % (err["ocr_words"].sum() / err["gold_words"].sum()))

    fig_cer_distribution(err, out)
    print("\nWrote fig11_cer_distribution.png")

    if args.predictions and os.path.isfile(rs(args.predictions)):
        ocr_preds = pd.read_csv(rs(args.predictions))
        gold_preds = (pd.read_csv(rs(args.gold_predictions))
                      if args.gold_predictions and os.path.isfile(rs(args.gold_predictions))
                      else None)
        fig_accuracy_vs_cer(err, ocr_preds, gold_preds, out)
        print("Wrote fig12_accuracy_vs_cer.png")
    else:
        print("\nNo OCR-caption predictions yet, so fig12 (accuracy vs CER) is skipped.")
        print("Run MAF on the _ocr splits, then re-run with --predictions <that CSV>.")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="OCR error-rate evidence")
    p.add_argument("--errors", default="../Dataset/caption_error_rates.csv")
    p.add_argument("--predictions", default=None,
                   help="predictions CSV from the MAF run trained on _ocr captions")
    p.add_argument("--gold_predictions", default="../MIMOSA on MAF/predictions_maf_fixed_sched.csv")
    p.add_argument("--out", default="../Analysis")
    main(p.parse_args())
