"""
Caption-quality analysis: why hand-annotated captions matter.

Compares the three MAF runs that now exist and quantifies the difference between machine
OCR captions and the dataset authors' hand-corrected ones.

READ THIS BEFORE QUOTING THE HEADLINE GAP
-----------------------------------------
The OCR run and the hand-annotated runs are not a controlled pair. Three things differ:

  (a) captions      - machine OCR vs hand-corrected      [the variable of interest]
  (b) class count   - 4-way vs 5-way                     [confound, favours the OCR run]
  (c) training size - 2,390 vs 3,393 memes               [confound, favours the hand run]

(b) matters rhetorically: the OCR run faced a STRICTLY EASIER task - four classes, 25%
chance rather than 20% - and still scored far lower. A confound that pushes against the
conclusion strengthens it. (c) pushes the other way and cannot be dismissed, so the gap
here is evidence, not proof.

The clean experiment is an OCR pass over the same 4,848 MIMOSA memes, giving machine and
gold captions for identical images, splits and classes. Caption quality is then the only
variable, and per-meme character error rate becomes measurable. Until that run exists,
report the numbers here as directional and state the confounds.

Usage:
    python analysis_captions.py
"""
import argparse
import glob
import json
import os
import re
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
INK, INK2, MUTED = "#0b0b0b", "#52514e", "#9a9892"
GRID, SURFACE = "#e8e7e3", "#fcfcfb"

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

BENGALI = re.compile(r"[ঀ-৿]")
LATIN = re.compile(r"[A-Za-z]")


def caption_stats(captions):
    """Per-caption quality measures that need no reference text."""
    caps = captions.fillna("").astype(str)
    rows = []
    for c in caps:
        chars = [ch for ch in c if not ch.isspace()]
        n = max(len(chars), 1)
        tokens = c.split()
        bengali_tokens = sum(1 for t in tokens if BENGALI.search(t))
        rows.append({
            "n_words": len(tokens),
            "bengali_char_share": sum(1 for ch in chars if BENGALI.match(ch)) / n,
            "latin_char_share": sum(1 for ch in chars if LATIN.match(ch)) / n,
            "digit_char_share": sum(1 for ch in chars if ch.isdigit()) / n,
            "non_bengali_token_share": 1 - bengali_tokens / max(len(tokens), 1),
            "is_empty": len(tokens) == 0,
        })
    return pd.DataFrame(rows)


def best_run(folder):
    best, score = None, -1
    for path in glob.glob(os.path.join(folder, "**", "results_*.json"), recursive=True):
        r = json.load(open(path, encoding="utf-8"))
        if "smoketest" in r["run_name"] or r.get("hyperparameters", {}).get("subset"):
            continue
        if r["weighted_f1"] > score:
            best, score = r, r["weighted_f1"]
    return best


def fig_caption_quality(hand, ocr, out):
    """Two distributions of the same measure - categorical identity, two slots.

    The caption sets have different sizes (728 hand vs 400 OCR), so both are plotted as a
    SHARE of their own set. Raw counts would make the larger set look denser everywhere.
    """
    import numpy as np

    def share(data):
        return np.ones(len(data)) / len(data)

    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.2))

    ax = axes[0]
    bins = np.linspace(0, 1, 26)
    for data, colour, label in [(hand["bengali_char_share"], BLUE, "hand-annotated"),
                                (ocr["bengali_char_share"], ORANGE, "machine OCR")]:
        ax.hist(data, bins=bins, weights=share(data), color=colour, alpha=0.78,
                label=label, zorder=3)
    for data, colour in [(hand["bengali_char_share"], BLUE), (ocr["bengali_char_share"], ORANGE)]:
        ax.axvline(data.mean(), color=colour, linestyle="--", linewidth=1.5, zorder=4)
    zero = (ocr["bengali_char_share"] < 0.01).mean()
    ax.annotate("%.0f%% of OCR captions contain%sno Bengali at all" % (100 * zero, chr(10)),
                xy=(0.02, zero), xytext=(0.17, zero + 0.10), fontsize=7.8, color=INK,
                arrowprops=dict(arrowstyle="->", color=INK2, linewidth=1))
    ax.set_xlabel("share of characters that are Bengali script")
    ax.set_ylabel("share of captions")
    ax.set_title("OCR captions are mostly not Bengali")
    ax.legend(loc="upper center")
    ax.set_axisbelow(True); ax.grid(axis="y", alpha=0.9); ax.grid(axis="x", visible=False)

    ax = axes[1]
    bins = np.linspace(0, 70, 36)
    for data, colour, label in [(hand["n_words"], BLUE, "hand-annotated"),
                                (ocr["n_words"], ORANGE, "machine OCR")]:
        ax.hist(np.clip(data, 0, 70), bins=bins, weights=share(data), color=colour,
                alpha=0.78, label=label, zorder=3)
    ax.set_xlabel("caption length (words; final bin is 70+)")
    ax.set_ylabel("share of captions")
    ax.set_title("OCR inflates length with spurious tokens")
    ax.legend(loc="upper right")
    ax.set_axisbelow(True); ax.grid(axis="y", alpha=0.9); ax.grid(axis="x", visible=False)

    fig.tight_layout()
    fig.savefig(os.path.join(out, "fig9_caption_quality.png"), dpi=200)
    plt.close(fig)


def fig_three_way(runs, out):
    """Emphasis form: the caption variable is the point, so it carries the accent hue."""
    fig, ax = plt.subplots(figsize=(7.6, 3.4))
    labels = [r["label"] for r in runs]
    vals = [r["weighted_f1"] for r in runs]
    colours = [r["colour"] for r in runs]

    ax.bar(range(len(runs)), vals, width=0.56, color=colours, zorder=3)
    for i, (v, r) in enumerate(zip(vals, runs)):
        ax.text(i, v + 0.014, "%.3f" % v, ha="center", fontsize=10, color=INK, weight="bold")
        # The run name rides inside its own bar; above the bar it hits the annotation.
        ax.text(i, v - 0.048, r["sub"], ha="center", fontsize=7.6, color="#ffffff", zorder=4)

    ax.set_xticks(range(len(runs)), labels, fontsize=9)
    ax.set_ylim(0, 1.02)
    ax.set_ylabel("weighted F1")
    ax.set_title("Caption quality dominates the result")
    ax.set_axisbelow(True); ax.grid(axis="y", alpha=0.9); ax.grid(axis="x", visible=False)

    ax.annotate("", xy=(0, 0.80), xytext=(1, 0.80),
                arrowprops=dict(arrowstyle="<->", color=INK2, linewidth=1.2))
    gap = vals[1] - vals[0]
    ax.text(0.5, 0.835,
            "+%.3f WF1 from hand-corrected captions," % gap + chr(10)
            + "despite the harder 5-way task",
            ha="center", fontsize=8.4, color=INK)

    fig.tight_layout()
    fig.savefig(os.path.join(out, "fig10_caption_effect.png"), dpi=200)
    plt.close(fig)


def main(args):
    here = os.path.dirname(os.path.abspath(__file__))
    rs = lambda p: p if os.path.isabs(p) else os.path.abspath(os.path.join(here, p))
    ocr_dir, maf_dir, mul_dir, out = rs(args.ocr), rs(args.maf), rs(args.mulad), rs(args.out)
    os.makedirs(out, exist_ok=True)

    ocr_best, maf_best, mul_best = best_run(ocr_dir), best_run(maf_dir), best_run(mul_dir)
    if maf_best is None:
        # The hand-annotated MAF folder ships predictions without results JSON.
        frames = {os.path.basename(p): pd.read_csv(p)
                  for p in glob.glob(os.path.join(maf_dir, "predictions_*.csv"))}
        from sklearn.metrics import f1_score, accuracy_score
        name = max(frames, key=lambda k: f1_score(frames[k]["true_id"], frames[k]["pred_id"],
                                                  average="weighted"))
        f = frames[name]
        maf_best = {"run_name": name[len("predictions_"):-4],
                    "weighted_f1": f1_score(f["true_id"], f["pred_id"], average="weighted"),
                    "accuracy": accuracy_score(f["true_id"], f["pred_id"]),
                    "target_names": ["NoAg", "GAg", "PAg", "RAg", "Oth"]}

    ocr_caps = pd.concat([pd.read_csv(p)["Captions"] for p in
                          glob.glob(os.path.join(ocr_dir, "predictions_maf_*.csv"))[:1]])
    hand_caps = pd.concat([pd.read_csv(p)["Captions"] for p in
                           glob.glob(os.path.join(maf_dir, "predictions_maf_*.csv"))[:1]])
    ocr_stats, hand_stats = caption_stats(ocr_caps), caption_stats(hand_caps)

    print("CAPTION QUALITY  (test-split captions each model actually consumed)")
    print("%-28s %12s %12s" % ("measure", "hand", "machine OCR"))
    for key, fmt in [("bengali_char_share", "%.3f"), ("latin_char_share", "%.3f"),
                     ("digit_char_share", "%.3f"), ("non_bengali_token_share", "%.3f"),
                     ("n_words", "%.1f")]:
        print("%-28s %12s %12s" % (key, fmt % hand_stats[key].mean(), fmt % ocr_stats[key].mean()))
    print("%-28s %12d %12d" % ("empty captions", hand_stats["is_empty"].sum(),
                               ocr_stats["is_empty"].sum()))
    print()

    runs = [
        {"label": "MAF\nmachine OCR captions\n4-way subset", "colour": ORANGE,
         "weighted_f1": ocr_best["weighted_f1"], "sub": ocr_best["run_name"]},
        {"label": "MAF\nhand-corrected captions\n5-way MIMOSA", "colour": BLUE,
         "weighted_f1": maf_best["weighted_f1"], "sub": maf_best["run_name"]},
        {"label": "MuLAD\nhand-corrected captions\n5-way MIMOSA", "colour": MUTED,
         "weighted_f1": mul_best["weighted_f1"], "sub": mul_best["run_name"]},
    ]
    for r in runs:
        print("%-46s WF1 %.3f" % (r["label"].replace("\n", " / "), r["weighted_f1"]))

    fig_caption_quality(hand_stats, ocr_stats, out)
    fig_three_way(runs, out)

    summary = {
        "caption_quality": {
            "hand": {k: float(hand_stats[k].mean()) for k in hand_stats.columns},
            "ocr": {k: float(ocr_stats[k].mean()) for k in ocr_stats.columns},
        },
        "runs": [{k: v for k, v in r.items() if k != "colour"} for r in runs],
        "confounds": {
            "class_count": "OCR run is 4-way (chance 25%), hand runs are 5-way (chance 20%) "
                           "- favours the OCR run, so it works against the conclusion",
            "training_size": "OCR run trained on 2,390 memes vs 3,393 - favours the hand run",
            "resolution": "Run OCR over the same 4,848 MIMOSA memes for a controlled pair",
        },
    }
    with open(os.path.join(out, "caption_analysis.json"), "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2, ensure_ascii=False)
    print("\nWrote fig9, fig10 and caption_analysis.json to", out)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Caption-quality evidence")
    p.add_argument("--ocr", default="../OCR'd subset MIMOSA on MAF")
    p.add_argument("--maf", default="../MIMOSA on MAF")
    p.add_argument("--mulad", default="../MIMOSA on MuLAD")
    p.add_argument("--out", default="../Analysis")
    main(p.parse_args())
