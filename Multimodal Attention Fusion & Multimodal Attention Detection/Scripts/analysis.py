"""
Evidential error analysis for the MAF / MuLAD comparison on MIMOSA.

Consumes the prediction CSVs both frameworks write and produces the figures and
statistics a report needs: paired significance tests, bootstrap confidence intervals,
per-class and per-error-severity breakdowns, the modality ablation, and a reproducibility
check on run-to-run variance.

Everything here works on results that already exist. The OCR-versus-gold caption analysis
is deliberately NOT here - it needs an OCR pass over the same 4,848 MIMOSA memes, which is
a separate run. See `analysis_ocr.py` once those captions exist.

Usage:
    python analysis.py --maf "../MIMOSA on MAF" --mulad "../MIMOSA on MuLAD" --out ../Analysis
"""
import argparse
import glob
import json
import os
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

from sklearn.metrics import accuracy_score, confusion_matrix, f1_score

# ---------------------------------------------------------------- palette
# Validated categorical slots; see the data-viz palette reference. Slot order is the
# colour-vision-deficiency safety mechanism, so it is used in order and never cycled.
BLUE, ORANGE, AQUA, YELLOW = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
MAGENTA, GREEN, VIOLET, RED = "#e87ba4", "#008300", "#4a3aa7", "#e34948"
INK, INK2, MUTED = "#0b0b0b", "#52514e", "#9a9892"
GRID, SURFACE = "#e8e7e3", "#fcfcfb"
# Sequential blue ramp, light -> dark, for magnitude (confusion matrices).
SEQ = LinearSegmentedColormap.from_list(
    "seq_blue", ["#cde2fb", "#9ec5f4", "#5598e7", "#2a78d6", "#1c5cab", "#0d366b"])

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "font.family": "DejaVu Sans", "font.size": 9,
    "axes.edgecolor": GRID, "axes.labelcolor": INK2, "axes.titlecolor": INK,
    "axes.titlesize": 10.5, "axes.titleweight": "bold", "axes.titlelocation": "left",
    "axes.titlepad": 10, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.7,
    "xtick.color": INK2, "ytick.color": INK2, "xtick.labelsize": 8.5, "ytick.labelsize": 8.5,
    "axes.spines.top": False, "axes.spines.right": False,
    "legend.frameon": False, "legend.fontsize": 8.5,
})


def finish(ax, xgrid=True):
    """Recessive grid on the measure axis only; marks sit above it."""
    ax.set_axisbelow(True)
    ax.grid(axis="x" if xgrid else "y", alpha=0.9)
    ax.grid(axis="y" if xgrid else "x", visible=False)


# ---------------------------------------------------------------- statistics
def bootstrap_ci(true, pred, metric, n=2000, seed=42):
    """Percentile bootstrap CI. The test set is fixed, so resample memes with replacement."""
    rng = np.random.RandomState(seed)
    true, pred = np.asarray(true), np.asarray(pred)
    stats = []
    for _ in range(n):
        idx = rng.randint(0, len(true), len(true))
        if len(np.unique(true[idx])) < 2:
            continue
        stats.append(metric(true[idx], pred[idx]))
    return float(np.percentile(stats, 2.5)), float(np.percentile(stats, 97.5))


def mcnemar(true, pred_a, pred_b):
    """Exact McNemar on the two discordant cells.

    Both models score the SAME memes, so the comparison is paired and an unpaired test
    would overstate the uncertainty. Only the disagreements carry information: b = A right
    where B is wrong, c = the reverse.
    """
    a_ok = np.asarray(pred_a) == np.asarray(true)
    b_ok = np.asarray(pred_b) == np.asarray(true)
    b = int(np.sum(a_ok & ~b_ok))
    c = int(np.sum(~a_ok & b_ok))
    try:
        from scipy.stats import binomtest
        p = binomtest(min(b, c), b + c, 0.5).pvalue if (b + c) else 1.0
    except ImportError:
        from scipy.stats import binom_test
        p = binom_test(min(b, c), b + c, 0.5) if (b + c) else 1.0
    return {"only_a_correct": b, "only_b_correct": c, "n_discordant": b + c,
            "p_value": float(p)}


# ---------------------------------------------------------------- loading
def load_predictions(folder, pattern="predictions_*.csv"):
    out = {}
    for path in sorted(glob.glob(os.path.join(folder, "**", pattern), recursive=True)):
        name = os.path.basename(path)[len("predictions_"):-len(".csv")]
        if "subset" in name:
            continue                                 # smoke tests are not results
        out[name] = pd.read_csv(path)
    return out


def metrics_of(frame):
    t, p = frame["true_id"].to_numpy(), frame["pred_id"].to_numpy()
    return {
        "accuracy": accuracy_score(t, p),
        "weighted_f1": f1_score(t, p, average="weighted", zero_division=0),
        "macro_f1": f1_score(t, p, average="macro", zero_division=0),
        "mmae": float(np.mean([np.abs(p[t == c] - t[t == c]).mean() for c in np.unique(t)])),
    }


# ---------------------------------------------------------------- figures
def fig_headline(best, out, names):
    """Emphasis form: the two frameworks' headline metrics with bootstrap CIs."""
    (maf_name, maf), (mul_name, mul) = best
    metrics = [("Accuracy", accuracy_score),
               ("Weighted F1", lambda t, p: f1_score(t, p, average="weighted", zero_division=0)),
               ("Macro F1", lambda t, p: f1_score(t, p, average="macro", zero_division=0))]

    fig, ax = plt.subplots(figsize=(7.2, 3.1))
    ypos, height = np.arange(len(metrics)), 0.34
    for offset, (frame, colour, label) in enumerate(
            [(maf, BLUE, "MAF"), (mul, ORANGE, "MuLAD")]):
        t, p = frame["true_id"].to_numpy(), frame["pred_id"].to_numpy()
        vals = [m(t, p) for _, m in metrics]
        cis = [bootstrap_ci(t, p, m) for _, m in metrics]
        y = ypos + (offset - 0.5) * height
        ax.barh(y, vals, height=height * 0.92, color=colour, label=label, zorder=3)
        ax.errorbar(vals, y, xerr=[[v - lo for v, (lo, _) in zip(vals, cis)],
                                   [hi - v for v, (_, hi) in zip(vals, cis)]],
                    fmt="none", ecolor=INK2, elinewidth=1.1, capsize=3, zorder=4)
        # Label past the CI whisker, never on top of it.
        for yy, v, (_, hi) in zip(y, vals, cis):
            ax.text(hi + 0.016, yy, "%.3f" % v, va="center", fontsize=8.5, color=INK)

    ax.axvline(0.742, color=MUTED, linestyle="--", linewidth=1.2, zorder=2)
    ax.set_yticks(ypos, [m for m, _ in metrics])
    ax.set_xlim(0, 0.95)
    ax.invert_yaxis()
    ax.set_ylim(len(metrics) - 0.5, -0.95)
    ax.text(0.742, -0.88, "published MAF  0.742", fontsize=8, color=INK2,
            ha="center", va="bottom")
    ax.set_title("MAF outperforms MuLAD on every headline metric")
    ax.set_xlabel("score (bars show 95% bootstrap CI)")
    ax.legend(loc="lower right")
    finish(ax)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "fig1_headline.png"), dpi=200)
    plt.close(fig)


def fig_confusions(best, target_names, out):
    """Row-normalised confusion matrices - magnitude, so one sequential hue."""
    fig, axes = plt.subplots(1, 2, figsize=(9.4, 4.0))
    for ax, (name, frame), title in zip(axes, best, ["MAF", "MuLAD"]):
        t, p = frame["true_id"], frame["pred_id"]
        cm = confusion_matrix(t, p, labels=range(len(target_names)))
        norm = cm / cm.sum(axis=1, keepdims=True)
        ax.imshow(norm, cmap=SEQ, vmin=0, vmax=1)
        for i in range(len(target_names)):
            for j in range(len(target_names)):
                ax.text(j, i, "%d\n%.0f%%" % (cm[i, j], 100 * norm[i, j]),
                        ha="center", va="center", fontsize=7.5,
                        color="#ffffff" if norm[i, j] > 0.45 else INK)
        ax.set_xticks(range(len(target_names)), target_names, fontsize=8)
        ax.set_yticks(range(len(target_names)), target_names, fontsize=8)
        ax.set_xlabel("predicted"); ax.set_ylabel("true")
        ax.set_title("%s  -  row-normalised" % title)
        ax.grid(False)
        for spine in ax.spines.values():
            spine.set_visible(False)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "fig2_confusion.png"), dpi=200)
    plt.close(fig)


def fig_per_class(best, target_names, out):
    fig, ax = plt.subplots(figsize=(7.2, 3.3))
    x, width = np.arange(len(target_names)), 0.36
    for offset, ((name, frame), colour, label) in enumerate(
            zip(best, [BLUE, ORANGE], ["MAF", "MuLAD"])):
        f1s = f1_score(frame["true_id"], frame["pred_id"],
                       labels=range(len(target_names)), average=None, zero_division=0)
        pos = x + (offset - 0.5) * width
        ax.bar(pos, f1s, width=width * 0.92, color=colour, label=label, zorder=3)
        for xx, v in zip(pos, f1s):
            ax.text(xx, v + 0.015, "%.2f" % v, ha="center", fontsize=8, color=INK)
    ax.set_xticks(x, target_names)
    ax.set_ylim(0, 1.0)
    ax.set_ylabel("per-class F1")
    ax.set_title("Which classes each framework fails on")
    ax.legend(loc="upper right", ncol=2)
    finish(ax, xgrid=False)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "fig3_per_class_f1.png"), dpi=200)
    plt.close(fig)


def fig_overlap(best, out):
    """Four-way partition of the test set - part-to-whole, so a single stacked bar."""
    (_, maf), (_, mul) = best
    t = maf["true_id"].to_numpy()
    a_ok, b_ok = maf["pred_id"].to_numpy() == t, mul["pred_id"].to_numpy() == t
    cells = [("both correct", int(np.sum(a_ok & b_ok)), AQUA),
             ("MAF only", int(np.sum(a_ok & ~b_ok)), BLUE),
             ("MuLAD only", int(np.sum(~a_ok & b_ok)), ORANGE),
             ("both wrong", int(np.sum(~a_ok & ~b_ok)), MUTED)]
    total = sum(c[1] for c in cells)

    fig, ax = plt.subplots(figsize=(7.4, 1.95))
    left = 0
    for label, count, colour in cells:
        ax.barh([0], [count], left=left, color=colour, height=0.55, zorder=3)
        share = count / total
        if share > 0.05:
            # Narrow segments get smaller type so the label stays inside its own fill.
            ax.text(left + count / 2, 0, "%s\n%d  (%.0f%%)" % (label, count, 100 * share),
                    ha="center", va="center", fontsize=8.5 if share > 0.13 else 7.2,
                    color="#ffffff" if colour != MUTED else INK)
        left += count + total * 0.004        # 2px-equivalent surface gap between segments
    ax.set_xlim(0, total); ax.set_ylim(-0.5, 0.5)
    ax.set_yticks([]); ax.set_xlabel("test memes (n=%d)" % total)
    ax.set_title("Where the two frameworks agree and disagree")
    ax.grid(False)
    for spine in ax.spines.values():
        spine.set_visible(False)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "fig4_error_overlap.png"), dpi=200)
    plt.close(fig)
    return cells


def fig_severity(best, out):
    """How far wrong are the mistakes, on the ordinal label scale MMAE assumes."""
    fig, ax = plt.subplots(figsize=(7.2, 2.9))
    width = 0.36
    dists = range(0, 5)
    for offset, ((name, frame), colour, label) in enumerate(
            zip(best, [BLUE, ORANGE], ["MAF", "MuLAD"])):
        d = np.abs(frame["pred_id"].to_numpy() - frame["true_id"].to_numpy())
        share = [np.mean(d == k) for k in dists]
        pos = np.arange(len(list(dists))) + (offset - 0.5) * width
        ax.bar(pos, share, width=width * 0.92, color=colour, label=label, zorder=3)
        for xx, v in zip(pos, share):
            if v > 0.01:
                ax.text(xx, v + 0.008, "%.0f%%" % (100 * v), ha="center", fontsize=8, color=INK)
    ax.set_xticks(range(len(list(dists))), ["0\n(correct)", "1", "2", "3", "4"])
    ax.set_xlabel("ordinal distance |predicted - true|")
    ax.set_ylabel("share of test set")
    ax.set_title("Error severity, not just error count")
    ax.legend(loc="upper right", ncol=2)
    finish(ax, xgrid=False)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "fig5_error_severity.png"), dpi=200)
    plt.close(fig)


def fig_modality(grid_csv, out):
    """The 39-run ablation: does concatenation fusion beat either modality alone?"""
    if not os.path.isfile(grid_csv):
        return None
    g = pd.read_csv(grid_csv)
    order = ["keras", "selftrained", "fasttext"]
    modalities = [("text", BLUE, "text only"), ("visual", AQUA, "visual only"),
                  ("multimodal", ORANGE, "fusion")]

    fig, ax = plt.subplots(figsize=(7.4, 3.3))
    x, width = np.arange(len(order)), 0.36
    for offset, (mod, colour, label) in enumerate(
            [("text", BLUE, "text only"), ("multimodal", ORANGE, "fusion")]):
        vals = [g[(g["modality"] == mod) & (g["embedding"] == emb)]["WF"].max()
                for emb in order]
        pos = x + (offset - 0.5) * width
        ax.bar(pos, vals, width=width * 0.92, color=colour, label=label, zorder=3)
        for xx, v in zip(pos, vals):
            if not np.isnan(v):
                ax.text(xx, v + 0.012, "%.3f" % v, ha="center", fontsize=8, color=INK)

    # Visual-only carries no embedding, so it is one level across the chart, not a bar
    # repeated three times.
    vis = g[g["modality"] == "visual"]["WF"].max()
    ax.axhline(vis, color=AQUA, linestyle="--", linewidth=1.6, zorder=4)
    # Sit the label in the gap between groups; the right edge collides with a bar label.
    ax.text(0.5, vis + 0.014, "best visual only  %.3f" % vis,
            fontsize=8, color=AQUA, ha="center", va="bottom")
    ax.set_xticks(x, ["Keras 64-d\n(trainable)", "self-trained 300-d\n(frozen)",
                      "FastText 300-d\n(frozen)"])
    ax.set_ylim(0, 0.68)
    ax.set_ylabel("best weighted F1")
    ax.set_title("Fusion beats both single modalities - the opposite of the paper's finding")
    ax.legend(loc="upper right", ncol=2)
    finish(ax, xgrid=False)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "fig6_modality_ablation.png"), dpi=200)
    plt.close(fig)
    return g


def fig_caption_length(best, out):
    """Accuracy against caption length - the gold-caption precursor to the OCR analysis."""
    fig, ax = plt.subplots(figsize=(7.2, 3.1))
    (_, maf), (_, mul) = best
    lengths = maf["Captions"].fillna("").astype(str).str.split().str.len()
    bins = [0, 8, 14, 22, 35, 10 ** 6]
    labels = ["1-8", "9-14", "15-22", "23-35", "36+"]
    idx = pd.cut(lengths, bins=bins, labels=labels, include_lowest=True)

    width = 0.36
    for offset, (frame, colour, label) in enumerate(
            [(maf, BLUE, "MAF"), (mul, ORANGE, "MuLAD")]):
        ok = (frame["pred_id"] == frame["true_id"]).groupby(idx, observed=False).mean()
        pos = np.arange(len(labels)) + (offset - 0.5) * width
        ax.bar(pos, ok.reindex(labels).to_numpy(), width=width * 0.92,
               color=colour, label=label, zorder=3)
    counts = idx.value_counts().reindex(labels)
    ax.set_xticks(range(len(labels)),
                  ["%s\nn=%d" % (l, counts[l]) for l in labels])
    ax.set_xlabel("caption length (words, gold captions)")
    ax.set_ylabel("accuracy")
    ax.set_ylim(0, 1.0)
    ax.set_title("Accuracy against caption length")
    ax.legend(loc="upper right", ncol=2)
    finish(ax, xgrid=False)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "fig7_caption_length.png"), dpi=200)
    plt.close(fig)
    return idx


def fig_reliability(mulad_frame, target_names, out):
    """Is MuLAD's confidence meaningful? Needs the probability columns."""
    cols = ["prob_%s" % n for n in target_names]
    if not all(c in mulad_frame.columns for c in cols):
        return None
    probs = mulad_frame[cols].to_numpy()
    conf = probs.max(axis=1)
    ok = (mulad_frame["pred_id"].to_numpy() == mulad_frame["true_id"].to_numpy())

    edges = np.linspace(conf.min(), 1.0, 7)
    centres, acc, counts = [], [], []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf >= lo) & (conf < hi if hi < 1.0 else conf <= hi)
        if m.sum() >= 10:
            centres.append((lo + hi) / 2); acc.append(ok[m].mean()); counts.append(int(m.sum()))

    fig, ax = plt.subplots(figsize=(5.4, 3.5))
    ax.plot([0, 1], [0, 1], color=MUTED, linestyle="--", linewidth=1.1, zorder=2)
    ax.text(0.62, 0.66, "perfect calibration", fontsize=7.5, color=INK2, rotation=38)
    ax.plot(centres, acc, color=ORANGE, linewidth=2, marker="o", markersize=7,
            markeredgecolor=SURFACE, markeredgewidth=1.6, zorder=4)
    for cx, cy, n in zip(centres, acc, counts):
        ax.annotate("n=%d" % n, (cx, cy), textcoords="offset points", xytext=(0, 9),
                    ha="center", fontsize=7, color=INK2)
    ax.set_xlabel("predicted confidence"); ax.set_ylabel("observed accuracy")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.set_title("MuLAD reliability diagram")
    ax.set_axisbelow(True); ax.grid(alpha=0.9)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "fig8_reliability.png"), dpi=200)
    plt.close(fig)
    return {"mean_confidence": float(conf.mean()), "accuracy": float(ok.mean())}


# ---------------------------------------------------------------- main
def main(args):
    here = os.path.dirname(os.path.abspath(__file__))
    rs = lambda p: p if os.path.isabs(p) else os.path.abspath(os.path.join(here, p))
    maf_dir, mulad_dir, out = rs(args.maf), rs(args.mulad), rs(args.out)
    os.makedirs(out, exist_ok=True)

    maf_preds = load_predictions(maf_dir)
    mul_preds = load_predictions(mulad_dir)
    if not maf_preds or not mul_preds:
        raise SystemExit("Need prediction CSVs in both --maf and --mulad folders.")

    target_names = ["NoAg", "GAg", "PAg", "RAg", "Oth"]
    for folder in (mulad_dir, maf_dir):
        for path in glob.glob(os.path.join(folder, "**", "results_*.json"), recursive=True):
            names = json.load(open(path, encoding="utf-8")).get("target_names")
            if names:
                target_names = names
                break

    scored = {k: metrics_of(v) for k, v in {**{"MAF/" + k: v for k, v in maf_preds.items()},
                                            **{"MuLAD/" + k: v for k, v in mul_preds.items()}}.items()}
    maf_best = max(maf_preds, key=lambda k: metrics_of(maf_preds[k])["weighted_f1"])
    mul_best = max(mul_preds, key=lambda k: metrics_of(mul_preds[k])["weighted_f1"])
    best = [(maf_best, maf_preds[maf_best]), (mul_best, mul_preds[mul_best])]

    if not (best[0][1]["image_name"].values == best[1][1]["image_name"].values).all():
        raise SystemExit("Prediction files are not row-aligned; paired tests would be invalid.")

    print("Best MAF  : %-28s WF %.3f" % (maf_best, metrics_of(best[0][1])["weighted_f1"]))
    print("Best MuLAD: %-28s WF %.3f" % (mul_best, metrics_of(best[1][1])["weighted_f1"]))
    print()

    t = best[0][1]["true_id"].to_numpy()
    mc = mcnemar(t, best[0][1]["pred_id"].to_numpy(), best[1][1]["pred_id"].to_numpy())

    # Reproducibility: identical configuration run twice (separate processes, same seed).
    repro = None
    pair = [k for k in mul_preds if k in ("mulad_proposed", "mulad_cnn_vgg16_keras")]
    if len(pair) == 2:
        a, b = (metrics_of(mul_preds[k])["weighted_f1"] for k in pair)
        agree = float(np.mean(mul_preds[pair[0]]["pred_id"].to_numpy()
                              == mul_preds[pair[1]]["pred_id"].to_numpy()))
        repro = {"runs": pair, "wf1": [a, b], "abs_gap": abs(a - b), "prediction_agreement": agree}

    print("Writing figures...")
    fig_headline(best, out, target_names)
    fig_confusions(best, target_names, out)
    fig_per_class(best, target_names, out)
    cells = fig_overlap(best, out)
    fig_severity(best, out)
    grid = fig_modality(os.path.join(mulad_dir, "mulad_grid.csv"), out)
    fig_caption_length(best, out)
    calib = fig_reliability(best[1][1], target_names, out)

    summary = {
        "best_maf": {"run": maf_best, **metrics_of(best[0][1])},
        "best_mulad": {"run": mul_best, **metrics_of(best[1][1])},
        "bootstrap_ci_weighted_f1": {
            "MAF": bootstrap_ci(t, best[0][1]["pred_id"],
                                lambda a, b: f1_score(a, b, average="weighted", zero_division=0)),
            "MuLAD": bootstrap_ci(t, best[1][1]["pred_id"],
                                  lambda a, b: f1_score(a, b, average="weighted", zero_division=0)),
        },
        "mcnemar_maf_vs_mulad": mc,
        "error_overlap": {label: count for label, count, _ in cells},
        "reproducibility": repro,
        "calibration": calib,
        "all_runs": scored,
        "target_names": target_names,
    }
    with open(os.path.join(out, "analysis_summary.json"), "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2, ensure_ascii=False)

    print()
    print("=" * 78)
    print("McNemar, best MAF vs best MuLAD (paired, n=%d)" % len(t))
    print("  MAF right / MuLAD wrong : %d" % mc["only_a_correct"])
    print("  MuLAD right / MAF wrong : %d" % mc["only_b_correct"])
    print("  p = %.3g %s" % (mc["p_value"],
                             "(significant)" if mc["p_value"] < 0.05 else "(not significant)"))
    if repro:
        print()
        print("Reproducibility - identical config and seed, two runs")
        print("  %s vs %s" % tuple(repro["runs"]))
        print("  WF1 %.3f vs %.3f  -> gap %.3f, predictions agree on %.1f%%"
              % (repro["wf1"][0], repro["wf1"][1], repro["abs_gap"],
                 100 * repro["prediction_agreement"]))
        print("  Differences below ~%.3f WF1 elsewhere are within run-to-run noise."
              % repro["abs_gap"])
    print("=" * 78)
    print("\nWrote figures + analysis_summary.json to", out)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="MAF / MuLAD evidential error analysis")
    p.add_argument("--maf", default="../MIMOSA on MAF")
    p.add_argument("--mulad", default="../MIMOSA on MuLAD")
    p.add_argument("--out", default="../Analysis")
    main(p.parse_args())
