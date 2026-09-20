import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

OUT = os.path.dirname(os.path.abspath(__file__))

BLUE, ORANGE, AQUA, YELLOW = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
VIOLET, RED = "#4a3aa7", "#e34948"
INK, INK2, MUTED = "#0b0b0b", "#52514e", "#9a9892"
GRID, SURFACE = "#e8e7e3", "#fcfcfb"
FROZEN, TRAINED = "#eef2f7", "#fdece4"

plt.rcParams.update({"font.family": "DejaVu Sans", "figure.facecolor": SURFACE})


def box(ax, x, y, w, h, label, face, edge, fontsize=8.2, weight="normal", text=INK):
    ax.add_patch(FancyBboxPatch(
        (x, y), w, h, boxstyle="round,pad=0.004,rounding_size=0.018",
        linewidth=1.3, facecolor=face, edgecolor=edge, zorder=3, clip_on=False))
    ax.text(x + w / 2, y + h / 2, label, ha="center", va="center", fontsize=fontsize,
            color=text, zorder=4, weight=weight, linespacing=1.5, clip_on=False)


def arrow(ax, x1, y1, x2, y2, colour=INK2, lw=1.3, scale=11):
    ax.add_patch(FancyArrowPatch(
        (x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=scale, linewidth=lw,
        color=colour, shrinkA=0, shrinkB=0, zorder=2, clip_on=False))


def line(ax, xs, ys, colour=MUTED, lw=1.1):
    ax.plot(xs, ys, color=colour, linewidth=lw, zorder=1, clip_on=False,
            solid_capstyle="round")


def blank(figsize, xlim, ylim):
    fig, ax = plt.subplots(figsize=figsize)
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.axis("off")
    return fig, ax


# =====================================================================
# Figure 1 - the whole experimental pipeline
# =====================================================================
def pipeline():
    # Margin on every side so no stroke sits on the clip boundary.
    fig, ax = blank((10.4, 5.9), (-0.03, 1.03), (-0.10, 1.02))

    ax.text(0.0, 0.975, "MIMOSA   4,848 Bengali memes   |   5 target classes   |   "
                        "70 / 15 / 15 split", fontsize=9.8, color=INK, weight="bold")

    # ---- inputs and encoders, side by side so the space below stays clear ----
    box(ax, 0.215, 0.855, 0.210, 0.080, "Meme image\n224$\\times$224$\\times$3",
        "#ffffff", MUTED, 8.4)
    box(ax, 0.500, 0.855, 0.210, 0.080, "Bengali caption\ngold  or  OCR",
        "#ffffff", MUTED, 8.4)
    box(ax, 0.215, 0.700, 0.210, 0.098,
        "Frozen ViT tower\nCLIP B/32 \u00b7 L/14 \u00b7 SigLIP", FROZEN, BLUE, 8.2)
    box(ax, 0.500, 0.700, 0.210, 0.098, "BanglaBERT\nfine-tuned", TRAINED, ORANGE, 8.2)
    arrow(ax, 0.320, 0.855, 0.320, 0.800)
    arrow(ax, 0.605, 0.855, 0.605, 0.800)

    # ---- approach columns ---------------------------------------------------
    y0, h, w = 0.330, 0.150, 0.215
    xs = [0.012, 0.258, 0.504, 0.750]
    cols = [
        ("A.  Zero-shot\nprompt similarity\n($\\S$IV-E)", "0 trained", MUTED, "#ffffff", "normal"),
        ("B.  Head-only probe\ncosine classifier\n($\\S$IV-E)", "3.8 K", AQUA, "#ffffff", "normal"),
        ("C.  AdaptFormer PEFT\nparallel bottleneck\n($\\S$IV-D)", "3.18 M  (1.03 %)",
         VIOLET, "#f3f0fb", "bold"),
        ("D.  Fusion baselines\nMAF  /  MuLAD\n($\\S$IV-B, C)", "167.4 M  /  1.4 M",
         ORANGE, "#ffffff", "normal"),
    ]
    centres = [x + w / 2 for x in xs]
    top = y0 + h

    # ---- image path: ViT -> A, B, C -----------------------------------------
    BUS_Y = 0.545
    line(ax, [0.260, 0.260], [0.700, BUS_Y], colour=BLUE, lw=1.3)
    line(ax, [centres[0], centres[2]], [BUS_Y, BUS_Y], colour=BLUE, lw=1.3)
    for cx in centres[:3]:
        arrow(ax, cx, BUS_Y, cx, top + 0.004, colour=BLUE)
    ax.text(0.128, BUS_Y + 0.012, "image features", fontsize=7.4, color=BLUE,
            ha="left", va="bottom", clip_on=False)

    # ---- fusion path: ViT and BanglaBERT merge, then one arrow -> D ---------
    MERGE_Y, VIT_X, TXT_X = 0.585, 0.790, 0.925
    line(ax, [0.380, 0.380], [0.700, 0.600], colour=BLUE, lw=1.3)
    line(ax, [0.380, VIT_X], [0.600, 0.600], colour=BLUE, lw=1.3)
    line(ax, [VIT_X, VIT_X], [0.600, MERGE_Y], colour=BLUE, lw=1.3)

    line(ax, [0.605, 0.605], [0.700, 0.630], colour=ORANGE, lw=1.3)
    line(ax, [0.605, TXT_X], [0.630, 0.630], colour=ORANGE, lw=1.3)
    line(ax, [TXT_X, TXT_X], [0.630, MERGE_Y], colour=ORANGE, lw=1.3)

    line(ax, [VIT_X, TXT_X], [MERGE_Y, MERGE_Y], colour=INK2, lw=1.4)
    arrow(ax, centres[3], MERGE_Y, centres[3], top + 0.004, colour=INK2, lw=1.4)
    ax.text(centres[3] + 0.018, 0.520, "image $+$ caption", fontsize=7.4, color=INK2,
            ha="left", va="center", clip_on=False)

    # ---- the columns themselves ---------------------------------------------
    for cx, x, (title, cost, colour, face, weight) in zip(centres, xs, cols):
        box(ax, x, y0, w, h, title, face, colour, 8.5, weight=weight)
        ax.text(cx, y0 - 0.016, cost, ha="center", va="top", fontsize=7.9, color=INK2,
                clip_on=False)

    # ---- collector bus down to the shared evaluation ------------------------
    COL_Y = 0.238
    for cx, (_, _, colour, _, _) in zip(centres, cols):
        line(ax, [cx, cx], [y0 - 0.056, COL_Y], colour=colour, lw=1.3)
    line(ax, [centres[0], centres[-1]], [COL_Y, COL_Y], colour=MUTED, lw=1.3)
    arrow(ax, 0.4885, COL_Y, 0.4885, 0.208)

    box(ax, 0.055, 0.035, 0.890, 0.172,
        "Identical evaluation protocol\n"
        "same 728-meme test split   |   model selected on validation\n"
        "metrics reported:   Acc, weighted F1, macro F1, MMAE",
        "#ffffff", INK2, 8.6)

    ax.text(0.4885, -0.060,
            "Figure 1.  Experimental pipeline. Blue = image path, orange = caption path. "
            "Only D consumes the caption; A, B and C are image-only.",
            ha="center", fontsize=7.7, color=MUTED, clip_on=False)
    fig.savefig(os.path.join(OUT, "fig_pipeline.png"), dpi=220, bbox_inches="tight")
    plt.close(fig)


# =====================================================================
# Figure 2 - AdaptFormer block, the proposed method
# =====================================================================
def adaptformer():
    fig, ax = blank((9.6, 4.8), (-0.03, 1.03), (-0.06, 1.02))

    SW = 0.255                                  # stack width
    LX, RX = 0.045, 0.495                       # left and right stack origins
    ADAPT_X, ADAPT_W = 0.775, 0.205             # adapter box, ends at 0.980

    ax.text(0.015, 0.960, "Stock CLIP residual block", fontsize=9.6, color=INK,
            weight="bold", clip_on=False)
    ax.text(0.435, 0.960, "AdaptMLP block  (proposed)", fontsize=9.6, color=VIOLET,
            weight="bold", clip_on=False)
    line(ax, [0.400, 0.400], [0.02, 0.925], colour=GRID, lw=1.4)

    def stack(x0, adapted):
        cx = x0 + SW / 2
        plus_x, plus_w = cx - 0.043, 0.086

        box(ax, x0, 0.055, SW, 0.072, "$x_{\\ell}$   (block input)", "#ffffff", MUTED, 8.3)
        box(ax, x0, 0.205, SW, 0.072, "LayerNorm $\\rightarrow$ Attention", FROZEN, BLUE, 8.3)
        box(ax, plus_x, 0.333, plus_w, 0.060, "$\\oplus$", "#ffffff", MUTED, 11)
        box(ax, x0, 0.475, SW, 0.072, "LayerNorm $\\rightarrow$ MLP", FROZEN, BLUE, 8.3)
        box(ax, plus_x, 0.603, plus_w, 0.060, "$\\oplus$", "#ffffff", MUTED, 11)
        box(ax, x0, 0.745, SW, 0.072, "$x_{\\ell+1}$", "#ffffff", MUTED, 8.3)

        arrow(ax, cx, 0.127, cx, 0.205)
        arrow(ax, cx, 0.277, cx, 0.333)
        arrow(ax, cx, 0.393, cx, 0.475)
        arrow(ax, cx, 0.547, cx, 0.603)
        arrow(ax, cx, 0.663, cx, 0.745)

        # residual skips, routed down the left side
        skip_x = x0 - 0.030
        for y_src, y_dst in ((0.091, 0.363), (0.363, 0.633)):
            line(ax, [x0, skip_x], [y_src, y_src])
            line(ax, [skip_x, skip_x], [y_src, y_dst])
            arrow(ax, skip_x, y_dst, plus_x, y_dst, colour=MUTED, lw=1.1)

        if not adapted:
            return cx, plus_x, plus_w

        # ---- the parallel adapter branch ----------------------------------
        # Taps the MLP sub-layer input (the first residual sum) and rejoins at the
        # second sum, which is exactly what Eq. (3) in the report describes.
        a_cx = ADAPT_X + ADAPT_W / 2
        box_y, box_h = 0.458, 0.150
        tap_y, join_y = 0.408, 0.633

        # in: along the MLP sub-layer input, right, then up into the box bottom
        line(ax, [cx, a_cx], [tap_y, tap_y], colour=VIOLET, lw=1.3)
        arrow(ax, a_cx, tap_y, a_cx, box_y, colour=VIOLET, scale=9)

        box(ax, ADAPT_X, box_y, ADAPT_W, box_h,
            "trainable adapter\n"
            "$W_{\\mathrm{down}}\\!: d \\rightarrow r$,   ReLU\n"
            "$W_{\\mathrm{up}}\\!: r \\rightarrow d$,   $\\times\\, s$",
            "#f3f0fb", VIOLET, 7.8)

        # out: up from the box top, then left into the second residual sum
        line(ax, [a_cx, a_cx], [box_y + box_h, join_y], colour=VIOLET, lw=1.3)
        arrow(ax, a_cx, join_y, plus_x + plus_w, join_y, colour=VIOLET)
        return cx, plus_x, plus_w

    stack(LX, False)
    stack(RX, True)

    ax.text(LX + SW / 2, 0.878, "$x \\leftarrow x + \\mathrm{MLP}(\\mathrm{LN}(x))$",
            fontsize=8.8, color=INK, ha="center", clip_on=False)
    ax.text(0.715, 0.878,
            "$x \\leftarrow x + \\mathrm{MLP}(\\mathrm{LN}(x))"
            " + s\\,W_{\\mathrm{up}}(\\mathrm{ReLU}(W_{\\mathrm{down}}(x)))$",
            fontsize=8.8, color=VIOLET, ha="center", clip_on=False)

    ax.text(0.50, -0.035,
            "Figure 2.  AdaptFormer attaches a trainable parallel bottleneck to the frozen "
            "MLP sub-layer ($r = 64$, $s = 0.1$). $W_{\\mathrm{up}}$ is "
            "zero-initialised, so the block is numerically identical to frozen CLIP "
            "at step 0.",
            ha="center", fontsize=7.7, color=MUTED, clip_on=False)
    fig.savefig(os.path.join(OUT, "fig_adaptformer.png"), dpi=220, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    pipeline()
    adaptformer()
    print("wrote fig_pipeline.png and fig_adaptformer.png to", OUT)
