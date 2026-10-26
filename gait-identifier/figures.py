"""
Make the two report figures in results/.

    python figures.py                 # uses fiona's recordings and models/
    python figures.py --person everyone

  system_diagram.png    how data flows: sensor -> board -> models -> laptop
  model_comparison.png  how each model decides, and how accurate each one is
                        on recordings it never trained on
"""

import argparse
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt                   # noqa: E402
import numpy as np                                # noqa: E402
from matplotlib.patches import FancyBboxPatch     # noqa: E402

import learn                                      # noqa: E402
from learn import S, features                     # noqa: E402
from models import tree_predict                   # noqa: E402
from train import FULL_DEPTH, KNN_K, PEAK_DEPTH   # noqa: E402

OUT = os.path.join(learn.HERE, "results")
INK, INK2, MUTED, GRID = "#0b0b0b", "#52514e", "#8a8984", "#e4e3df"
SURFACE = "#fcfcfb"
# Fixed activity -> color/marker, so each activity looks the same in every panel.
COLOR = {"still": "#2a78d6", "walk": "#eb6834", "run": "#1baf7a", "bike": "#6250d6"}
MARKER = {"still": "o", "walk": "s", "run": "^", "bike": "D"}
M1, M2 = "#8a8984", "#2a78d6"          # model 1 gray, model 2 blue

plt.rcParams.update({
    "font.family": "sans-serif", "font.size": 10, "text.color": INK,
    "axes.edgecolor": GRID, "axes.labelcolor": INK2, "axes.titlesize": 12,
    "axes.titleweight": "bold", "axes.titlelocation": "left",
    "xtick.color": INK2, "ytick.color": INK2, "axes.spines.top": False,
    "axes.spines.right": False, "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
})


# --- Figure 1: system diagram -----------------------------------------------

def box(ax, x, y, w, h, title, lines, fill="#ffffff", edge=GRID, title_color=INK):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=0.12",
                                fc=fill, ec=edge, lw=1.5))
    ax.text(x + 0.15, y + h - 0.22, title, ha="left", va="top", fontsize=10.5,
            fontweight="bold", color=title_color)
    ax.text(x + 0.15, y + h - 0.62, "\n".join(lines), ha="left", va="top",
            fontsize=8.6, color=INK2, linespacing=1.45)


def arrow(ax, x1, y1, x2, y2, label=None, dy=0.13):
    ax.annotate("", (x2, y2), (x1, y1),
                arrowprops=dict(arrowstyle="-|>", color=INK2, lw=1.4, shrinkA=2, shrinkB=2))
    if label:
        ax.text((x1 + x2) / 2, (y1 + y2) / 2 + dy, label, ha="center", va="bottom",
                fontsize=8, color=MUTED)


def system_diagram(peak_name, cv):
    fig, ax = plt.subplots(figsize=(13, 6.4))
    ax.set_xlim(0, 13)
    ax.set_ylim(0, 6.4)
    ax.axis("off")
    ax.text(0.1, 6.25, "Gait identifier: how it works", fontsize=15, fontweight="bold", va="top")
    ax.text(0.1, 5.82, "The board classifies on its own every 0.5 s; the laptop only trains "
            "the models and displays the answers.", fontsize=9.5, color=INK2, va="top")

    # Wearable row
    ax.text(0.1, 5.25, "ON THE THIGH (UNO Q)", fontsize=8.5, color=MUTED, fontweight="bold")
    box(ax, 0.1, 3.05, 2.35, 2.0, "MPU-6050 sensor",
        ["3-axis accelerometer (g)", "3-axis gyroscope (deg/s)", "front of thigh,",
         "+X toward the knee"])
    box(ax, 2.95, 3.05, 2.35, 2.0, "Microcontroller",
        ["Arduino sketch", "reads sensor at 100 Hz", "into a ring buffer"])
    box(ax, 5.8, 3.05, 2.35, 2.0, "Linux side (Python)",
        ["every 0.5 s: last 2 s", "= 200 samples x 6", "-> both models", "-> vote over last 3"])
    box(ax, 8.65, 3.85, 2.05, 1.2, "Model 1: peak", [f"1 number: peak {peak_name}",
        f"threshold tree, {100 * cv['peak']:.0f}%"], edge=M1)
    box(ax, 8.65, 2.35, 2.05, 1.2, "Model 2: full", ["46 features, KNN (k=5)",
        f"{100 * cv['full']:.0f}% correct"], edge=M2, title_color=M2)
    arrow(ax, 2.45, 4.05, 2.95, 4.05, "I2C")
    arrow(ax, 5.3, 4.05, 5.8, 4.05, "Bridge")
    arrow(ax, 8.15, 4.3, 8.65, 4.45)
    arrow(ax, 8.15, 3.6, 8.65, 3.05)

    box(ax, 11.2, 3.05, 1.7, 2.0, "Laptop",
        ["live_view.py", "shows both", "answers live"])
    arrow(ax, 10.7, 4.45, 11.2, 4.3)
    arrow(ax, 10.7, 3.0, 11.2, 3.6)
    ax.text(12.05, 2.9, "MQTT over Wi-Fi\n(broker.emqx.io)", ha="center", va="top",
            fontsize=8, color=MUTED)

    # Training row
    ax.plot([0.1, 12.9], [1.95, 1.95], color=GRID, lw=1)
    ax.text(0.1, 1.72, "TRAINING (LAPTOP, ONCE PER PERSON)", fontsize=8.5, color=MUTED,
            fontweight="bold")
    steps = [("guided_train.py", "12 recordings x 60 s:\nstill / walk / run / bike,\n3 paces each"),
             ("Windows", "cut into 2 s windows\nevery 0.5 s; trim\nfirst & last 3 s"),
             ("train.py", "tests each model on\nrecordings it never saw;\nkeeps the best"),
             ("deploy.py", "copies the model\nJSON files to the\nboard over SSH")]
    for i, (t, body) in enumerate(steps):
        x = 0.1 + i * 3.25
        box(ax, x, 0.05, 2.75, 1.45, t, body.split("\n"), fill="#f4f3f0")
        if i < 3:
            arrow(ax, x + 2.75, 0.75, x + 3.25, 0.75)
    arrow(ax, 10.2, 1.5, 10.2, 2.35)
    ax.text(10.32, 1.92, "trained models", fontsize=8, color=MUTED, va="center")
    fig.savefig(os.path.join(OUT, "system_diagram.png"), dpi=160, bbox_inches="tight")
    plt.close(fig)


# --- Figure 2: the two models side by side ----------------------------------

def model_comparison(ds, peak_spec, preds, person):
    axis, sign = peak_spec["axis"], peak_spec["sign"]
    peak_name = features.direction_name(axis, sign)
    Xp = ds.peak(axis, sign)[:, 0]
    Xf = ds.full()
    present = [lab for lab in S.LABELS if (ds.y == lab).any()]

    fig = plt.figure(figsize=(14, 9.6))
    gs = fig.add_gridspec(2, 2, height_ratios=[1, 0.82], hspace=0.42, wspace=0.22, top=0.92)
    fig.suptitle(f"Model 1 vs Model 2 ({person}'s recordings)", x=0.065, y=0.975,
                 ha="left", fontsize=15, fontweight="bold")

    # A. Model 1: one number per window, and the thresholds it learned.
    ax = fig.add_subplot(gs[0, 0])
    lo, hi = 0, max(Xp.max() * 1.05, 2.5)
    cuts = [lo] + learn.tree_thresholds(peak_spec["tree"]) + [hi]
    for a, b in zip(cuts, cuts[1:]):
        lab = tree_predict(peak_spec["tree"], np.array([(a + b) / 2]))
        ax.axvspan(a, b, color=COLOR[lab], alpha=0.10, lw=0)
        ax.text((a + b) / 2, len(present) - 0.35, lab, ha="center", va="bottom",
                fontsize=8, color=INK2, rotation=90 if b - a < 0.25 else 0)
    for t in cuts[1:-1]:
        ax.axvline(t, color=INK2, lw=1, ls="--")
    rng = np.random.default_rng(0)
    for i, lab in enumerate(reversed(present)):
        v = Xp[ds.y == lab]
        ax.scatter(v, i + rng.uniform(-0.28, 0.28, len(v)), s=14, color=COLOR[lab],
                   marker=MARKER[lab], alpha=0.75, lw=0)
    ax.set_yticks(range(len(present)), list(reversed(present)))
    ax.set_ylim(-0.6, len(present) + 0.25)
    ax.set_xlim(lo, hi)
    ax.set_xlabel(f"peak acceleration along {peak_name} in each 2 s window (g)")
    ax.set_title("A.  Model 1 sorts windows by ONE number")
    ax.text(0, -0.2, "Shaded bands = the activity the learned thresholds (dashed) assign. "
            "Walk and bike overlap,\nso no threshold can separate them.",
            transform=ax.transAxes, va="top", fontsize=8.5, color=INK2)
    ax.grid(axis="x", color=GRID, lw=0.6)
    ax.set_axisbelow(True)

    # B. Model 2: two of its 46 features already separate all four.
    ax = fig.add_subplot(gs[0, 1])
    fx, fy = features.FULL_NAMES.index("pitch_deg"), features.FULL_NAMES.index("ax_std")
    for lab in present:
        m = ds.y == lab
        ax.scatter(Xf[m, fx], Xf[m, fy], s=16, color=COLOR[lab], marker=MARKER[lab],
                   alpha=0.75, lw=0, label=lab)
        clusters = [m]
        if lab == "bike":       # real bike (thigh forward) vs air-bike (lying on back)
            clusters = [m & (Xf[:, fx] < 0), m & (Xf[:, fx] >= 0)]
        for c, extra in zip(clusters, ["", " (air-bike, on back)"] if len(clusters) > 1 else [""]):
            if c.any():
                ax.annotate(lab + extra, (np.median(Xf[c, fx]), np.percentile(Xf[c, fy], 97)),
                            xytext=(0, 6), textcoords="offset points", ha="center",
                            fontsize=9, fontweight="bold", color=INK)
    ax.set_yscale("log")
    ax.set_xlabel("thigh angle from gravity, pitch (deg)   [-90 = thigh vertical]")
    ax.set_ylabel("ax shake, std (g, log scale)")
    ax.set_title("B.  Model 2 sees many features at once")
    ax.text(0, -0.2, "Just 2 of its 46 features: shake separates still from moving; thigh "
            "angle separates\nbike (thigh tilted forward) from walk and run (thigh near vertical).",
            transform=ax.transAxes, va="top", fontsize=8.5, color=INK2)
    ax.legend(frameon=False, fontsize=8.5, loc="lower right")
    ax.grid(color=GRID, lw=0.6)
    ax.set_axisbelow(True)

    # C. Accuracy per activity on held-out recordings.
    ax = fig.add_subplot(gs[1, 0])
    groups = present + ["overall"]

    def per_class(pred):
        return [100 * np.mean(pred[ds.y == lab] == lab) for lab in present] + \
               [100 * learn.accuracy(ds.y, pred)]

    acc1, acc2 = per_class(preds["peak"]), per_class(preds["full"])
    x = np.arange(len(groups))
    w = 0.36
    for off, acc, col, name in ((-w / 2 - 0.01, acc1, M1, f"Model 1 (peak {peak_name})"),
                                (w / 2 + 0.01, acc2, M2, "Model 2 (all IMU, KNN)")):
        bars = ax.bar(x + off, acc, w, color=col, label=name)
        for b, v in zip(bars, acc):
            ax.text(b.get_x() + b.get_width() / 2, v + 1.5, f"{v:.0f}", ha="center",
                    fontsize=8.5, color=INK)
    ax.set_xticks(x, groups)
    ax.set_ylim(0, 118)
    ax.set_ylabel("% of windows correct")
    ax.set_title("C.  Accuracy on recordings the model never saw")
    ax.legend(frameon=False, fontsize=8.5, loc="upper left", bbox_to_anchor=(0, 1.02), ncol=2)
    ax.grid(axis="y", color=GRID, lw=0.6)
    ax.set_axisbelow(True)
    ax.text(0, -0.17, "Leave-one-recording-out: train on 11 recordings, test on the 12th, "
            "repeat for all 12.", transform=ax.transAxes, va="top", fontsize=8.5, color=INK2)

    # D. Where model 1 goes wrong vs model 2 (confusion matrices).
    sub = gs[1, 1].subgridspec(1, 2, wspace=0.35)
    for k, (key, title, col) in enumerate((("peak", "Model 1", M1), ("full", "Model 2", M2))):
        ax = fig.add_subplot(sub[0, k])
        labels, m = learn.confusion(ds.y, preds[key])
        frac = m / np.maximum(m.sum(axis=1, keepdims=True), 1)
        ax.imshow(frac, cmap="Blues", vmin=0, vmax=1)
        for i in range(len(labels)):
            for j in range(len(labels)):
                ax.text(j, i, f"{100 * frac[i, j]:.0f}", ha="center", va="center", fontsize=8.5,
                        color="white" if frac[i, j] > 0.6 else INK)
        ax.set_xticks(range(len(labels)), labels, fontsize=8, rotation=45)
        ax.set_yticks(range(len(labels)), labels, fontsize=8)
        ax.set_xlabel("predicted", fontsize=8.5)
        if k == 0:
            ax.set_ylabel("true", fontsize=8.5)
            ax.text(0, -0.42, "Each row: where that activity's windows went (%). Model 1 "
                    "confuses walk and bike;\nModel 2's only real mistake is some bike "
                    "windows called walk.", transform=ax.transAxes, va="top",
                    fontsize=8.5, color=INK2)
        ax.set_title(("D.  " if k == 0 else "") + title, fontsize=11, color=INK)
        for s in ax.spines.values():
            s.set_visible(False)

    fig.savefig(os.path.join(OUT, "model_comparison.png"), dpi=160, bbox_inches="tight")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description="Make report figures")
    ap.add_argument("--person", default="fiona", help="whose recordings ('everyone' for all)")
    args = ap.parse_args()
    person = None if args.person == "everyone" else args.person

    ds = learn.Dataset(learn.load_sessions(os.path.join(learn.HERE, "data"), person))
    with open(os.path.join(learn.HERE, "models", "peak_model.json")) as f:
        peak_spec = json.load(f)
    fold_list = learn.folds(ds, learn.pick_fold_type(ds))
    preds = {
        "peak": learn.cross_val(learn.tree_fitter(PEAK_DEPTH, 10),
                                ds.peak(peak_spec["axis"], peak_spec["sign"]), ds.y, fold_list),
        "full": learn.cross_val(learn.knn_fitter(KNN_K), ds.full(), ds.y, fold_list),
    }
    cv = {k: learn.accuracy(ds.y, p) for k, p in preds.items()}
    os.makedirs(OUT, exist_ok=True)
    system_diagram(features.direction_name(peak_spec["axis"], peak_spec["sign"]), cv)
    model_comparison(ds, peak_spec, preds, args.person)
    print(f"Model 1 {100 * cv['peak']:.1f}%, Model 2 {100 * cv['full']:.1f}%")
    print(f"Saved {OUT}/system_diagram.png and model_comparison.png")


if __name__ == "__main__":
    main()
