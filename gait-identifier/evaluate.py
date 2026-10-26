"""
Compare the two models on data they never trained on, and make plots.

    python evaluate.py                  # uses data/ and models/
    python evaluate.py --data data_fake

Prints, for each model: cross-validated accuracy and a confusion matrix
(which activities get mixed up), for leave-one-recording-out and, if there
is more than one person, leave-one-person-out. Saves plots to results/:

  peak_histogram.png    model 1's one number, per activity, with its thresholds
  confusion.png         confusion matrices side by side
  feature_scatter.png   the two features model 2's tree leans on most
"""

import argparse
import json
import os

import numpy as np

import learn
from learn import S, features
from train import FULL_DEPTH, KNN_K, PEAK_DEPTH


def load_spec(name):
    path = os.path.join(learn.HERE, "models", f"{name}_model.json")
    if not os.path.exists(path):
        raise SystemExit(f"{path} not found: run train.py first.")
    with open(path) as f:
        return json.load(f)


def main():
    ap = argparse.ArgumentParser(description="Evaluate both gait models")
    ap.add_argument("--data", default=os.path.join(learn.HERE, "data"))
    ap.add_argument("--person", help="only this person's recordings")
    ap.add_argument("--no-plots", action="store_true")
    args = ap.parse_args()

    peak_spec, full_spec = load_spec("peak"), load_spec("full")
    ds = learn.Dataset(learn.load_sessions(args.data, args.person))
    print(f"Data from {args.data}:\n{ds.summary()}\n")

    axis, sign = peak_spec["axis"], peak_spec["sign"]
    Xp, Xf = ds.peak(axis, sign), ds.full()
    contenders = {
        f"Model 1: peak {features.direction_name(axis, sign)} only": (learn.tree_fitter(PEAK_DEPTH, 10), Xp),
        f"Model 2: all IMU, decision tree": (learn.tree_fitter(FULL_DEPTH), Xf),
        f"Model 2: all IMU, KNN k={KNN_K}": (learn.knn_fitter(KNN_K), Xf),
    }

    fold_types = [learn.pick_fold_type(ds)]
    if len(set(ds.person)) > 1:
        fold_types.append("person")

    results = {}
    for ft in fold_types:
        fold_list = learn.folds(ds, ft)
        print("=" * 72)
        print(f"Hold out one {ft} at a time ({len(fold_list)} folds)")
        print("=" * 72)
        for name, (fit, X) in contenders.items():
            pred = learn.cross_val(fit, X, ds.y, fold_list)
            results[(ft, name)] = pred
            print(f"\n{name}: {100 * learn.accuracy(ds.y, pred):.1f}% correct")
            print(learn.confusion_text(ds.y, pred))
        print()

    print("Model 1 rule (trained on all data):")
    print(learn.tree_text(peak_spec["tree"], [f"peak {features.direction_name(axis, sign)} (g)"]))
    full_tree = learn.fit_tree(Xf, ds.y, FULL_DEPTH)
    print(f"\nModel 2 decision tree (trained on all data; board uses: {full_spec['kind']}):")
    print(learn.tree_text(full_tree, features.FULL_NAMES))

    if not args.no_plots:
        make_plots(ds, Xp, Xf, peak_spec, full_tree, results, fold_types[0], axis, sign)


def make_plots(ds, Xp, Xf, peak_spec, full_tree, results, ft, axis, sign):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out = os.path.join(learn.HERE, "results")
    os.makedirs(out, exist_ok=True)
    colors = {"still": "#4C72B0", "walk": "#55A868", "run": "#C44E52", "bike": "#8172B2"}
    present = [lab for lab in S.LABELS if (ds.y == lab).any()]

    # Model 1: histogram of the single feature + learned thresholds.
    fig, ax = plt.subplots(figsize=(8, 4.5))
    bins = np.linspace(Xp.min(), Xp.max(), 60)
    for lab in present:
        ax.hist(Xp[ds.y == lab, 0], bins=bins, alpha=0.6, label=lab, color=colors[lab])
    for t in learn.tree_thresholds(peak_spec["tree"]):
        ax.axvline(t, color="black", linestyle="--", linewidth=1)
    ax.set_xlabel(f"peak acceleration along {features.direction_name(axis, sign)} in a 2 s window (g)")
    ax.set_ylabel("windows")
    ax.set_title("Model 1: one number per window (dashed = learned thresholds)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(out, "peak_histogram.png"), dpi=130)
    plt.close(fig)

    # Confusion matrices.
    names = [n for (f, n) in results if f == ft]
    fig, axes = plt.subplots(1, len(names), figsize=(4.6 * len(names), 4.4))
    for a, name in zip(np.atleast_1d(axes), names):
        labels, m = learn.confusion(ds.y, results[(ft, name)])
        frac = m / np.maximum(m.sum(axis=1, keepdims=True), 1)
        a.imshow(frac, cmap="Blues", vmin=0, vmax=1)
        for i in range(len(labels)):
            for j in range(len(labels)):
                a.text(j, i, m[i, j], ha="center", va="center",
                       color="white" if frac[i, j] > 0.6 else "black")
        a.set_xticks(range(len(labels)), labels)
        a.set_yticks(range(len(labels)), labels)
        a.set_xlabel("predicted")
        a.set_ylabel("true")
        acc = learn.accuracy(ds.y, results[(ft, name)])
        a.set_title(f"{name}\n{100 * acc:.1f}% (hold out one {ft})", fontsize=9)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "confusion.png"), dpi=130)
    plt.close(fig)

    # Model 2: the two features its tree splits on first.
    used = [n["f"] for n in full_tree["nodes"] if "leaf" not in n]
    f1 = used[0] if used else 0
    f2 = next((f for f in used if f != f1), (f1 + 1) % Xf.shape[1])
    fig, ax = plt.subplots(figsize=(6.5, 5))
    for lab in present:
        m = ds.y == lab
        ax.scatter(Xf[m, f1], Xf[m, f2], s=8, alpha=0.6, label=lab, color=colors[lab])
    ax.set_xlabel(features.FULL_NAMES[f1])
    ax.set_ylabel(features.FULL_NAMES[f2])
    ax.set_title("Model 2: the tree's top two features")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(out, "feature_scatter.png"), dpi=130)
    plt.close(fig)
    print(f"\nPlots saved in {out}/")


if __name__ == "__main__":
    main()
