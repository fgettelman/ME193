"""
Train the stairs classifier and the navigator on the laptop, from the walks
in data/ (python sync.py copies them off the board), and save plots.

    python train.py              # stairs (if there are recordings) + navigator
    python train.py --push       # ...and put the models on the board
    python train.py --no-search  # navigator with the default feature weights

The board can train too (Train tab), with a smaller weight search. Here it
tries 81 weightings of the navigator's features and keeps the one with the
best leave-one-walk-out balanced accuracy.

Saves results/:
  routes.png      every walk's dead-reckoned path, coloured by instruction
  confusion.png   leave-one-walk-out: true vs predicted instruction
  timeline.png    for each walk: the reviewed instruction vs what the
                  navigator would say (trained on the other walks)
"""

import argparse
import os

import numpy as np

import common
import deploy
import navcore
from common import S, st

RESULTS = os.path.join(common.HERE, "results")
COLORS = {"straight": "#8a948e", "left": "#2563eb", "right": "#ea580c",
          "stairs_up": "#16a34a", "stairs_down": "#9333ea", "arrived": "#dc2626"}


def plots(store, data, result):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    os.makedirs(RESULTS, exist_ok=True)

    # Paths
    keys = sorted({k for k, *_ in data})
    fig, axes = plt.subplots(1, len(keys), figsize=(6 * len(keys), 6), squeeze=False)
    for ax, key in zip(axes[0], keys):
        for k, _, _, wid in data:
            if k != key:
                continue
            rows = store.track(wid)["rows"]
            labs = navcore.label_track(rows, store.labels(wid)["events"])
            xs, ys = [-r["y"] for r in rows], [r["x"] for r in rows]
            for i in range(1, len(rows)):
                if labs[i]:
                    ax.plot(xs[i - 1:i + 1], ys[i - 1:i + 1], color=COLORS[labs[i]], lw=2.5)
            ax.annotate(store.meta(wid).get("person", ""), (xs[-1], ys[-1]), fontsize=8)
        ax.plot(0, 0, "ks")
        ax.set_title(key)
        ax.set_aspect("equal")
        ax.set_xlabel("m (left of the start direction is -)")
        ax.set_ylabel("m forward from the start")
    handles = [plt.Line2D([], [], color=c, lw=3, label=a) for a, c in COLORS.items()]
    fig.legend(handles=handles, loc="lower center", ncol=6)
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    fig.savefig(os.path.join(RESULTS, "routes.png"), dpi=120)
    plt.close(fig)

    # Confusion matrix
    conf = result.get("confusion") or {}
    if conf:
        acts = [a for a in S.ACTIONS if any(k.startswith(a + "|") for k in conf)]
        M = np.array([[conf.get(f"{t}|{p}", 0) for p in acts] for t in acts], float)
        Mn = M / np.maximum(M.sum(1, keepdims=True), 1)
        fig, ax = plt.subplots(figsize=(6, 5))
        ax.imshow(Mn, cmap="Greens", vmin=0, vmax=1)
        for i in range(len(acts)):
            for j in range(len(acts)):
                ax.text(j, i, int(M[i, j]), ha="center", va="center",
                        color="white" if Mn[i, j] > 0.5 else "black")
        ax.set_xticks(range(len(acts)), acts, rotation=30)
        ax.set_yticks(range(len(acts)), acts)
        ax.set_xlabel("navigator said")
        ax.set_ylabel("reviewed label")
        ax.set_title(f"Leave-one-walk-out, balanced accuracy {result['balanced_accuracy']:.0%}")
        fig.tight_layout()
        fig.savefig(os.path.join(RESULTS, "confusion.png"), dpi=120)
        plt.close(fig)

    # Timelines: truth vs held-out prediction
    w = result["weights"]
    fig, axes = plt.subplots(len(data), 1, figsize=(10, 0.9 * len(data) + 1), squeeze=False)
    for ax, (i, (key, X, y, wid)) in zip(axes[:, 0], enumerate(data)):
        others = [(Xo, yo) for j, (k2, Xo, yo, _) in enumerate(data) if j != i and k2 == key]
        ax.set_yticks([0.25, 0.75], ["guess", "label"])
        ax.set_xticks([])
        ax.set_ylabel(wid[-18:], rotation=0, ha="right", fontsize=7)
        for j, lab in enumerate(y):
            ax.axvspan(j, j + 1, 0.5, 1, color=COLORS[lab], lw=0)
        if others:
            pred = navcore.predict_many(([r for Xo, _ in others for r in Xo],
                                         [l for _, yo in others for l in yo]), X, w)
            for j, lab in enumerate(pred):
                ax.axvspan(j, j + 1, 0, 0.5, color=COLORS[lab], lw=0)
        ax.set_xlim(0, max(len(y), 1))
    axes[-1, 0].set_xlabel("along the walk (one sample per ~0.4 m)")
    fig.tight_layout()
    fig.savefig(os.path.join(RESULTS, "timeline.png"), dpi=120)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description="Train the stairs classifier and the navigator")
    ap.add_argument("--push", action="store_true", help="put the models on the board")
    ap.add_argument("--host", default=deploy.BOARD_HOST)
    ap.add_argument("--no-search", action="store_true")
    args = ap.parse_args()
    store = st.Store(common.DATA)

    print("--- Stairs classifier ---")
    if store.stairs_recordings():
        res = store.train_stairs()
        print(res if not res.get("ok") else
              f"strides {res['strides']}, accuracy on held-out recordings {res['accuracy']:.0%} "
              f"(per kind {res['recall']})")
        for r in res.get("too_short", []):
            print(f"   too short to use (no full stride): {r['id']} ({r['seconds']} s)")
    else:
        print("No stairs recordings (Train tab -> Stairs on the phone); stairs stay 'flat'.")

    print("\n--- Navigator ---")
    res = store.train_policy("none" if args.no_search else "full")
    if not res.get("ok"):
        raise SystemExit(res.get("error"))
    print(f"{res['walks']} walks, {res['samples']} samples, weights {res['weights']}")
    if res.get("skipped_over_limit"):
        print(f"Left out (only {S.MAX_WALKS} walks allowed): {res['skipped_over_limit']}")
    if res.get("balanced_accuracy") is None:
        print(res.get("note"))
    else:
        print(f"Leave-one-walk-out balanced accuracy: {res['balanced_accuracy']:.0%}")
        for a, r in res["recall"].items():
            print(f"   {a:12s} {r:.0%}")
    data, _ = store.training_set(retrack=False)
    plots(store, data, res)
    print(f"Plots in {os.path.relpath(RESULTS)}/")

    if args.push:
        import sync
        sync.push(args.host, models=True)


if __name__ == "__main__":
    main()
