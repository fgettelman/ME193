"""
Width vs depth: train FingerNet at several sizes on the same pictures and
compare test accuracy, size, and how long one picture takes (what matters on
the UNO Q).

    python sweep.py                 # 6 nets, 25 epochs each (~5-10 min)
    python sweep.py --epochs 15

Saves results/sweep.png and results/sweep.json.

WIDTH = channels per layer (how many different features each layer can look
for). DEPTH = number of conv layers (how many times features get combined into
bigger features; each 3x3 conv + pool also grows what one value can 'see').
Weights grow with width squared (a conv has in_channels x out_channels
kernels), but only linearly with depth.
"""

import argparse
import json
import os
import time

import numpy as np

import train as T
from model import n_params, export, S
import fingernet

CONFIGS = [(4, 4), (8, 4), (16, 4), (32, 4), (16, 2), (16, 3), (16, 5)]


def board_ms(model, image, repeats=10):
    """Time the numpy version (what runs on the board) on one picture, on this
    laptop. The UNO Q's cores are several times slower."""
    net = fingernet.FingerNet(export(model))
    net.probs_small(image)
    t = time.perf_counter()
    for _ in range(repeats):
        net.probs_small(image)
    return (time.perf_counter() - t) / repeats * 1000


def main():
    ap = argparse.ArgumentParser(description="Compare widths and depths")
    ap.add_argument("--epochs", type=int, default=25)
    ap.add_argument("--data", default=T.DATA)
    args = ap.parse_args()

    train_items, val_items = T.load_data(args.data)
    if not val_items:
        raise SystemExit("Need pictures with 2+ recordings per count (collect.py).")
    results = []
    for width, depth in CONFIGS:
        t = time.time()
        model, hist, _ = T.train(train_items, val_items, width, depth, args.epochs, verbose=False)
        r = {"width": width, "depth": depth, "params": n_params(model),
             "val_acc": max(hist["val_acc"]), "train_acc": hist["train_acc"][-1],
             "ms": board_ms(model, val_items[0][0]), "train_s": time.time() - t}
        results.append(r)
        print(f"width {width:3d} depth {depth}: {r['params']:8,d} weights  "
              f"test {r['val_acc']:.1%}  train {r['train_acc']:.1%}  "
              f"{r['ms']:.1f} ms/picture (laptop, numpy)", flush=True)

    os.makedirs(T.RESULTS, exist_ok=True)
    with open(os.path.join(T.RESULTS, "sweep.json"), "w") as f:
        json.dump(results, f, indent=1)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    for key, depth_fixed, a in [("width", 4, ax[0]), ("depth", 16, ax[1])]:
        rs = [r for r in results if r["depth" if key == "width" else "width"] == depth_fixed]
        rs.sort(key=lambda r: r[key])
        xs = [r[key] for r in rs]
        a.plot(xs, [r["val_acc"] * 100 for r in rs], "o-", label="test recordings")
        a.plot(xs, [r["train_acc"] * 100 for r in rs], "s--", alpha=0.6, label="train (augmented)")
        for r in rs:
            a.annotate(f"{r['params'] / 1000:.0f}k w\n{r['ms']:.0f} ms", (r[key], r["val_acc"] * 100),
                       textcoords="offset points", xytext=(0, -30), ha="center", fontsize=8)
        a.set(xlabel=key, ylabel="% right", ylim=(0, 105),
              title=f"Changing {key} ({'depth 4' if key == 'width' else 'width 16'})")
        if key == "width":
            a.set_xscale("log", base=2)
        a.set_xticks(xs, [str(x) for x in xs])
        a.grid(alpha=0.3)
        a.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(os.path.join(T.RESULTS, "sweep.png"), dpi=120)
    print(f"Saved {os.path.relpath(T.RESULTS)}/sweep.png and sweep.json")


if __name__ == "__main__":
    main()
