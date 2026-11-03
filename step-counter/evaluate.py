"""
Test the step counter on recordings it never saw.

    python evaluate.py                  # everyone in data/
    python evaluate.py --person fiona
    python evaluate.py --epochs 30      # faster, a bit less accurate

Leave-one-recording-out: for each recording, train a fresh net on all the
others, then replay the held-out one like the board does. Reports how far the
step total is from the labels (and from your own count, for counted
recordings) and how often the activity was right. Plots go in results/:

  step_counts.png   steps counted per held-out recording: net vs labels vs you
  step_trace.png    one held-out walk: the thigh swing the labels come from,
                    and the count climbing over time, net vs labels
"""

import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt                   # noqa: E402
import numpy as np                                # noqa: E402

import learn                                      # noqa: E402
from learn import S                               # noqa: E402
import counter                                    # noqa: E402
from net import export                            # noqa: E402
import stepnet                                    # noqa: E402
import train                                      # noqa: E402

RESULTS = os.path.join(learn.HERE, "results")

# Reference palette (dataviz skill), light mode.
SURFACE, INK, INK_2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
NET, LABELS, YOU = "#2a78d6", "#eb6834", "#1baf7a"


def style(ax):
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=INK_2, labelsize=9)
    ax.grid(color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def plot_counts(rows):
    rows = [r for r in rows if r["label"] != "still"] or rows
    names = [r["name"].replace("_", " ") for r in rows]
    y = np.arange(len(rows))
    h = 0.26
    fig, ax = plt.subplots(figsize=(8, 0.55 * len(rows) + 1.6), facecolor=SURFACE)
    style(ax)
    ax.grid(axis="y", visible=False)
    ax.barh(y - h, [r["net"] for r in rows], h * 0.9, color=NET, label="neural net")
    ax.barh(y, [r["labelled"] for r in rows], h * 0.9, color=LABELS, label="automatic labels")
    you = [r["counted"] or 0 for r in rows]
    if any(you):
        ax.barh(y + h, you, h * 0.9, color=YOU, label="your count")
    for i, r in enumerate(rows):
        ax.text(r["net"] + 0.5, i - h, f"{r['net']:.0f}", va="center", fontsize=8, color=INK_2)
    ax.set_yticks(y, names, color=INK)
    ax.invert_yaxis()
    ax.set_xlabel("steps", color=INK_2)
    ax.set_title("Steps counted on recordings the net never saw", color=INK, loc="left", pad=26)
    ax.legend(frameon=False, loc="lower left", bbox_to_anchor=(0, 1.0), ncol=3,
              labelcolor=INK_2, fontsize=9, borderaxespad=0.2)
    fig.tight_layout()
    path = os.path.join(RESULTS, "step_counts.png")
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path


def plot_trace(s, net):
    t = np.arange(len(s["imu"])) / S.SAMPLE_HZ
    lo, hi = learn.usable_range(s)
    c = counter.StepCounter(net)
    tt, counts = [], []
    for start in range(lo, hi - S.WINDOW_N + 1, S.STEP_N):
        c.update(s["imu"][start:start + S.WINDOW_N])
        tt.append((start + S.WINDOW_N) / S.SAMPLE_HZ)
        counts.append(c.steps)
    first = lo + S.WINDOW_N - S.STEP_N
    labelled = s["cum"][(np.array(tt) * S.SAMPLE_HZ).astype(int)] - s["cum"][first]

    fig, (a1, a2) = plt.subplots(2, 1, figsize=(8, 5.5), sharex=True, facecolor=SURFACE,
                                 gridspec_kw={"height_ratios": [1, 1.3]})
    for ax in (a1, a2):
        style(ax)
    swing = learn.thigh_swing(s["imu"])
    a1.plot(t, swing, color=INK_2, linewidth=1)
    starts = learn.stride_starts(s["imu"])
    a1.plot(t[starts], swing[starts], "o", color=LABELS, markersize=5,
            markeredgecolor=SURFACE, markeredgewidth=1.5)
    a1.set_ylabel("thigh swing (deg/s)", color=INK_2)
    a1.set_title(f"{s['name']} (held out): each dot = 1 stride = 2 steps",
                 color=INK, loc="left", fontsize=11)

    a2.plot(tt, labelled, color=LABELS, linewidth=2)
    a2.plot(tt, counts, color=NET, linewidth=2)
    a2.text(tt[-1], counts[-1], f"  net {counts[-1]:.0f}", color=INK, va="bottom", fontsize=9)
    a2.text(tt[-1], labelled[-1], f"  labels {labelled[-1]:.0f}", color=INK, va="top", fontsize=9)
    a2.set_ylabel("steps so far", color=INK_2)
    a2.set_xlabel("time (s)", color=INK_2)
    a2.set_xlim(t[0], t[-1] * 1.12)
    fig.tight_layout()
    path = os.path.join(RESULTS, "step_trace.png")
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path


def main():
    ap = argparse.ArgumentParser(description="Leave-one-recording-out test of the step counter")
    ap.add_argument("--data", default=learn.DATA_DIR)
    ap.add_argument("--person")
    ap.add_argument("--epochs", type=int, default=train.EPOCHS)
    args = ap.parse_args()

    sessions = learn.load_sessions(args.data, args.person)
    if len(sessions) < 2:
        raise SystemExit("Need at least 2 recordings.")
    lengths, _ = learn.step_lengths(sessions, args.person)
    print(f"Leave-one-recording-out: {len(sessions)} nets, one per held-out recording "
          f"({args.epochs} epochs each)\n")
    print(f"  {'held out':32s} {'net':>6s} {'labels':>7s} {'you':>6s}   activity right")
    rows, nets = [], {}
    for k, s in enumerate(sessions):
        others = [o for i, o in enumerate(sessions) if i != k]
        model, mean, std = train.fit(others, args.epochs, seed=k, log=None)
        net = stepnet.StepNet(export(model, mean, std, {"step_length_m": lengths}))
        nets[s["name"]] = net
        got, guesses = train.replay(net, s)
        labelled, counted = train.target(s)
        right = np.mean([g == s["label"] for g in guesses]) if not s["counted"] else None
        rows.append({"name": s["name"], "label": s["label"], "net": got,
                     "labelled": labelled, "counted": counted, "right": right})
        you = f"{counted:6.1f}" if counted is not None else "     -"
        act = f"{100 * right:5.0f}%" if right is not None else "     -"
        print(f"  {s['name']:32s} {got:6.1f} {labelled:7.1f} {you}   {act}", flush=True)

    moving = [r for r in rows if r["label"] != "still" and r["labelled"] > 0]
    print()
    if moving:
        err = np.mean([abs(r["net"] - r["labelled"]) / r["labelled"] for r in moving])
        print(f"Walk/run: step total off from the labels by {100 * err:.1f}% on average")
    still = [r for r in rows if r["label"] == "still"]
    if still:
        print(f"Still:    {max(r['net'] for r in still):.1f} steps at most counted while standing")
    mine = [r for r in rows if r["counted"]]
    if mine:
        err = np.mean([abs(r["net"] - r["counted"]) / r["counted"] for r in mine])
        print(f"Counted:  off from YOUR count by {100 * err:.1f}% on average "
              f"({len(mine)} recordings)")
    acts = [r["right"] for r in rows if r["right"] is not None]
    print(f"Activity: right {100 * np.mean(acts):.1f}% of the time")

    os.makedirs(RESULTS, exist_ok=True)
    print("\nPlots: " + plot_counts(rows))
    walks = [s for s in sessions if s["label"] == "walk"]
    if walks:
        s = max(walks, key=lambda s: s["cum"][-1])
        print("       " + plot_trace(s, nets[s["name"]]))


if __name__ == "__main__":
    main()
