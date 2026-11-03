"""
Train the step-counting neural net with PyTorch and save it for the board.

    python train.py                  # everyone's recordings in data/
    python train.py --person fiona   # only fiona's
    python train.py --epochs 80

1. Loads the recordings and labels every step automatically (learn.py).
2. Trains the net (net.py) to say how many steps are in each 2 s window and
   whether you're still, walking or running.
3. Replays every recording through the finished net exactly like the board
   does, and compares its step total with the labels (and with your own
   count, for counted recordings).
4. Saves models/step_model.json and copies it to unoq_app/python/models/ so
   it goes to the board with the app.

For a fair test on recordings the net never saw, run evaluate.py.
"""

import argparse
import json
import os
import shutil
import time

import numpy as np
import torch
import torch.nn.functional as F

import learn
from learn import S
from net import StepNet, check_export, export, to_inputs
import counter
import stepnet

EPOCHS = 60
BATCH = 64
STRETCH = (0.75, 1.3)        # train on 0.75x to 1.3x the recorded cadence
CLASS_WEIGHT = 0.5           # how much the activity matters next to the step count


def fit(sessions, epochs=EPOCHS, seed=0, log=print):
    """Train a fresh net on these sessions. Returns (model, mean, std)."""
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    X0, _, _, _ = learn.windows(sessions)
    x0 = np.stack([stepnet.inputs(w) for w in X0])
    mean = x0.mean(axis=(0, 2)).astype(np.float32)
    std = (x0.std(axis=(0, 2)) + 1e-6).astype(np.float32)

    model = StepNet()
    opt = torch.optim.Adam(model.parameters(), lr=3e-3)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
    for epoch in range(epochs):
        # Fresh, randomly stretched windows every epoch.
        X, steps, act, _ = learn.windows(sessions, stretch=STRETCH, rng=rng)
        x, y_steps, y_act = to_inputs(X, mean, std), torch.from_numpy(steps), torch.from_numpy(act)
        model.train()
        order = torch.randperm(len(x))
        total = 0.0
        for i in range(0, len(x), BATCH):
            b = order[i:i + BATCH]
            pred_steps, logits = model(x[b])
            loss = F.smooth_l1_loss(pred_steps, y_steps[b])
            known = y_act[b] >= 0
            if known.any():
                loss = loss + CLASS_WEIGHT * F.cross_entropy(logits[known], y_act[b][known])
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += loss.item() * len(b)
        sched.step()
        if log and (epoch % 10 == 0 or epoch == epochs - 1):
            log(f"  epoch {epoch:3d}  loss {total / len(x):.3f}")
    model.eval()
    return model, mean, std


def replay(net, s):
    """Run one recording through the board's counter, a window every 0.5 s,
    like the live app. Returns (steps counted, activity guesses per window)."""
    lo, hi = learn.usable_range(s)
    c = counter.StepCounter(net)
    guesses = []
    for start in range(lo, hi - S.WINDOW_N + 1, S.STEP_N):
        guesses.append(c.update(s["imu"][start:start + S.WINDOW_N])["raw"])
    return c.steps, guesses


def target(s):
    """(steps the labels say, steps you counted or None) for the part the
    counter sees."""
    lo, hi = learn.usable_range(s)
    labelled = s["cum"][hi] - s["cum"][lo + S.WINDOW_N - S.STEP_N]
    counted = s["counted"]["steps"] if s["counted"] else None
    if counted is not None:
        # Your count is for the whole recording; scale to the counted part.
        whole = s["cum"][hi] - s["cum"][lo]
        counted = counted * (labelled / whole) if whole > 0 else 0.0
    return labelled, counted


def report(net, sessions, title):
    print(title)
    print(f"  {'recording':32s} {'net':>6s} {'labels':>7s} {'you':>6s}   activity right")
    rows = []
    for s in sessions:
        got, guesses = replay(net, s)
        labelled, counted = target(s)
        right = np.mean([g == s["label"] for g in guesses]) if not s["counted"] else float("nan")
        you = f"{counted:6.1f}" if counted is not None else "     -"
        act = f"{100 * right:5.0f}%" if not np.isnan(right) else "     -"
        print(f"  {s['name']:32s} {got:6.1f} {labelled:7.1f} {you}   {act}")
        rows.append((s, got, labelled, counted, right))
    moving = [(g, t) for s, g, t, _, _ in rows if s["label"] != "still" and t > 0]
    if moving:
        err = np.mean([abs(g - t) / t for g, t in moving])
        print(f"  Walk/run recordings: off from the labels by {100 * err:.1f}% on average")
    mine = [(g, c) for _, g, _, c, _ in rows if c]
    if mine:
        err = np.mean([abs(g - c) / c for g, c in mine])
        print(f"  Counted recordings:  off from YOUR count by {100 * err:.1f}% on average")
    return rows


def main():
    ap = argparse.ArgumentParser(description="Train the step-counting net")
    ap.add_argument("--data", default=learn.DATA_DIR)
    ap.add_argument("--person", help="train only on this person's recordings")
    ap.add_argument("--epochs", type=int, default=EPOCHS)
    args = ap.parse_args()

    sessions = learn.load_sessions(args.data, args.person)
    if not sessions:
        raise SystemExit("No recordings in data/ yet. Record some:  python guided_train.py")
    missing = [lab for lab in S.LABELS if not any(s["label"] == lab for s in sessions)]
    if missing:
        raise SystemExit(f"No recordings of {', '.join(missing)} yet. Record them first.")

    who = args.person or "everyone"
    print(f"Recordings ({who}):")
    for lab in S.LABELS:
        mine = [s for s in sessions if s["label"] == lab]
        steps = sum(s["cum"][-1] for s in mine)
        print(f"  {lab:6s} {len(mine)} recordings, {steps:5.0f} steps found by the labeler")
    X, _, _, _ = learn.windows(sessions)
    print(f"  {len(X)} windows of {S.WINDOW_S:.0f} s\n")

    print(f"Training the net ({args.epochs} epochs)...")
    t0 = time.time()
    model, mean, std = fit(sessions, args.epochs)
    print(f"  done in {time.time() - t0:.0f} s\n")

    lengths, how = learn.step_lengths(sessions, args.person)
    spec = export(model, mean, std, {
        "step_length_m": lengths,
        "person": who,
        "recordings": [s["name"] for s in sessions],
    })
    diff = check_export(model, spec, X[::max(1, len(X) // 20)], mean, std)
    print(f"Numpy copy for the board matches PyTorch to within {diff:.1e} steps.")
    if diff > 1e-3:
        raise SystemExit("The board's numpy version disagrees with PyTorch; not saving.")

    report(stepnet.StepNet(spec), sessions, "\nReplaying the training recordings like the board does:")
    print("\nStep length: " + ", ".join(f"{a} {lengths[a]:.2f} m ({how[a]})" for a in ("walk", "run")))

    out = os.path.join(learn.HERE, "models", "step_model.json")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w") as f:
        json.dump(spec, f)
    real = os.path.abspath(args.data) == learn.DATA_DIR
    if real:
        board_dir = os.path.join(learn.HERE, "unoq_app", "python", "models")
        os.makedirs(board_dir, exist_ok=True)
        shutil.copy(out, os.path.join(board_dir, "step_model.json"))
    print("\nSaved models/step_model.json"
          + (" (also copied to unoq_app/python/models/ for the board)." if real else "."))
    print("Next: python evaluate.py   tests it on recordings it never saw.")


if __name__ == "__main__":
    main()
