"""
Machine-learning swing detector: record 10 swings and 10 non-swings with the
paddle (Double Motor IMU), then train a logistic-regression classifier on them.

    python3 train_swing.py               # record 10 + 10, then train
    python3 train_swing.py --fresh       # throw away old recordings first
    python3 train_swing.py --train-only  # retrain on what's already recorded
    python3 train_swing.py --test        # live test: swing and watch it decide

How it works
  1. Record: each trial is a 2 s clip of gyro (x, y, z) + accelerometer
     (x, y, z) at ~50 Hz. The window shows SWING (green) or NO SWING (red);
     press SPACE when ready, it counts down 3-2-1, then records. A SWING with
     no real movement is thrown out and redone. For NO SWING, do the suggested movement
     -- those "almost swings" are what teach the model the difference.
  2. Features: from the 0.4 s around the biggest gyro spike in each clip we
     measure 10 numbers (peak spin, average spin, how jerky it is, peak
     acceleration...). See FEATURES.
  3. Train: logistic regression, written in numpy (gradient descent on the
     log-loss with a little L2). Accuracy is checked with leave-one-out
     cross-validation: train on 19 clips, test on the 1 left out, x20.
  4. Save: swing_model.json (weights + scaling) and swing_training.png
     (plots you can show).

Recordings are kept in swing_data.npz and new runs add to them, so the model
gets better the more you record.
"""

import argparse
import json
import math
import os
import time

import cv2
import numpy as np

import config

HERE = os.path.dirname(os.path.abspath(__file__))
DATA_FILE = os.path.join(HERE, "swing_data.npz")
MODEL_FILE = os.path.join(HERE, "swing_model.json")
PLOT_FILE = os.path.join(HERE, "swing_training.png")

N_EACH = 10            # swings and non-swings per run
RATE_HZ = 50           # IMU sample rate we record at
CLIP_S = 2.0           # length of each recording
SUB_S = 0.4            # window around the gyro peak the features come from
COUNTDOWN_S = 2.1      # 3, 2, 1 (0.7 s each)
REST_S = 0.9           # "Got it" pause between trials
MIN_SWING_SPIN = 300   # a SWING recording whose gyro never gets this high is redone

NO_SWING_MOVES = [
    "Hold the paddle still",
    "Move it slowly to a new spot",
    "Raise it up to hitting height",
    "Tilt / turn it in your hand",
    "Get ready as if a ball is coming (don't swing!)",
]

FEATURES = [
    "gyro peak", "gyro mean", "gyro std", "gyro max jump",
    "accel peak", "accel std", "accel range",
    "gyro |x| max", "gyro |y| max", "gyro |z| max",
]

W, H = 960, 540
GREEN, RED, WHITE, GRAY, YELLOW = (60, 200, 60), (60, 60, 230), (255, 255, 255), (150, 150, 150), (0, 220, 255)


# --- features + model (pure numpy, no hardware needed) ----------------------

def peak_window(clip):
    """The SUB_S seconds of a clip centered on its biggest gyro spike."""
    n = int(SUB_S * RATE_HZ)
    gm = np.linalg.norm(clip[:, :3], axis=1)
    c = int(np.argmax(gm))
    start = min(max(0, c - n // 2), len(clip) - n)
    return clip[start:start + n]


def features(win):
    """10 numbers describing a (samples x 6) window of gyro xyz + accel xyz."""
    g, a = win[:, :3], win[:, 3:]
    gm, am = np.linalg.norm(g, axis=1), np.linalg.norm(a, axis=1)
    return np.array([
        gm.max(), gm.mean(), gm.std(), np.abs(np.diff(gm)).max(),
        am.max(), am.std(), am.max() - am.min(),
        *np.abs(g).max(axis=0),
    ])


def fit_logistic(X, y, l2=0.01, lr=0.1, steps=3000):
    """Logistic regression by gradient descent. Returns (mean, std, w, b)."""
    mu, sd = X.mean(axis=0), X.std(axis=0) + 1e-9
    Z = (X - mu) / sd
    w, b = np.zeros(Z.shape[1]), 0.0
    for _ in range(steps):
        p = 1 / (1 + np.exp(-(Z @ w + b)))
        w -= lr * (Z.T @ (p - y) / len(y) + l2 * w)
        b -= lr * np.mean(p - y)
    return mu, sd, w, b


def predict_proba(model, x):
    mu, sd, w, b = model
    return 1 / (1 + np.exp(-(((x - mu) / sd) @ w + b)))


def leave_one_out(X, y):
    """Predicted probability for each clip from a model that never saw it."""
    probs = np.zeros(len(y))
    for i in range(len(y)):
        keep = np.arange(len(y)) != i
        probs[i] = predict_proba(fit_logistic(X[keep], y[keep]), X[i])
    return probs


def train(clips, labels):
    X = np.array([features(peak_window(c)) for c in clips])
    y = np.asarray(labels, float)
    print(f"\nTraining on {int(y.sum())} swings + {int((1 - y).sum())} non-swings...")

    probs = leave_one_out(X, y)
    pred = probs >= 0.5
    acc = np.mean(pred == y)
    tp, fn = int(np.sum(pred & (y == 1))), int(np.sum(~pred & (y == 1)))
    fp, tn = int(np.sum(pred & (y == 0))), int(np.sum(~pred & (y == 0)))
    print(f"Leave-one-out accuracy: {acc * 100:.0f}%  ({int(acc * len(y))}/{len(y)} clips right)")
    print("                 predicted SWING   predicted NO SWING")
    print(f"  real SWING          {tp:3d}               {fn:3d}")
    print(f"  real NO SWING       {fp:3d}               {tn:3d}")

    mu, sd, w, b = fit_logistic(X, y)       # final model uses every clip
    print("\nWhat the model learned (weight on each standardized feature):")
    for name, wi in sorted(zip(FEATURES, w), key=lambda t: -abs(t[1])):
        print(f"  {name:15s} {wi:+.2f}  {'(more -> swing)' if wi > 0 else '(more -> not a swing)'}")

    with open(MODEL_FILE, "w") as f:
        json.dump({"features": FEATURES, "mean": mu.tolist(), "std": sd.tolist(),
                   "weights": w.tolist(), "bias": float(b), "rate_hz": RATE_HZ,
                   "window_s": SUB_S, "loo_accuracy": float(acc), "n_clips": len(y)}, f, indent=2)
    print(f"\nSaved model -> {os.path.relpath(MODEL_FILE)}")
    save_plot(clips, y, X, probs, acc)
    return (mu, sd, w, b)


def save_plot(clips, y, X, probs, acc):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
    for clip, label in zip(clips, y):
        axes[0].plot(np.arange(len(clip)) / RATE_HZ, np.linalg.norm(clip[:, :3], axis=1),
                     color="#2e9e4f" if label else "#d64545", alpha=0.6, lw=1.4)
    axes[0].set(title="Gyro magnitude per recording", xlabel="time (s)", ylabel="spin")
    axes[0].plot([], [], color="#2e9e4f", label="swing")
    axes[0].plot([], [], color="#d64545", label="no swing")
    axes[0].legend(frameon=False)

    i, j = FEATURES.index("gyro peak"), FEATURES.index("accel peak")
    for label, color, name in ((1, "#2e9e4f", "swing"), (0, "#d64545", "no swing")):
        m = y == label
        axes[1].scatter(X[m, i], X[m, j], color=color, label=name, s=45)
    axes[1].set(title="Two of the features", xlabel="gyro peak", ylabel="accel peak")
    axes[1].legend(frameon=False)

    order = np.argsort(probs)
    axes[2].bar(range(len(probs)), probs[order],
                color=["#2e9e4f" if y[k] else "#d64545" for k in order])
    axes[2].axhline(0.5, color="#888", ls="--", lw=1)
    axes[2].set(title=f"Leave-one-out P(swing) - accuracy {acc * 100:.0f}%",
                xlabel="recordings (sorted)", ylabel="P(swing)", ylim=(0, 1), xticks=[])
    for ax in axes:
        ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(PLOT_FILE, dpi=130)
    print(f"Saved plots -> {os.path.relpath(PLOT_FILE)}")


def load_model():
    with open(MODEL_FILE) as f:
        m = json.load(f)
    return np.array(m["mean"]), np.array(m["std"]), np.array(m["weights"]), m["bias"]


# --- data on disk -------------------------------------------------------------

def load_data():
    if not os.path.exists(DATA_FILE):
        return [], []
    d = np.load(DATA_FILE)
    return list(d["clips"]), list(d["labels"])


def save_data(clips, labels):
    n = int(CLIP_S * RATE_HZ)
    clips = [c[:n] if len(c) >= n else np.vstack([c, np.repeat(c[-1:], n - len(c), axis=0)]) for c in clips]
    np.savez(DATA_FILE, clips=np.array(clips), labels=np.array(labels))


# --- paddle + screen ----------------------------------------------------------

def connect():
    from lelib import doubleMotor
    dm = doubleMotor()
    print("Connecting to Double Motor (paddle)...")
    dm.connect(card_serial=config.CARD_SERIAL, card_color=config.CARD_COLOR)
    dm.device_notification_request(config.NOTIFICATION_DELAY_MS)
    print("Paddle connected.")
    return dm


def read_imu(dm):
    imu = dm.imu_device
    v = (imu.gyroscopeX, imu.gyroscopeY, imu.gyroscopeZ,
         imu.accelerometerX, imu.accelerometerY, imu.accelerometerZ)
    if any(isinstance(x, float) and math.isnan(x) for x in v):
        return None
    return np.array(v, float)


def text(img, msg, y, scale=1.0, color=WHITE, thick=2, x=None):
    (tw, _), _ = cv2.getTextSize(msg, cv2.FONT_HERSHEY_SIMPLEX, scale, thick)
    x = (W - tw) // 2 if x is None else x
    cv2.putText(img, msg, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, color, thick, cv2.LINE_AA)


def draw_trace(img, trace, top=330, height=170, scale=1600.0):
    cv2.rectangle(img, (40, top), (W - 40, top + height), (60, 60, 60), 1)
    if len(trace) > 1:
        xs = np.linspace(40, W - 40, len(trace)).astype(int)
        ys = (top + height - np.clip(np.array(trace) / scale, 0, 1) * height).astype(int)
        cv2.polylines(img, [np.stack([xs, ys], 1)], False, YELLOW, 2, cv2.LINE_AA)
    text(img, "gyro", top + height + 25, 0.6, GRAY, 1, x=40)


def screen(title, color, line2="", line3="", trace=None, progress=""):
    img = np.full((H, W, 3), 25, np.uint8)
    cv2.rectangle(img, (0, 0), (W, 90), color, -1)
    text(img, title, 62, 1.6, WHITE, 4)
    if progress:
        text(img, progress, 130, 0.8, GRAY)
    if line2:
        text(img, line2, 200, 1.4, WHITE, 3)
    if line3:
        text(img, line3, 260, 0.9, WHITE)
    if trace is not None:
        draw_trace(img, trace)
    cv2.imshow("Swing training", img)
    return cv2.waitKey(1) & 0xFF


def record(dm, n_each):
    """Alternate SWING / NO SWING trials. Returns (clips, labels); q stops early."""
    plan = [1, 0] * n_each
    clips, labels = [], []
    k = 0
    while k < len(plan):
        label = plan[k]
        title = "SWING" if label else "NO SWING"
        color = GREEN if label else RED
        how = "Swing the paddle like hitting a ball" if label else \
            NO_SWING_MOVES[(k // 2) % len(NO_SWING_MOVES)]
        progress = f"Recording {k + 1} of {len(plan)}   (q = quit, r = redo last)"

        # Wait until the player is ready, so prompts never run ahead of them.
        key = 0
        while key not in (ord(" "), ord("q"), ord("r")):
            key = screen(title, color, "Press SPACE when ready", how, progress=progress)
            time.sleep(0.02)
        if key == ord("q"):
            return clips, labels
        if key == ord("r"):
            if clips:
                clips.pop(); labels.pop(); k -= 1
            continue

        t_end = time.monotonic() + COUNTDOWN_S
        while (left := t_end - time.monotonic()) > 0:
            key = screen(title, color, f"{math.ceil(left / 0.7)}...", how, progress=progress)
            if key == ord("q"):
                return clips, labels
            if key == ord("r") and clips:
                clips.pop(); labels.pop(); k -= 1
                break
        else:
            clip, trace = [], []
            t_end = time.monotonic() + CLIP_S
            next_t = time.monotonic()
            while time.monotonic() < t_end:
                s = read_imu(dm)
                if s is not None:
                    clip.append(s)
                    trace.append(float(np.linalg.norm(s[:3])))
                screen(title, color, "GO!", how, trace=trace, progress=progress)
                next_t += 1 / RATE_HZ
                time.sleep(max(0.0, next_t - time.monotonic()))
            if len(clip) < int(SUB_S * RATE_HZ):
                print("Too few IMU samples -- is the paddle connected? Retrying this one.")
                continue
            peak = max(trace) if trace else 0.0
            if label == 1 and peak < MIN_SWING_SPIN:
                t_end = time.monotonic() + 1.5
                while time.monotonic() < t_end:
                    screen(title, color, "No swing seen - again", f"peak spin {peak:.0f}", trace=trace,
                           progress=progress)
                continue
            clips.append(np.array(clip)); labels.append(label)
            t_end = time.monotonic() + REST_S
            while time.monotonic() < t_end:
                screen(title, color, "Got it", f"peak spin {peak:.0f}", trace=trace, progress=progress)
            k += 1
    return clips, labels


def live_test(dm, model):
    """Swing and watch the model decide, with its probability."""
    n = int(SUB_S * RATE_HZ)
    buf, trace = [], []
    verdict, verdict_until, p_last = "", 0.0, 0.0
    next_t = time.monotonic()
    while True:
        s = read_imu(dm)
        if s is not None:
            buf.append(s); buf = buf[-n:]
            trace.append(float(np.linalg.norm(s[:3]))); trace = trace[-150:]
            if len(buf) == n:
                gm = np.linalg.norm(np.array(buf)[:, :3], axis=1)
                if np.argmax(gm) == n // 2 and gm[n // 2] > 50:   # judge each spike once, centered
                    p_last = float(predict_proba(model, features(np.array(buf))))
                    verdict = "SWING!" if p_last >= 0.5 else "not a swing"
                    verdict_until = time.monotonic() + 1.0
        show = verdict if time.monotonic() < verdict_until else "move the paddle..."
        color = GREEN if show == "SWING!" else RED if show == "not a swing" else (90, 90, 90)
        key = screen("LIVE TEST", color, show, f"last P(swing) = {p_last:.2f}", trace=trace,
                     progress="q = quit")
        if key == ord("q"):
            return
        next_t += 1 / RATE_HZ
        time.sleep(max(0.0, next_t - time.monotonic()))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--fresh", action="store_true", help="delete old recordings first")
    ap.add_argument("--train-only", action="store_true", help="just retrain on saved recordings")
    ap.add_argument("--test", action="store_true", help="live test the saved model")
    args = ap.parse_args()

    if args.train_only:
        clips, labels = load_data()
        if len(set(labels)) < 2:
            raise SystemExit("Need recordings of both kinds first: python3 train_swing.py")
        train(clips, labels)
        return

    dm = connect()
    try:
        if args.test:
            live_test(dm, load_model())
            return
        clips, labels = ([], []) if args.fresh else load_data()
        if clips:
            print(f"Adding to {len(clips)} earlier recordings (use --fresh to start over).")
        new_clips, new_labels = record(dm, N_EACH)
        clips += new_clips
        labels += new_labels
        if new_clips:
            save_data(clips, labels)
            print(f"Saved {len(new_clips)} new recordings -> {os.path.relpath(DATA_FILE)} ({len(clips)} total)")
        if len(set(labels)) == 2 and len(labels) >= 4:
            model = train(clips, labels)
            live_test(dm, model)
        else:
            print("Not enough recordings of both kinds to train yet.")
    finally:
        cv2.destroyAllWindows()
        dm.disconnect()


if __name__ == "__main__":
    main()
