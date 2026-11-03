"""
Laptop-side data code: load recordings, label every step automatically, and
cut recordings into training windows. train.py and evaluate.py use this.

Where the step labels come from
-------------------------------
Nobody counted the steps in the activity recordings, so a simple detector
finds them: the sensor is on one thigh, and that thigh swings forward once
per stride (= 2 steps, one per leg). The swing shows up as a big, slow
rotation on the gyro, so each upward crossing of that rotation is one stride.
Between crossings the step count goes up smoothly, which gives a (fractional)
number of steps for any window. The net learns from these labels.

"Counted" recordings are the honest test: you walk a number of steps you
count yourself and type it in (guided_train.py). Their counts are stored in
data/counts.json: {"<file name>": {"steps": 40, "meters": 30.0 or null}}.
"""

import glob
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "unoq_app", "python"))

import settings as S                              # noqa: E402

DATA_DIR = os.path.join(HERE, "data")
COUNTS = "counts.json"
PEOPLE = os.path.join(HERE, "people.json")

MIN_SWING_DPS = 25      # thigh rotation (deg/s) that counts as a stride, not fidgeting
MAX_STRIDE_S = 2.5      # a longer gap between strides means you stopped
MOTION_SCALE = (0.5, 1.4)   # training: walk/run motion made this much gentler/harder


# --- Recordings -------------------------------------------------------------

def read_counts(folder=DATA_DIR):
    try:
        with open(os.path.join(folder, COUNTS)) as f:
            return json.load(f)
    except (OSError, ValueError):
        return {}


def save_count(folder, name, steps, meters=None):
    counts = read_counts(folder)
    counts[name] = {"steps": steps, "meters": meters}
    with open(os.path.join(folder, COUNTS), "w") as f:
        json.dump(counts, f, indent=1)


def load_sessions(folder=DATA_DIR, person=None):
    """Every <label>_<person>_<time>.csv in folder (only that person's, if
    given), with its step labels."""
    counts = read_counts(folder)
    sessions = []
    for path in sorted(glob.glob(os.path.join(folder, "*.csv"))):
        name = os.path.splitext(os.path.basename(path))[0]
        parts = name.split("_")
        if len(parts) < 3 or parts[0] not in S.LABELS:
            continue                       # e.g. old bike recordings
        if person and parts[1] != person:
            continue
        data = np.loadtxt(path, delimiter=",", skiprows=1, ndmin=2)
        s = {"name": name, "label": parts[0], "person": parts[1],
             "imu": data[:, 1:7], "counted": counts.get(name)}
        s["cum"] = step_labels(s["imu"], s["label"])
        sessions.append(s)
    return sessions


# --- Automatic step labels --------------------------------------------------

def _lowpass(x, cutoff_hz):
    X = np.fft.rfft(x)
    X[np.fft.rfftfreq(len(x), 1 / S.SAMPLE_HZ) > cutoff_hz] = 0
    return np.fft.irfft(X, len(x))


def thigh_swing(imu):
    """The thigh's forward/back rotation (deg/s): the gyro direction that
    moves the most, smoothed. Works however the sensor is turned on the strap."""
    g = imu[:, 3:] - imu[:, 3:].mean(axis=0)
    direction = np.linalg.svd(g, full_matrices=False)[2][0]
    return _lowpass(g @ direction, 3.0)


def stride_starts(imu):
    """Sample index where each stride starts (upward crossing of the swing)."""
    swing = thigh_swing(imu)
    thr = MIN_SWING_DPS / 2
    starts, armed = [], False
    for i in range(1, len(swing)):
        if swing[i] < -thr:
            armed = True
        if armed and swing[i - 1] < thr <= swing[i]:
            if not starts or i - starts[-1] > 0.4 * S.SAMPLE_HZ:
                starts.append(i)
            armed = False
    return np.array(starts, dtype=int)


def step_labels(imu, label):
    """cum[i] = steps taken before sample i (length N + 1), so the steps in
    samples a..b-1 are cum[b] - cum[a]."""
    n = len(imu)
    cum = np.zeros(n + 1)
    starts = stride_starts(imu) if label != "still" else []
    if len(starts) < 2:
        return cum
    gaps = np.diff(starts)
    typical = np.median(gaps)
    knots_t, knots_c = [], []
    total = 0.0
    for k, a in enumerate(starts):
        if k == 0 or gaps[k - 1] > MAX_STRIDE_S * S.SAMPLE_HZ:
            # A new bout of walking: count the step before the first swing.
            lead = int(min(a, typical / 2))
            if knots_t:                    # finish the last bout with 1 step
                end = knots_t[-1] + int(min(typical / 2, a - knots_t[-1]))
                total += 1
                knots_t.append(end)
                knots_c.append(total)
            knots_t.append(a - lead)
            knots_c.append(total)
            total += 1
        else:
            # A stride is 2 steps; a gap of 2 typical strides means one was missed.
            total += 2 * max(1, round(gaps[k - 1] / typical))
        knots_t.append(a)
        knots_c.append(total)
    end = min(n, knots_t[-1] + int(typical / 2))
    knots_t.append(end)
    knots_c.append(total + 1)
    cum[:] = np.interp(np.arange(n + 1), knots_t, knots_c)
    return cum


# --- Windows ----------------------------------------------------------------

def usable_range(s):
    """Activity recordings lose their first and last 3 s (getting started);
    counted recordings are used whole (you stand still at both ends)."""
    if s["counted"]:
        return 0, len(s["imu"])
    trim = int(S.TRIM_S * S.SAMPLE_HZ)
    return trim, len(s["imu"]) - trim


def window_activity(s, steps):
    """Activity label for a window: the recording's label, except that in a
    counted recording a window with no steps is "still". -1 = unclear, skip."""
    if not s["counted"] or s["label"] == "still":
        return S.LABELS.index(s["label"])
    if steps >= 1.0:
        return S.LABELS.index(s["label"])
    return S.LABELS.index("still") if steps <= 0.2 else -1


def windows(sessions, every=S.STEP_N, stretch=None, rng=None):
    """Cut sessions into windows. Returns X (n, N, 6), steps (n,), activity
    (n,), session index (n,).

    stretch=(lo, hi) is for training: each window is cut from a stretch of
    lo..hi times 2 s and squeezed back into 2 s, which looks like walking
    faster or slower, and its motion is scaled by MOTION_SCALE, which looks
    like walking more gently or harder. So the net sees more kinds of walking
    than were recorded."""
    X, steps, act, sess = [], [], [], []
    for k, s in enumerate(sessions):
        lo_i, hi_i = usable_range(s)
        start = lo_i
        while True:
            scale = rng.uniform(*stretch) if stretch else 1.0
            length = int(round(S.WINDOW_N * scale))
            if start + length > hi_i:
                break
            seg = s["imu"][start:start + length]
            if length != S.WINDOW_N:
                src = np.linspace(0, length - 1, S.WINDOW_N)
                seg = np.stack([np.interp(src, np.arange(length), seg[:, c])
                                for c in range(6)], axis=1)
            if stretch and s["label"] != "still":
                # Gentler or harder movement: scale the motion, keep gravity.
                still_part = seg.mean(axis=0)
                seg = still_part + (seg - still_part) * rng.uniform(*MOTION_SCALE)
            n_steps = s["cum"][start + length] - s["cum"][start]
            X.append(seg)
            steps.append(n_steps)
            act.append(window_activity(s, n_steps))
            sess.append(k)
            start += every if not stretch else int(rng.integers(every // 2, every * 3 // 2))
    return (np.array(X, dtype=np.float32), np.array(steps, dtype=np.float32),
            np.array(act), np.array(sess))


# --- People -----------------------------------------------------------------

def read_people():
    try:
        with open(PEOPLE) as f:
            return json.load(f)
    except (OSError, ValueError):
        return {}


def save_height(person, height_cm):
    people = read_people()
    people.setdefault(person, {})["height_cm"] = height_cm
    with open(PEOPLE, "w") as f:
        json.dump(people, f, indent=1)


def step_lengths(sessions, person):
    """Metres per step for walk and run. From recordings over a measured
    distance if there are any, otherwise from height."""
    height = read_people().get(person or "", {}).get("height_cm", S.DEFAULT_HEIGHT_CM)
    out, how = {}, {}
    for act in ("walk", "run"):
        pairs = [(s["counted"]["meters"], s["counted"]["steps"]) for s in sessions
                 if s["label"] == act and s["counted"] and s["counted"].get("meters")]
        if pairs:
            out[act] = sum(m for m, _ in pairs) / sum(n for _, n in pairs)
            how[act] = f"measured over {len(pairs)} walk(s) of known distance"
        else:
            out[act] = S.STEP_LENGTH_PER_HEIGHT[act] * height / 100
            how[act] = f"{S.STEP_LENGTH_PER_HEIGHT[act]} x height ({height:.0f} cm)"
    out["still"] = 0.0
    return out, how
