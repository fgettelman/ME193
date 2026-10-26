"""
Turn a window of IMU samples into features. Used on the UNO Q (live) and on
the laptop (training), so both always compute exactly the same numbers.

A window is a numpy array of shape (N, 6): columns ax, ay, az (g) and
gx, gy, gz (deg/s), sampled at settings.SAMPLE_HZ.
"""

import numpy as np

from settings import COLUMNS, SAMPLE_HZ

# --- Model 1: peak acceleration in one direction ----------------------------

# The six directions the peak model can use: (column, sign).
PEAK_DIRECTIONS = [(c, s) for c in ("ax", "ay", "az") for s in (1, -1)]


def direction_name(axis, sign):
    return ("+" if sign > 0 else "-") + axis


def peak(window, axis, sign):
    """Largest acceleration along one direction in the window, in g."""
    return float(np.max(sign * window[:, COLUMNS.index(axis)]))


# --- Model 2: everything ----------------------------------------------------

def _dominant(x, lo=0.3, hi=5.0):
    """Strongest rhythm in x between lo and hi Hz: (frequency, share of power)."""
    x = (x - x.mean()) * np.hanning(len(x))
    power = np.abs(np.fft.rfft(x)) ** 2
    freqs = np.fft.rfftfreq(len(x), 1 / SAMPLE_HZ)
    band = (freqs >= lo) & (freqs <= hi)
    total = power[(freqs >= lo) & (freqs <= 15)].sum()
    if total <= 1e-12 or not band.any():
        return 0.0, 0.0
    i = np.argmax(np.where(band, power, -1))
    return float(freqs[i]), float(power[i] / total)


def _corr(a, b):
    if a.std() < 1e-9 or b.std() < 1e-9:
        return 0.0
    return float(np.corrcoef(a, b)[0, 1])


def _feature_names():
    names = []
    for c in COLUMNS:
        names += [f"{c}_mean", f"{c}_std", f"{c}_min", f"{c}_max", f"{c}_range"]
    for m in ("acc_mag", "gyro_mag"):
        names += [f"{m}_mean", f"{m}_std", f"{m}_max", f"{m}_p90"]
    names += ["acc_rhythm_hz", "acc_rhythm_strength",
              "gyro_rhythm_hz", "gyro_rhythm_strength",
              "pitch_deg", "roll_deg", "jerk", "acc_gyro_corr"]
    return names


FULL_NAMES = _feature_names()


def full_features(window):
    """About 46 numbers describing the window (see FULL_NAMES for the order)."""
    w = np.asarray(window, dtype=float)
    out = []
    for i in range(6):
        x = w[:, i]
        out += [x.mean(), x.std(), x.min(), x.max(), x.max() - x.min()]

    acc, gyro = w[:, :3], w[:, 3:]
    acc_mag = np.linalg.norm(acc, axis=1)
    gyro_mag = np.linalg.norm(gyro, axis=1)
    for m in (acc_mag, gyro_mag):
        out += [m.mean(), m.std(), m.max(), np.percentile(m, 90)]

    # Rhythm: steps (acceleration) and leg swing / pedal turns (the gyro axis
    # that moves the most, whichever way the sensor is mounted).
    main_gyro = gyro[:, np.argmax(gyro.std(axis=0))]
    out += [*_dominant(acc_mag), *_dominant(main_gyro)]

    # Orientation from the average gravity direction: pitch = angle of +X
    # (along the thigh) above/below horizontal. Standing ~ +/-90, lying ~ 0.
    g = acc.mean(axis=0)
    out += [np.degrees(np.arctan2(g[0], np.hypot(g[1], g[2]))),
            np.degrees(np.arctan2(g[1], g[2]))]

    # Smoothness: foot strikes make sharp jumps, pedalling doesn't.
    out.append(float(np.mean(np.abs(np.diff(acc_mag))) * SAMPLE_HZ))
    # Coupling between how hard it's shaking and how fast the leg rotates.
    out.append(_corr(acc_mag, np.abs(main_gyro)))
    return np.array(out, dtype=float)
