"""
Counts steps from the gyroscope. Runs live on the UNO Q (main.py) and on the
laptop (to test it on recordings), so both count the same way.

The sensor is on one thigh. Every step you take moves that thigh:

  - a step with the tracker leg swings it forward: a SPIKE up in the gyro
  - a step with the other leg happens while the tracker leg is planted and
    the body passes over it, so the tracker thigh swings back: a DIP down

So the steps are the spikes and dips, one step each, and they alternate.
Whichever foot you start and stop on, the count comes out right: start on
the other foot and there's a dip before the first spike; stop on it and
there's a dip after the last one.

  1. Smooth the three gyro channels (low-pass at 4 Hz) to remove jitter.
  2. Look along the direction the thigh is rotating the most (worked out
     from the last 2 s, so it doesn't matter how the sensor sits on the
     strap) and turn it so the forward swings point up. That is the
     "thigh swing" signal the dashboard draws.
  3. A spike is the signal going above SPIKE_DPS, a dip going below
     -DIP_DPS. Each has to come back near 0 before the next.
  4. Steps alternate legs, so a second spike (or dip) in a row is a wobble,
     not a step.
  5. One step alone is probably fidgeting, so a step only counts once
     another follows within MAX_GAP_S. Then both count.
"""

import math
from collections import deque

import numpy as np

SAMPLE_HZ = 100
LOWPASS_HZ = 4.0
SPIKE_DPS = 80.0         # tracker leg: the forward swing must reach this (deg/s)
DIP_DPS = 35.0           # other leg: the tracker thigh must swing back this fast
REARM_DPS = 15.0         # and the signal must come back within this of 0 before the next
MAX_GAP_S = 2.0          # a step with no other step this close is fidgeting
AXIS_N = 200             # samples used to find the swing direction (2 s)
AXIS_EVERY = 25          # ...redone every 0.25 s
AXIS_MIN_DPS = 20.0      # only while the thigh is moving this much
RUN_CADENCE = 140        # steps/min above this = running

DEFAULT_CALIBRATION = {"step_length_m": {"walk": 0.70, "run": 1.0}}


class LowPass:
    """Second-order Butterworth low-pass, one sample at a time, on 3 channels."""

    def __init__(self, cutoff_hz, fs):
        w = 2 * math.pi * cutoff_hz / fs
        alpha = math.sin(w) / (2 * math.sqrt(0.5))
        a0 = 1 + alpha
        self.b = np.array([(1 - math.cos(w)) / 2, 1 - math.cos(w), (1 - math.cos(w)) / 2]) / a0
        self.a = np.array([-2 * math.cos(w), 1 - alpha]) / a0
        self.x1 = self.x2 = self.y1 = self.y2 = None

    def __call__(self, x):
        if self.x1 is None:
            self.x1 = self.x2 = self.y1 = self.y2 = x.copy()
        y = (self.b[0] * x + self.b[1] * self.x1 + self.b[2] * self.x2
             - self.a[0] * self.y1 - self.a[1] * self.y2)
        self.x2, self.x1 = self.x1, x
        self.y2, self.y1 = self.y1, y
        return y


class SpikeCounter:
    def __init__(self, calibration=None):
        self.calibrate(calibration)
        self.lowpass = LowPass(LOWPASS_HZ, SAMPLE_HZ)
        self.recent = deque(maxlen=AXIS_N)      # smoothed gyro, for the axis
        self.axis = np.array([0.0, 1.0, 0.0])
        self.since_axis = 0
        self.inside = None                      # "tracker" while in a spike, "other" in a dip
        self.armed = {"tracker": True, "other": True}   # came back near 0 since the last one
        self.extreme = None                     # (t, value) of the spike or dip so far
        # Spikes and dips: {"t", "v", "leg", "counted"}, plus "repeat" for a
        # second one in a row on the same leg (not a step).
        self.spikes = deque(maxlen=80)
        self.reset()

    def calibrate(self, calibration):
        lengths = dict(DEFAULT_CALIBRATION["step_length_m"])
        lengths.update((calibration or {}).get("step_length_m", {}))
        self.cal = {"step_length_m": lengths}

    def reset(self):
        self.steps = 0.0
        self.distance_m = 0.0
        self.seconds = {"still": 0.0, "walk": 0.0, "run": 0.0}
        self.last_t = None

    # --- Signal ---------------------------------------------------------

    def _update_axis(self):
        g = np.array(self.recent)
        g = g - g.mean(axis=0)
        vals, vecs = np.linalg.eigh(g.T @ g / len(g))
        if math.sqrt(max(vals[-1], 0)) < AXIS_MIN_DPS:
            return                               # not moving enough to tell
        axis = vecs[:, -1]
        if axis @ self.axis < 0:                 # keep it pointing the same way
            axis = -axis
        p = g @ axis
        if -np.percentile(p, 2) > 1.3 * np.percentile(p, 98):
            axis = -axis                         # make the big swings point up
        self.axis = axis

    def add(self, t, gyro):
        """One sample (t in s, gyro in deg/s). Returns the thigh swing value."""
        t = float(t)
        g = self.lowpass(np.asarray(gyro, dtype=float))
        self.recent.append(g)
        self.since_axis += 1
        if self.since_axis >= AXIS_EVERY and len(self.recent) == AXIS_N:
            self.since_axis = 0
            self._update_axis()
        s = float(g @ self.axis)
        self._detect(t, s)
        if self.last_t is not None:
            self.seconds[self.activity(t)] += t - self.last_t
        self.last_t = t
        return s

    # --- Spikes and dips -------------------------------------------------

    def _detect(self, t, s):
        if s < REARM_DPS:
            self.armed["tracker"] = True
        if s > -REARM_DPS:
            self.armed["other"] = True
        if self.inside == "tracker":
            if s > self.extreme[1]:
                self.extreme = (t, s)
            if s < SPIKE_DPS:
                self.inside = None
                self._step("tracker", *self.extreme)
        elif self.inside == "other":
            if s < self.extreme[1]:
                self.extreme = (t, s)
            if s > -DIP_DPS:
                self.inside = None
                self._step("other", *self.extreme)
        elif self.armed["tracker"] and s >= SPIKE_DPS:
            self.inside, self.armed["tracker"], self.extreme = "tracker", False, (t, s)
        elif self.armed["other"] and s <= -DIP_DPS:
            self.inside, self.armed["other"], self.extreme = "other", False, (t, s)

    def _step(self, leg, t, v):
        step = {"t": round(t, 3), "v": round(v, 1), "leg": leg, "counted": False}
        steps = [e for e in self.spikes if not e.get("repeat")]
        prev = steps[-1] if steps else None
        near = prev is not None and t - prev["t"] <= MAX_GAP_S
        if near and prev["leg"] == leg:
            step["repeat"] = True            # same leg twice in a row: a wobble
            self.spikes.append(step)
            return
        self.spikes.append(step)
        if near:
            if not prev["counted"]:
                prev["counted"] = True
                self._add_step(prev["t"])
            step["counted"] = True
            self._add_step(t)

    def _add_step(self, t):
        self.steps += 1
        self.distance_m += self.cal["step_length_m"][self.activity(t, moving=True)]

    # --- What to show ---------------------------------------------------

    def _step_gap(self):
        """Typical time between the last few counted steps (None if < 2)."""
        times = [e["t"] for e in self.spikes if e["counted"]][-7:]
        return (float(np.median(np.diff(times))), times[-1]) if len(times) >= 2 else (None, None)

    def cadence(self, t):
        """Steps per minute from the last few counted steps (0 if stopped)."""
        gap, last = self._step_gap()
        if gap is None or t - last > MAX_GAP_S or gap <= 0:
            return 0.0
        return 60.0 / gap

    def activity(self, t, moving=False):
        """still / walk / run, from how fast the counted steps come."""
        gap, last = self._step_gap()
        if not moving and (gap is None or t - last > MAX_GAP_S):
            return "still"
        if not gap:
            return "walk"
        return "run" if 60.0 / gap >= RUN_CADENCE else "walk"

    def summary(self, t):
        return {
            "steps": round(self.steps, 1),
            "distance_m": round(self.distance_m, 2),
            "cadence_spm": round(self.cadence(t), 1),
            "activity": self.activity(t),
            "seconds": {k: round(v, 1) for k, v in self.seconds.items()},
            "step_length_m": self.cal["step_length_m"],
        }
