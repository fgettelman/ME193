"""
Campus navigation core. Runs live on the UNO Q (main.py) and on the laptop
(train.py, replaying recorded walks), so a walk turns into exactly the same
states whether it is live or replayed.

Built on top of the step counter, not on raw IMU windows:

  raw IMU (100 Hz, thigh)
    -> SpikeCounter (spikes.py, the step counter): steps
    -> Heading: which way is "up" (gravity, from the accelerometer, kept
       up to date between steps with the gyro), and the turn rate about up
       (gyro . up). Integrated, that's the heading relative to where you
       started. The gyro's bias is re-learned whenever you stand still.
    -> TurnDetector: a turn = the heading changed > 50 deg and settled
    -> stride features -> StairsModel (Gaussian naive Bayes):
       flat / up / down for each stride; 2 strides in a row = a flight
    -> NavTracker: the navigator's state, 5 times a second:
         distance walked, heading, dead-reckoned x/y, turns made so far,
         distance since the last turn, flights of stairs so far, on stairs?

  The navigator (Policy) is behavior cloning: k-nearest-neighbours from
  those states to the instruction the demonstrator's walk implies at that
  moment (straight / left / right / stairs up / stairs down / arrived).
"""

import json
import math
from collections import deque

import numpy as np

import settings as S
import spikes

DT = 1.0 / S.SAMPLE_HZ
R2D = 180.0 / math.pi


def _norm(v):
    n = math.sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2])
    return (v[0] / n, v[1] / n, v[2] / n) if n > 1e-9 else (0.0, 0.0, 1.0), n


# --- Heading ------------------------------------------------------------------

class Heading:
    """Heading (deg, left turns positive) from a thigh-worn IMU, whatever way
    the sensor sits on the strap."""

    def __init__(self):
        self.up = None                    # unit "up" in the sensor's frame
        self.ref_up = None                # "up" while standing still (thigh vertical)
        self.bias = [0.0, 0.0, 0.0]       # gyro bias, deg/s
        self.heading = 0.0
        self.still_n = 0
        self.still_s = 0.0                # total seconds of stillness seen (bias quality)
        self.yaw_rate = 0.0
        self.angle = 0.0                  # thigh angle from standing, deg

    @property
    def calibrated(self):
        """True once the gyro bias has been measured standing still (~1.5 s).
        Without it the heading drifts ~1 deg/s (tested on the step-counter walks)."""
        return self.still_s >= 1.5

    def zero(self):
        self.heading = 0.0

    def add(self, acc, gyro, dt=DT):
        a_hat, a_n = _norm(acc)
        if self.up is None:
            self.up = a_hat
            self.ref_up = a_hat

        # Standing still: learn the gyro bias and what "thigh vertical" looks like.
        w0 = [gyro[i] - self.bias[i] for i in range(3)]
        still = (abs(a_n - 1.0) < S.STILL_ACCEL_G
                 and math.sqrt(w0[0] ** 2 + w0[1] ** 2 + w0[2] ** 2) < S.STILL_GYRO_DPS)
        self.still_n = self.still_n + 1 if still else 0
        if self.still_n > S.SAMPLE_HZ // 2:
            self.still_s += dt
            # Fast average for the first stillness, then a slow update.
            g = max(S.BIAS_GAIN, dt / max(self.still_s, dt)) if self.still_s < 3 else S.BIAS_GAIN
            for i in range(3):
                self.bias[i] += g * (gyro[i] - self.bias[i])
            r = self.ref_up
            self.ref_up = _norm([r[i] + 0.05 * (a_hat[i] - r[i]) for i in range(3)])[0]

        w = [gyro[i] - self.bias[i] for i in range(3)]
        # "Up" is fixed in the world; seen from the rotating sensor it turns by
        # -w dt (exact rotation, Rodrigues), ...
        wn = math.sqrt(w[0] ** 2 + w[1] ** 2 + w[2] ** 2)
        u = self.up
        th = wn / R2D * dt
        if th > 1e-9:
            k = (-w[0] / wn, -w[1] / wn, -w[2] / wn)
            c, s = math.cos(th), math.sin(th)
            kd = k[0] * u[0] + k[1] * u[1] + k[2] * u[2]
            kx = (k[1] * u[2] - k[2] * u[1], k[2] * u[0] - k[0] * u[2], k[0] * u[1] - k[1] * u[0])
            u = tuple(u[i] * c + kx[i] * s + k[i] * kd * (1 - c) for i in range(3))
        # ...and pulled toward the accelerometer only when the thigh is nearly
        # still (stance): mid-swing the accelerometer reads swing, not gravity.
        if abs(a_n - 1.0) < 0.1 and wn < 60.0:
            k = S.GRAVITY_GAIN
            u = tuple(u[i] + k * (a_hat[i] - u[i]) for i in range(3))
        self.up = _norm(u)[0]

        self.yaw_rate = w[0] * self.up[0] + w[1] * self.up[1] + w[2] * self.up[2]
        self.heading += self.yaw_rate * dt
        c = sum(self.up[i] * self.ref_up[i] for i in range(3))
        self.angle = math.acos(max(-1.0, min(1.0, c))) * R2D
        return self.heading


# --- Turns ------------------------------------------------------------------

class TurnDetector:
    def __init__(self):
        self.seg = 0.0                     # heading of the current straight
        self.rates = deque(maxlen=int(1.5 * S.SAMPLE_HZ))
        self.rate_sum = 0.0
        self.turns = 0
        self.last = None                   # (t, "left"/"right", degrees)

    def zero(self):
        self.seg = 0.0
        self.turns = 0
        self.last = None

    def add(self, t, heading, yaw_rate):
        """Returns "left"/"right" on the sample a turn is recognised, else None."""
        if len(self.rates) == self.rates.maxlen:
            self.rate_sum -= self.rates[0]
        self.rates.append(yaw_rate)
        self.rate_sum += yaw_rate
        settled = abs(self.rate_sum / len(self.rates)) < S.TURN_SETTLE_DPS
        diff = heading - self.seg
        if settled and abs(diff) >= S.TURN_DEG:
            self.turns += 1
            self.seg = heading
            self.last = (t, "left" if diff > 0 else "right", round(diff))
            return self.last[1]
        if settled and abs(diff) < S.DRIFT_ABSORB_DEG:
            self.seg += 0.002 * diff       # follow gentle bends and drift (~5 s)
        return None


# --- Stairs -----------------------------------------------------------------

# Stride features that say "stairs or not" without saying "fast or slow":
#   angle_range       how far the thigh swings, deg (up stairs: ~63, else ~35)
#   bounce_per_swing  accel jolt relative to swing speed (stairs jolt more)
#   back_swing_frac   share of the stride the thigh swings back (less going down)
# Raw swing speed, stride time and bounce were dropped: they mostly measure
# walking pace, so a brisk flat walk looked like stairs. Picked by
# leave-one-recording-out accuracy on our recordings: 89% vs 69% before.
# (The peak thigh angle was dropped earlier: it depends on a standing reference.)
STRIDE_FEATURES = ["angle_range", "bounce_per_swing", "back_swing_frac"]


def stride_features(samples):
    """samples: list of (t, thigh angle, thigh swing, |accel|) over one stride."""
    a = np.array(samples)
    ang, sw, acc = a[:, 1], a[:, 2], a[:, 3]
    smax, smin = float(sw.max()), float(sw.min())
    return [float(ang.max() - ang.min()),
            float(acc.std() / max(smax, 1.0) * 100),
            float(np.mean(sw < -0.3 * abs(smin)))]


class StairsModel:
    """Gaussian naive Bayes over stride features: flat / up / down."""

    def __init__(self, d=None):
        self.d = d

    @property
    def ready(self):
        # A model trained on the old feature list is ignored (retrain stairs).
        return self.d is not None and self.d.get("features") == STRIDE_FEATURES

    @property
    def id(self):
        return self.d.get("id") if self.d else None

    @staticmethod
    def fit(X, y, model_id):
        X, y = np.asarray(X, float), np.asarray(y)
        classes = [c for c in S.STAIR_CLASSES if (y == c).any()]
        mean = [X[y == c].mean(0).tolist() for c in classes]
        var = [(X[y == c].var(0) + 1e-3 + 0.05 * X.var(0)).tolist() for c in classes]
        return StairsModel({"type": "gnb", "id": model_id, "classes": classes,
                            "features": STRIDE_FEATURES, "mean": mean, "var": var,
                            "counts": [int((y == c).sum()) for c in classes]})

    def probs(self, f):
        if not self.ready:
            return {"flat": 1.0}
        f = np.asarray(f, float)
        m, v = np.array(self.d["mean"]), np.array(self.d["var"])
        ll = -0.5 * (((f - m) ** 2) / v + np.log(v)).sum(1)
        p = np.exp(ll - ll.max())
        p /= p.sum()
        return dict(zip(self.d["classes"], p.tolist()))


# --- The navigator's state ----------------------------------------------------

class NavTracker:
    """Feed it raw samples; it keeps the navigator's state and logs a track
    (TRACK_HZ rows/s) and the events it noticed (turns, flights of stairs)."""

    def __init__(self, step_length_m=0.70, stairs_model=None, keep_strides=False):
        self.counter = spikes.SpikeCounter({"step_length_m": {"walk": step_length_m,
                                                              "run": step_length_m * 1.4}})
        self.step_len = step_length_m
        self.heading = Heading()
        self.turn = TurnDetector()
        self.stairs = stairs_model or StairsModel()
        self.buf = deque(maxlen=3 * S.SAMPLE_HZ)    # (t, angle, swing, |a|)
        self.last_spike_t = None
        self.last_tracker_t = None
        self.keep_strides = keep_strides
        self.strides = []                           # (t, features) if keep_strides
        self.zero(None)

    def zero(self, t):
        """Start of a walk: distance, heading and counts back to 0 here."""
        self.t0 = t
        self.heading.zero()
        self.turn.zero()
        self.steps_seen = self.steps0 = self.counter.steps
        self.d = self.x = self.y = 0.0
        self.since_turn_d = 0.0
        self.flights = 0
        self.on = "flat"                       # flat / up / down
        self.stride_cls = deque(maxlen=max(S.FLIGHT_STRIDES, S.FLIGHT_END_STRIDES))
        self.flight_start = None
        self.events = []                       # auto-detected: turns, flights
        self.track = []
        self.next_row = t
        self.last_t = t

    def add(self, t, acc, gyro):
        dt = DT if self.last_t is None else min(max(t - self.last_t, 0.0), 0.1) or DT
        self.last_t = t
        h = self.heading.add(acc, gyro, dt)
        swing = self.counter.add(t, gyro)
        self.buf.append((t, self.heading.angle, swing, _norm(acc)[1]))

        turned = self.turn.add(t, h, self.heading.yaw_rate)
        if turned:
            self.since_turn_d = 0.0
            self.events.append({"t": round(t, 2), "type": turned, "auto": True,
                                "deg": self.turn.last[2]})

        sp = self.counter.spikes[-1] if self.counter.spikes else None
        if sp is not None and sp["t"] != self.last_spike_t:
            self.last_spike_t = sp["t"]
            if sp["leg"] == "tracker" and not sp.get("repeat"):
                self._stride(sp["t"])

        new = self.counter.steps - self.steps_seen
        if new > 0:
            self.steps_seen = self.counter.steps
            L = S.STAIR_STEP_M if self.on != "flat" else self.step_len
            hr = math.radians(h)
            self.d += new * L
            self.since_turn_d += new * L
            self.x += new * L * math.cos(hr)
            self.y += new * L * math.sin(hr)

        if self.next_row is None:
            self.next_row = t
        if t >= self.next_row:
            self.next_row = t + 1.0 / S.TRACK_HZ
            self.track.append(self.row(t))

    def _stride(self, t):
        prev, self.last_tracker_t = self.last_tracker_t, t
        if prev is None or not (0.5 <= t - prev <= 2.5):
            return
        win = [s for s in self.buf if prev <= s[0] <= t]
        if len(win) < 20:
            return
        f = stride_features(win)
        if self.keep_strides:
            self.strides.append((t, f))
        p = self.stairs.probs(f)
        cls = max(p, key=p.get)
        if cls != "flat" and p[cls] < S.STAIR_MIN_P:     # not sure: call it flat
            cls = "flat"
        self.stride_cls.append((prev, cls))
        recent = [c for _, c in self.stride_cls]
        start_n, end_n = S.FLIGHT_STRIDES, S.FLIGHT_END_STRIDES
        if (self.on == "flat" and cls != "flat" and len(recent) >= start_n
                and all(c == cls for c in recent[-start_n:])):
            self.on, self.flight_start = cls, self.stride_cls[-start_n][0]
        elif (self.on != "flat" and len(recent) >= end_n
              and all(c == "flat" for c in recent[-end_n:])):
            self.flights += 1
            self.events.append({"t": round(self.flight_start, 2), "t_end": round(t, 2),
                                "type": "stairs_" + self.on, "auto": True})
            self.on = "flat"

    def row(self, t):
        return {"t": round(t, 2), "steps": int(self.counter.steps - self.steps0), "d": round(self.d, 2),
                "h": round(self.heading.heading, 1), "x": round(self.x, 2), "y": round(self.y, 2),
                "turns": self.turn.turns, "dst": round(self.since_turn_d, 2),
                "flights": self.flights, "on": self.on,
                "moving": bool(self.counter.cadence(t) > 0)}


def replay(raw, step_length_m, stairs_model=None, keep_strides=False, t0=None):
    """raw: array of rows t, ax, ay, az, gx, gy, gz (as saved). The walk is
    zeroed at t0 (when it started, after standing still; saved in the walk's
    meta), or at the first row. The rows before t0 let it learn the gyro bias
    the same way it did live. Returns the tracker."""
    tr = NavTracker(step_length_m, stairs_model, keep_strides)
    raw = np.asarray(raw, float)
    zeroed = False
    for r in raw:
        if not zeroed and (t0 is None or r[0] >= t0 - 1e-6):
            tr.zero(r[0])
            zeroed = True
        tr.add(r[0], (r[1], r[2], r[3]), (r[4], r[5], r[6]))
    if not zeroed:
        tr.zero(None)
    return tr


# --- Labels: from the reviewed events to an instruction for every moment -------

def label_track(track, events, lookahead=S.LOOKAHEAD_M):
    """For each track row, the instruction a guide should be giving then.

    events: [{"t", "type", optional "t_end"}], type in ACTIONS (not "straight").
    A turn/stairs instruction shows from `lookahead` metres before the event
    until it's done; "arrived" from the arrival on. Returns a list of labels
    (None for rows after the walk really ended)."""
    ts = np.array([r["t"] for r in track])
    ds = np.array([r["d"] for r in track])

    def d_at(t):
        return float(np.interp(t, ts, ds))

    evs = sorted(events, key=lambda e: e["t"])
    arrive = next((e["t"] for e in evs if e["type"] == "arrived"), None)
    spans = []
    for e in evs:
        if e["type"] in ("arrived", "straight"):
            continue
        end = e.get("t_end", e["t"])
        spans.append((d_at(e["t"]) - lookahead, e["t"], end, e["type"]))
    out = []
    for r in track:
        t, d = r["t"], r["d"]
        if arrive is not None and t >= arrive:
            out.append("arrived" if t <= arrive + 4.0 else None)
            continue
        lab = "straight"
        for d0, t_ev, t_end, typ in spans:
            if t <= t_end and (d >= d0 or t >= t_ev - 1.0):
                lab = typ
                break
        out.append(lab)
    return out


def stair_spans(events):
    """(t_start, t_end, "up"/"down"/"flat") for reviewed stairs and flat spans."""
    out = []
    for e in events:
        if e["type"] in ("stairs_up", "stairs_down"):
            out.append((e["t"], e.get("t_end", e["t"] + 6.0), e["type"][7:]))
        elif e["type"] == "straight" and e.get("t_end") is not None:
            out.append((e["t"], e["t_end"], "flat"))
    return out


def apply_reviewed_stairs(rows, events):
    """The track with "on" and "flights" taken from the reviewed stairs, not
    the classifier's guesses, so the navigator trains on what really happened."""
    spans = [s for s in stair_spans(events) if s[2] != "flat"]
    out = []
    for r in rows:
        t = r["t"]
        on = next((k for t0, t1, k in spans if t0 <= t <= t1), "flat")
        out.append({**r, "on": on, "flights": sum(1 for _, t1, _ in spans if t1 < t)})
    return out


def label_strides(strides, events, margin_s=1.0):
    """Stairs-classifier training data from a reviewed walk: each stride is
    up/down inside a reviewed stairs span, flat elsewhere; strides within
    margin_s of a span's edge (getting on/off) are skipped."""
    spans = stair_spans(events)
    stairs = [s for s in spans if s[2] != "flat"]
    X, y = [], []
    for t, f in strides:
        if any(abs(t - t0) < margin_s or abs(t - t1) < margin_s for t0, t1, _ in stairs):
            continue
        X.append(f)
        y.append(next((k for t0, t1, k in spans if t0 <= t <= t1), "flat"))
    return X, y


# --- The navigator: k-nearest neighbours over the state ------------------------

FEATURES = ["turns", "dst", "d", "cos_h", "sin_h", "x", "y", "flights", "on_up", "on_down"]
SCALE = np.array([1.0, 10.0, 30.0, 0.5, 0.5, 20.0, 20.0, 1.0, 1.0, 1.0])
GROUPS = {"progress": [0, 7, 8, 9], "distance": [1, 2], "heading": [3, 4], "position": [5, 6]}
DEFAULT_WEIGHTS = {"progress": 2.0, "distance": 1.0, "heading": 1.0, "position": 1.0}


def state_vector(r):
    h = math.radians(r["h"])
    return [r["turns"], r["dst"], r["d"], math.cos(h), math.sin(h), r["x"], r["y"],
            r["flights"], float(r["on"] == "up"), float(r["on"] == "down")]


def weight_vector(w):
    v = np.ones(len(FEATURES))
    for g, idx in GROUPS.items():
        v[idx] = w.get(g, 1.0)
    return v / SCALE


def training_rows(track, labels, min_step_m=0.4):
    """Keep a row every ~0.4 m of walking (and whenever the label changes),
    so standing around doesn't flood the data."""
    X, y, last_d, last_lab = [], [], None, None
    for r, lab in zip(track, labels):
        if lab is None:
            continue
        if last_d is None or r["d"] - last_d >= min_step_m or lab != last_lab:
            X.append(state_vector(r))
            y.append(lab)
            last_d, last_lab = r["d"], lab
    return X, y


class Policy:
    def __init__(self, d=None):
        self.d = d
        if d:
            self.X = np.array(d["X"], float)
            self.y = np.array(d["y"])
            self.route = np.array(d["route"])
            self.wv = weight_vector(d["weights"])

    @property
    def ready(self):
        return self.d is not None and len(self.d.get("X", [])) > 0

    def routes(self):
        return self.d["routes"] if self.d else []

    @staticmethod
    def build(walk_data, weights=None, k=5):
        """walk_data: list of (route_key, X, y). Returns a Policy."""
        routes = sorted({w[0] for w in walk_data})
        X, y, ri = [], [], []
        for key, Xw, yw in walk_data:
            X += Xw
            y += yw
            ri += [routes.index(key)] * len(yw)
        return Policy({"type": "knn", "k": k, "weights": weights or DEFAULT_WEIGHTS,
                       "routes": routes, "X": X, "y": y, "route": ri})

    def probs(self, state, route_key):
        """state: state_vector(row). Returns {action: probability}."""
        if not self.ready or route_key not in self.d["routes"]:
            return {}
        sel = self.route == self.d["routes"].index(route_key)
        X, y = self.X[sel], self.y[sel]
        if not len(y):
            return {}
        dist = np.sqrt((((X - np.asarray(state)) * self.wv) ** 2).sum(1))
        k = min(self.d["k"], len(y))
        nn = np.argpartition(dist, k - 1)[:k]
        votes = {}
        for i in nn:
            votes[y[i]] = votes.get(y[i], 0.0) + 1.0 / (dist[i] + 0.05)
        tot = sum(votes.values())
        return {a: votes.get(a, 0.0) / tot for a in S.ACTIONS}

    def to_json(self):
        return json.dumps(self.d)


def predict_many(train, test_X, weights, k=5):
    """Plain kNN for cross-validation: train = (X, y), returns labels."""
    X, y = np.asarray(train[0], float), np.asarray(train[1])
    T = np.asarray(test_X, float)
    wv = weight_vector(weights)
    D = np.sqrt(((((T[:, None, :] - X[None, :, :]) * wv) ** 2).sum(2)))
    k = min(k, len(y))
    out = []
    for row in D:
        nn = np.argpartition(row, k - 1)[:k]
        votes = {}
        for i in nn:
            votes[y[i]] = votes.get(y[i], 0.0) + 1.0 / (row[i] + 0.05)
        out.append(max(votes, key=votes.get))
    return out


# --- Live guidance: smooth the policy's answers --------------------------------

class Guide:
    """Turns the policy's per-row probabilities into a steady instruction:
    average of the last 3, and a new instruction must win twice in a row."""

    def __init__(self, policy, route_key, route_len_m=None):
        self.policy, self.route = policy, route_key
        self.route_len = route_len_m
        self.hist = deque(maxlen=3)
        self.shown = "straight"
        self.pending, self.pending_n = None, 0
        self.probs = {}

    def update(self, row):
        p = self.policy.probs(state_vector(row), self.route)
        if not p:
            return self.shown
        self.hist.append(p)
        avg = {a: sum(h.get(a, 0) for h in self.hist) / len(self.hist) for a in S.ACTIONS}
        self.probs = avg
        best = max(avg, key=avg.get)
        if best == "arrived" and avg[best] < 0.6:
            best = "straight"
        if best == self.shown:
            self.pending, self.pending_n = None, 0
        elif best == self.pending:
            self.pending_n += 1
            if self.pending_n >= 2:
                self.shown, self.pending, self.pending_n = best, None, 0
        else:
            self.pending, self.pending_n = best, 1
        return self.shown
