"""
Campus navigator, UNO Q Linux side.

Reads the thigh IMU from the sketch (100 Hz), runs the step counter + heading
+ turns + stairs (navcore.NavTracker) all the time, and serves a phone page
on port 8050 (server.py, web/index.html) with these modes:

  idle     the live state is shown (heading, steps, turns): handy to test
  demo     you walk a route; the walk is recorded (raw IMU + the navigator's
           states) so it can be reviewed and become training data
  nav      the trained navigator tells you what to do; tap a correction
           whenever it's wrong (DAgger) and the walk is recorded too
  stairs   record flat walking / stairs up / stairs down for the stairs
           classifier (these don't count toward the 10 walks)

Every demo/nav walk starts by standing still ~1.5 s, so the gyro bias is
measured fresh and the heading starts at 0 facing the way you face then.
Walks are saved under python/data/ (sync.py on the laptop copies them).
"""

import os
import sys
import threading
import time
from collections import deque

import numpy as np
from arduino.app_utils import App, Bridge

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import navcore                                    # noqa: E402
import server                                     # noqa: E402
import settings as S                              # noqa: E402
import store as st                                # noqa: E402

DATA = os.path.join(HERE, "data")
SCALE = np.array([1 / S.ACCEL_COUNTS_PER_G] * 3 + [1 / S.GYRO_COUNTS_PER_DPS] * 3)
PREROLL_S = 4.0                 # raw data kept from before a walk's start (for replay)
HOLD_STILL_S = 2.5             # standing still before a walk (gyro bias); < PREROLL_S


class Live:
    def __init__(self, store):
        self.store = store
        self.lock = threading.RLock()
        self.person = next(iter(store.people()), "fiona")
        self.mode = "idle"
        self.phase = None               # "hold_still" then "go" in demo/nav
        self.route = None
        self.stairs_cls = None
        self.rec = []
        self.pre = deque(maxlen=int(PREROLL_S * S.SAMPLE_HZ))
        self.taps = []
        self.shown_log = []
        self.guide = None
        self.t = 0.0
        self.t0 = None
        self.rate = 0.0
        self.message = ""
        self.sensor_ok = True
        self.last_saved = None
        self.stairs = store.stairs_model()
        self.policy = store.policy()
        self.tracker = navcore.NavTracker(store.step_length(self.person), self.stairs)
        self.tracker.zero(None)

    # --- called from the web page (server threads) -------------------------

    def reload_models(self):
        with self.lock:
            self.stairs = self.store.stairs_model()
            self.policy = self.store.policy()
            self.tracker.stairs = self.stairs

    def set_person(self, name):
        with self.lock:
            self.person = name
            self.tracker.step_len = self.store.step_length(name)

    def start(self, b):
        with self.lock:
            if self.mode != "idle":
                return {"ok": False, "error": f"already in {self.mode}: stop first"}
            kind = b.get("kind")
            if kind == "zero":
                self.tracker.zero(self.t)
                return {"ok": True}
            if kind == "stairs":
                if b.get("cls") not in S.STAIR_CLASSES:
                    return {"ok": False, "error": "pick flat / up / down"}
                self.mode, self.stairs_cls, self.rec = "stairs", b["cls"], list(self.pre)
                self.message = f"recording {b['cls']}: walk normally, then Stop"
                return {"ok": True}
            if kind not in ("demo", "nav"):
                return {"ok": False, "error": "unknown kind"}
            start = (b.get("start") or "").strip()
            dest = (b.get("dest") or S.DEFAULT_DEST).strip()
            if not start:
                return {"ok": False, "error": "name the starting point"}
            key = st.route_key(start, dest)
            if kind == "nav":
                if not self.policy.ready or key not in self.policy.routes():
                    return {"ok": False, "error": f"no trained navigator for {key}: train first"}
                rl = next((r["length_m"] for r in self.store.routes() if r["key"] == key), None)
                self.guide = navcore.Guide(self.policy, key, rl)
            self.mode, self.phase = kind, "hold_still"
            self.route = {"start": start, "dest": dest, "note": (b.get("note") or "").strip()}
            self.taps, self.shown_log, self.rec = [], [], []
            self.tracker.step_len = self.store.step_length(self.person)
            self.tracker.heading.still_n = 0
            self.message = "Stand still, facing the start direction..."
            return {"ok": True}

    def tap(self, kind, label):
        with self.lock:
            if self.mode not in ("demo", "nav") or self.phase != "go":
                return {"ok": False, "error": "not walking"}
            tap = {"t": round(self.t, 2), "kind": kind, "label": label,
                   "d": round(self.tracker.d, 1)}
            if self.guide:
                tap["shown"] = self.guide.shown
            self.taps.append(tap)
            if label == "arrived":
                return self.stop(True)
            if kind == "correct" and self.guide:
                # Show the correction straight away; the guide resumes after.
                self.guide.shown = label
            return {"ok": True, "taps": len(self.taps)}

    def stop(self, save=True):
        with self.lock:
            mode, out = self.mode, {"ok": True}
            if mode == "stairs" and save and len(self.rec) > S.SAMPLE_HZ * 3:
                out["saved"] = self.store.save_stairs(self.stairs_cls, self.person, self.rec)
            elif mode in ("demo", "nav") and save and self.phase == "go" and self.tracker.track:
                wid = self.store.new_id("dagger" if mode == "nav" else "demo", self.person)
                meta = {"person": self.person, "kind": "dagger" if mode == "nav" else "demo",
                        "route": self.route, "taps": self.taps, "t0": self.t0,
                        "step_length_m": self.tracker.step_len, "shown": self.shown_log,
                        "stairs_model": self.stairs.id}
                self.store.save_walk(wid, meta, self.rec, self.tracker)
                out["saved"] = self.last_saved = wid
            self.mode, self.phase, self.guide, self.rec = "idle", None, None, []
            self.message = f"saved {out['saved']}: review it" if out.get("saved") else "stopped"
            return out

    def state(self):
        with self.lock:
            tr = self.tracker
            row = tr.row(self.t) if self.t else None
            s = {"mode": self.mode, "phase": self.phase, "person": self.person,
                 "route": self.route, "message": self.message, "rate": round(self.rate),
                 "sensor_ok": self.sensor_ok, "calibrated": tr.heading.calibrated, "still": tr.heading.still_n > 20,
                 "row": row, "taps": self.taps[-5:], "n_taps": len(self.taps),
                 "auto_events": tr.events[-5:], "last_saved": self.last_saved,
                 "bias": [round(b, 2) for b in tr.heading.bias],
                 "rec_s": round(len(self.rec) / S.SAMPLE_HZ, 1),
                 "stairs_cls": self.stairs_cls}
            if self.guide:
                g = self.guide
                s["instruction"] = g.shown
                s["probs"] = {a: round(p, 2) for a, p in g.probs.items()}
                s["progress"] = (min(1.0, tr.d / g.route_len) if g.route_len else None)
                # For the phone to keep guiding through Wi-Fi drops: what's
                # coming (metres ahead -> instruction), and how fast you walk.
                s["plan"] = g.plan
                s["route_len"] = g.route_len
            if self.mode in ("demo", "nav") and self.phase == "go":
                s["d"] = round(tr.d, 2)
                s["speed"] = round(tr.counter.cadence(self.t) / 60.0 * tr.step_len, 2)
                s["board_t"] = round(self.t, 2)
            return s

    # --- called from the main loop ------------------------------------------

    def add_samples(self, samples):
        with self.lock:
            for t, v in samples:
                self.t = t
                row = (t, *v)
                tr = self.tracker
                n_rows = len(tr.track)
                tr.add(t, v[:3], v[3:])
                if self.mode in ("demo", "nav"):
                    if self.phase == "hold_still":
                        self.pre.append(row)
                        if tr.heading.still_n >= HOLD_STILL_S * S.SAMPLE_HZ and tr.heading.calibrated:
                            # A fresh tracker that has only seen the pre-roll, so the
                            # live states are exactly what replaying raw.csv gives.
                            tr = self.tracker = navcore.NavTracker(tr.step_len, self.stairs)
                            for r in self.pre:
                                tr.add(r[0], r[1:4], r[4:7])
                            tr.zero(t)
                            self.t0 = t
                            self.rec = list(self.pre)
                            self.phase = "go"
                            self.message = ("Go! Walk the route" if self.mode == "demo"
                                            else "Go! Follow the instructions")
                            n_rows = len(tr.track)
                    else:
                        self.rec.append(row)
                elif self.mode == "stairs":
                    self.rec.append(row)
                else:
                    self.pre.append(row)
                if self.guide and self.phase == "go" and len(tr.track) > n_rows:
                    before = self.guide.shown
                    now = self.guide.update(tr.track[-1])
                    if now != before or not self.shown_log:
                        self.shown_log.append({"t": round(t, 2), "shown": now})
            if self.mode == "idle" and len(self.tracker.track) > 3000:
                del self.tracker.track[:1500]


store = st.Store(DATA)
live = Live(store)
# Full weight search: ~3x slower than a laptop, still only seconds with 10 walks.
app = server.App(store, live, train_search="full")
try:
    server.serve(app)
    print(f"Phone page on port {S.WEB_PORT}")
except OSError as e:
    print(f"Web server didn't start: {e}")

counter = {"n": 0, "zeros": 0, "t": time.monotonic()}


def read_samples():
    text = Bridge.call("read_batch")
    out = []
    for chunk in text.split(";"):
        parts = chunk.split(",")
        if len(parts) != 7:
            continue
        raw = np.array([int(p) for p in parts[1:]], dtype=float) * SCALE
        out.append((int(parts[0]) / 1000.0, raw.tolist()))
    return out


def loop():
    try:
        samples = read_samples()
    except Exception as e:
        print(f"bridge error: {e}")
        # The microcontroller isn't answering (seen after re-flashing the
        # sketch): say so on the page, so it doesn't just look frozen.
        counter["bridge_fails"] = counter.get("bridge_fails", 0) + 1
        if counter["bridge_fails"] >= 2:
            live.sensor_ok, live.rate = False, 0
            live.message = ("The board's microcontroller isn't answering. Unplug the board's "
                            "power for 10 s and plug it back in.")
        time.sleep(1.0)
        return
    counter["bridge_fails"] = 0
    # An accelerometer reading exactly 0 g is impossible (gravity): it means the
    # MPU-6050 is asleep after a power blip. Don't feed those to the heading.
    good = [s for s in samples if any(s[1][:3])]
    counter["zeros"] += len(samples) - len(good)
    if good:
        live.add_samples(good)
    counter["n"] += len(good)
    el = time.monotonic() - counter["t"]
    if el >= 5:
        live.rate = counter["n"] / el
        zeros = counter["zeros"]
        live.sensor_ok = counter["n"] > 0
        if counter["n"] == 0 or zeros:
            try:
                info = Bridge.call("sensor_info")
            except Exception:
                info = ""
            live.message = (("Sensor is sending zeros: check the VCC/GND wires. " if zeros else
                             "No IMU data: ") + info)
        elif live.message.startswith(("Sensor is sending zeros", "No IMU data")):
            live.message = "Sensor OK again"
        print(f"mode={live.mode} {live.rate:.0f} samples/s, {zeros} zero samples")
        counter["n"], counter["zeros"], counter["t"] = 0, 0, time.monotonic()
    time.sleep(0.05)


App.run(user_loop=loop)
