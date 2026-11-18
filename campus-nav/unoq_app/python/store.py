"""
Walks, labels and training. Used by the board (main.py, through the web page)
and by the laptop (train.py, studio.py) on a copy of the same folder layout:

  <root>/walks/<id>/meta.json     who, route, demo/dagger, taps, step length
  <root>/walks/<id>/raw.csv       t, ax, ay, az, gx, gy, gz at 100 Hz (g, deg/s)
  <root>/walks/<id>/track.json    the navigator's state, 5 rows/s, and auto events
  <root>/walks/<id>/labels.json   the reviewed events (written by Review)
  <root>/stairs/<id>/meta.json    a stairs recording: class flat/up/down
  <root>/stairs/<id>/raw.csv
  <root>/models/policy.json       the navigator
  <root>/models/stairs.json       the stairs classifier
  <root>/people.json              step length per person
"""

import json
import os
import shutil
import time

import numpy as np

import navcore
import settings as S

RAW_HEADER = "t,ax,ay,az,gx,gy,gz"


def route_key(start, dest):
    return f"{start.strip()} -> {dest.strip()}"


def jsonable(o):
    """json.dump(default=...): numpy numbers/bools/arrays to plain Python."""
    if isinstance(o, np.generic):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    raise TypeError(f"{type(o).__name__} is not JSON serializable")


def _read(path, default=None):
    try:
        with open(path) as f:
            return json.load(f)
    except (OSError, ValueError):
        return default


def _write(path, obj):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(obj, f, default=jsonable)
    os.replace(tmp, path)


def load_raw(path):
    try:
        a = np.loadtxt(path, delimiter=",", skiprows=1, ndmin=2)
    except (OSError, ValueError):
        return np.zeros((0, 7))
    return a


def write_raw(path, rows):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        f.write(RAW_HEADER + "\n")
        for r in rows:
            f.write(f"{r[0]:.3f}," + ",".join(f"{x:.4f}" for x in r[1:]) + "\n")


def merge_turns(events, within_s=4.0):
    """A slow turn can show up as two auto turns in a row; join them."""
    out = []
    for e in sorted(events, key=lambda e: e["t"]):
        if (out and e["type"] in ("left", "right") and out[-1]["type"] == e["type"]
                and e["t"] - out[-1]["t"] <= within_s):
            out[-1] = {**out[-1], "t": e["t"], "deg": out[-1].get("deg", 0) + e.get("deg", 0)}
        else:
            out.append(dict(e))
    return out


def suggested_events(meta, track):
    """What Review starts from: the auto-detected turns and flights, plus the
    taps made on the walk (Arrived, and DAgger corrections)."""
    evs = merge_turns(track.get("events", []))
    taps = meta.get("taps", [])
    for tap in taps:
        lab = tap.get("label")
        if lab == "arrived":
            evs.append({"t": tap["t"], "type": "arrived", "from": "tap"})
        elif tap.get("kind") == "correct" and lab in ("left", "right", "stairs_up", "stairs_down"):
            # A correction says this instruction should be showing now. If no
            # matching auto event follows soon, put one a few seconds ahead.
            soon = [e for e in evs if e["type"] == lab and 0 <= e["t"] - tap["t"] <= 15]
            if not soon:
                ev = {"t": tap["t"] + 3.0, "type": lab, "from": "correction"}
                if lab.startswith("stairs"):
                    ev["t_end"] = ev["t"] + 6.0
                evs.append(ev)
    if not any(e["type"] == "arrived" for e in evs) and track.get("rows"):
        evs.append({"t": track["rows"][-1]["t"], "type": "arrived", "from": "end"})
    return sorted(evs, key=lambda e: e["t"])


class Store:
    def __init__(self, root):
        self.root = root
        for d in ("walks", "stairs", "models"):
            os.makedirs(os.path.join(root, d), exist_ok=True)

    # --- People -----------------------------------------------------------

    def people(self):
        return _read(os.path.join(self.root, "people.json"), {})

    def set_person(self, name, step_length_m):
        p = self.people()
        p[name] = {"step_length_m": round(float(step_length_m), 3)}
        _write(os.path.join(self.root, "people.json"), p)

    def step_length(self, name):
        return self.people().get(name, {}).get("step_length_m", 0.70)

    # --- Walks ------------------------------------------------------------

    def _wdir(self, wid):
        return os.path.join(self.root, "walks", os.path.basename(wid))

    def new_id(self, kind, person):
        base = f"{time.strftime('%Y%m%d-%H%M%S')}_{kind}_{person}"
        wid, n = base, 1
        while os.path.exists(self._wdir(wid)):
            n += 1
            wid = f"{base}-{n}"
        return wid

    def save_walk(self, wid, meta, raw_rows, tracker):
        d = self._wdir(wid)
        write_raw(os.path.join(d, "raw.csv"), raw_rows)
        _write(os.path.join(d, "track.json"), {"rows": tracker.track, "events": tracker.events,
                                               "stairs_model": tracker.stairs.id})
        meta = {**meta, "id": wid, "saved": time.time(),
                "length_m": round(tracker.d, 1), "seconds": round(
                    tracker.track[-1]["t"] - tracker.track[0]["t"], 1) if tracker.track else 0}
        _write(os.path.join(d, "meta.json"), meta)
        return meta

    def walk_ids(self):
        d = os.path.join(self.root, "walks")
        return sorted(w for w in os.listdir(d) if os.path.exists(os.path.join(d, w, "meta.json")))

    def meta(self, wid):
        return _read(os.path.join(self._wdir(wid), "meta.json"), {})

    def track(self, wid):
        return _read(os.path.join(self._wdir(wid), "track.json"), {"rows": [], "events": []})

    def labels(self, wid):
        return _read(os.path.join(self._wdir(wid), "labels.json"))

    def save_labels(self, wid, labels):
        labels = {**labels, "reviewed": time.time()}
        _write(os.path.join(self._wdir(wid), "labels.json"), labels)

    def delete_walk(self, wid):
        shutil.rmtree(self._wdir(wid), ignore_errors=True)

    def walk_summary(self, wid):
        m, lab = self.meta(wid), self.labels(wid)
        return {"id": wid, "person": m.get("person"), "kind": m.get("kind"),
                "route": m.get("route"), "length_m": m.get("length_m"),
                "seconds": m.get("seconds"), "reviewed": lab is not None,
                "use": bool(lab and lab.get("use", True)),
                "reviewed_at": (lab or {}).get("reviewed", 0),
                "corrections": sum(1 for t in m.get("taps", []) if t.get("kind") == "correct")}

    def walks(self):
        return [self.walk_summary(w) for w in self.walk_ids()]

    def walk_for_review(self, wid, suggest=False):
        m, tr = self.meta(wid), self.track(wid)
        lab = self.labels(wid)
        events = lab["events"] if lab and not suggest else suggested_events(m, tr)
        rows = tr.get("rows", [])
        labels = navcore.label_track(rows, events) if rows else []
        return {"meta": m, "rows": rows, "auto": merge_turns(tr.get("events", [])),
                "events": events, "use": lab.get("use", True) if lab else True,
                "reviewed": lab is not None, "labels": labels}

    def routes(self):
        """Every route with walks, and how long it is (median of reviewed walks)."""
        out = {}
        for w in self.walks():
            r = w.get("route") or {}
            if not r.get("start"):
                continue
            key = route_key(r["start"], r.get("dest", S.DEFAULT_DEST))
            e = out.setdefault(key, {"key": key, "start": r["start"],
                                     "dest": r.get("dest", S.DEFAULT_DEST),
                                     "note": r.get("note", ""), "lengths": [], "walks": 0})
            e["walks"] += 1
            if r.get("note"):
                e["note"] = r["note"]
            if w["use"] and w["length_m"]:
                e["lengths"].append(w["length_m"])
        for e in out.values():
            e["length_m"] = float(np.median(e.pop("lengths"))) if e.get("lengths") else None
        return sorted(out.values(), key=lambda e: e["key"])

    # --- Stairs recordings -----------------------------------------------

    def save_stairs(self, cls, person, raw_rows):
        sid = f"{time.strftime('%Y%m%d-%H%M%S')}_{cls}_{person}"
        d = os.path.join(self.root, "stairs", sid)
        write_raw(os.path.join(d, "raw.csv"), raw_rows)
        _write(os.path.join(d, "meta.json"), {"id": sid, "cls": cls, "person": person,
                                              "seconds": round(raw_rows[-1][0] - raw_rows[0][0], 1)
                                              if raw_rows else 0})
        return sid

    def stairs_recordings(self):
        d = os.path.join(self.root, "stairs")
        out = []
        for s in sorted(os.listdir(d)):
            m = _read(os.path.join(d, s, "meta.json"))
            if m:
                out.append(m)
        return out

    def delete_stairs(self, sid):
        shutil.rmtree(os.path.join(self.root, "stairs", os.path.basename(sid)), ignore_errors=True)

    # --- Models -----------------------------------------------------------

    def stairs_model(self):
        return navcore.StairsModel(_read(os.path.join(self.root, "models", "stairs.json")))

    def policy(self):
        return navcore.Policy(_read(os.path.join(self.root, "models", "policy.json")))

    def train_stairs(self):
        """Fit the stairs classifier on the stairs recordings (strides from each,
        minus the first and last stride, which are getting on / off), plus the
        strides of every reviewed walk: flat except where Review says stairs.
        So fixing the stairs in Review also teaches the classifier."""
        X, y, group = [], [], []          # group: which recording/walk a stride came from
        too_short = []
        for m in self.stairs_recordings():
            raw = load_raw(os.path.join(self.root, "stairs", m["id"], "raw.csv"))
            step = self.step_length(m.get("person"))
            tr = navcore.replay(raw, step, keep_strides=True)
            strides = tr.strides[1:-1]
            if len(strides) < 3:
                too_short.append({"id": m["id"], "cls": m["cls"], "seconds": m.get("seconds")})
            X += [f for _, f in strides]
            y += [m["cls"]] * len(strides)
            group += [m["id"]] * len(strides)
        from_walks = 0
        for w in self.walks():
            if not w["reviewed"]:
                continue
            m = self.meta(w["id"])
            raw = load_raw(os.path.join(self._wdir(w["id"]), "raw.csv"))
            tr = navcore.replay(raw, m.get("step_length_m", 0.70), keep_strides=True, t0=m.get("t0"))
            Xw, yw = navcore.label_strides(tr.strides, self.labels(w["id"])["events"])
            X += Xw
            y += yw
            group += [w["id"]] * len(yw)
            from_walks += len(yw)
        counts = {c: y.count(c) for c in S.STAIR_CLASSES}
        if counts["flat"] < 4 or counts["up"] + counts["down"] < 4:
            return {"ok": False, "too_short": too_short,
                    "error": f"need more strides: {counts}. Record ~30 s of "
                             "each (flat walking, stairs up, stairs down)."}
        # Honest accuracy: hold out one whole recording (or walk) at a time and
        # test on it. Strides from the same recording are nearly identical, so
        # holding out single strides made it look far better than it is.
        X_a, y_a, g_a = np.array(X), np.array(y), np.array(group)
        per = {c: [0, 0] for c in S.STAIR_CLASSES}
        for gid in np.unique(g_a):
            test = g_a == gid
            if len(set(y_a[~test])) < 2:
                continue
            m = navcore.StairsModel.fit(X_a[~test], y_a[~test], "cv")
            for f, t in zip(X_a[test], y_a[test]):
                p = m.probs(f)
                per[t][1] += 1
                per[t][0] += max(p, key=p.get) == t
        tested = sum(n for _, n in per.values())
        recall = {c: round(r / n, 3) for c, (r, n) in per.items() if n}
        model = navcore.StairsModel.fit(X, y, f"stairs-{int(time.time())}")
        _write(os.path.join(self.root, "models", "stairs.json"), model.d)
        self.retrack_all(model)
        return {"ok": True, "strides": counts, "from_walks": from_walks,
                "accuracy": round(sum(r for r, _ in per.values()) / tested, 3) if tested else None,
                "recall": recall, "groups": int(len(np.unique(g_a))), "too_short": too_short}

    def retrack_all(self, stairs=None):
        """Re-run every walk with the current stairs model, so Review shows
        what the new classifier thinks."""
        stairs = stairs or self.stairs_model()
        for wid in self.walk_ids():
            if self.track(wid).get("stairs_model") != stairs.id:
                self.retrack(wid, stairs)

    def retrack(self, wid, stairs=None):
        """Replay a walk's raw data (e.g. after the stairs model changed)."""
        m = self.meta(wid)
        raw = load_raw(os.path.join(self._wdir(wid), "raw.csv"))
        stairs = stairs or self.stairs_model()
        tr = navcore.replay(raw, m.get("step_length_m", 0.70), stairs, t0=m.get("t0"))
        _write(os.path.join(self._wdir(wid), "track.json"),
               {"rows": tr.track, "events": tr.events, "stairs_model": stairs.id})

    def training_set(self, retrack=True):
        """(route_key, X, y, walk id) for every reviewed walk marked 'use',
        at most MAX_WALKS (oldest first)."""
        stairs = self.stairs_model()
        out, skipped = [], []
        for w in self.walks():
            if not (w["reviewed"] and w["use"]):
                continue
            if len(out) >= S.MAX_WALKS:
                skipped.append(w["id"])
                continue
            if retrack and self.track(w["id"]).get("stairs_model") != stairs.id:
                self.retrack(w["id"], stairs)
            r = w["route"] or {}
            events = self.labels(w["id"])["events"]
            rows = navcore.apply_reviewed_stairs(self.track(w["id"])["rows"], events)
            labels = navcore.label_track(rows, events)
            X, y = navcore.training_rows(rows, labels)
            out.append((route_key(r.get("start", "?"), r.get("dest", S.DEFAULT_DEST)), X, y, w["id"]))
        return out, skipped

    def train_policy(self, search="full"):
        data, skipped = self.training_set()
        if not data:
            return {"ok": False, "error": "no reviewed walks yet: walk a route, then Review it"}
        result = cross_validate(data, search)
        policy = navcore.Policy.build([(k, X, y) for k, X, y, _ in data], result["weights"])
        policy.d.update({"trained_on": [w for *_, w in data], "trained": time.time(),
                         "lowo": result})
        _write(os.path.join(self.root, "models", "policy.json"), policy.d)
        return {"ok": True, "walks": len(data), "skipped_over_limit": skipped,
                "samples": len(policy.d["y"]), **result}


WEIGHT_GRID_SMALL = [
    {"progress": p, "distance": 1.0, "heading": h, "position": x}
    for p in (1.0, 3.0) for h in (0.5, 1.5) for x in (0.0, 1.0)
]
WEIGHT_GRID = [
    {"progress": p, "distance": d, "heading": h, "position": x}
    for p in (1.0, 2.0, 4.0) for d in (0.5, 1.0, 2.0) for h in (0.5, 1.0, 2.0) for x in (0.0, 0.5, 1.0)
]


def cross_validate(data, search="full"):
    """Leave-one-walk-out: train on the other walks of the same route, test on
    this one. Score = balanced accuracy (mean recall over the instructions),
    since 'straight' is most of every walk. Picks the feature weights that
    score best."""
    def score(weights):
        per = {a: [0, 0] for a in S.ACTIONS}
        conf = {}
        for i, (key, X, y, _) in enumerate(data):
            others = [(Xo, yo) for j, (k2, Xo, yo, _) in enumerate(data) if j != i and k2 == key]
            if not others or not y:
                continue
            tx = [r for Xo, _ in others for r in Xo]
            ty = [l for _, yo in others for l in yo]
            pred = navcore.predict_many((tx, ty), X, weights)
            for t, p in zip(y, pred):
                per[t][1] += 1
                per[t][0] += t == p
                conf[f"{t}|{p}"] = conf.get(f"{t}|{p}", 0) + 1
        recalls = {a: round(c / n, 3) for a, (c, n) in per.items() if n}
        bal = float(np.mean(list(recalls.values()))) if recalls else None
        return bal, recalls, conf

    grid = {"full": WEIGHT_GRID, "small": WEIGHT_GRID_SMALL}.get(search, [navcore.DEFAULT_WEIGHTS])
    best = None
    for w in grid:
        bal, rec, conf = score(w)
        if bal is None:
            break
        if best is None or bal > best[0] + 1e-9:
            best = (bal, rec, conf, w)
    if best is None:
        return {"weights": navcore.DEFAULT_WEIGHTS, "balanced_accuracy": None,
                "note": "need 2+ walks of the same route to cross-validate"}
    return {"weights": best[3], "balanced_accuracy": round(best[0], 3),
            "recall": best[1], "confusion": best[2]}
