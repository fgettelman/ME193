"""
The phone page and its JSON API. The board runs it with a live sensor
(main.py); the laptop runs it without one (studio.py) to review and train on
walks copied off the board.

  GET  /                      the page (web/index.html)
  GET  /api/state             mode, live navigator state, walks, routes, models
  POST /api/person            {name, step_length_m}
  POST /api/start             {kind: demo|nav|stairs|still, start, dest, note, cls}
  POST /api/stop              {save: true/false}
  POST /api/tap               {kind: mark|correct|arrived, label}
  GET  /api/walk?id=          one walk for Review (track, events, labels)
  POST /api/labels            {id, events, use}
  POST /api/delete            {id} or {stairs_id}
  POST /api/train             {what: policy|stairs}
  GET  /api/bundle?id=        everything about a walk (laptop sync)
  POST /api/bundle            upload a walk (laptop sync: your partner's walks)
  GET  /api/model?name=       policy|stairs model JSON
  POST /api/model?name=       upload a model trained on the laptop
"""

import json
import os
import threading
import traceback
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

import settings as S
import store as st

HERE = os.path.dirname(os.path.abspath(__file__))
PAGE = os.path.join(HERE, "web", "index.html")


class App:
    """What the handler talks to. `live` is the board's controller (None on the laptop)."""

    def __init__(self, store, live=None, train_search="full"):
        self.store = store
        self.live = live
        self.train_search = train_search
        self.lock = threading.RLock()
        self.training = None              # "policy"/"stairs" while training runs
        self.last_train = {}

    def state(self):
        with self.lock:
            walks = self.store.walks()
            pol = self.store.policy()
            stairs = self.store.stairs_model()
            out = {
                "live_available": self.live is not None,
                "people": self.store.people(),
                "walks": walks,
                "walks_used": sum(1 for w in walks if w["use"] and w["reviewed"]),
                "walks_total": len(walks),
                "max_walks": S.MAX_WALKS,
                "routes": self.store.routes(),
                "policy": {"ready": pol.ready, "routes": pol.routes(),
                           "trained_on": len(pol.d.get("trained_on", [])) if pol.d else 0,
                           "lowo": pol.d.get("lowo") if pol.d else None},
                "stairs": {"ready": stairs.ready,
                           "counts": stairs.d.get("counts") if stairs.ready else None,
                           "recordings": self.store.stairs_recordings()},
                "training": self.training,
                "last_train": self.last_train,
                "actions": S.ACTIONS,
                "lookahead_m": S.LOOKAHEAD_M,
            }
            if self.live is not None:
                out["live"] = self.live.state()
            return out

    def train(self, what):
        with self.lock:
            if self.training:
                return {"ok": False, "error": f"already training {self.training}"}
            self.training = what

        def work():
            try:
                if what == "stairs":
                    res = self.store.train_stairs()
                else:
                    res = self.store.train_policy(self.train_search)
            except Exception as e:
                traceback.print_exc()
                res = {"ok": False, "error": str(e)}
            with self.lock:
                self.training = None
                self.last_train = {"what": what, **res}
                if self.live is not None and res.get("ok"):
                    self.live.reload_models()
            print(f"trained {what}: {json.dumps(res)[:300]}")

        threading.Thread(target=work, daemon=True).start()
        return {"ok": True, "started": what}


def make_handler(app):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def _send(self, code, body, ctype="application/json"):
            if not isinstance(body, (bytes, bytearray)):
                body = json.dumps(body, default=st.jsonable).encode()
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def _body(self):
            n = int(self.headers.get("Content-Length") or 0)
            if not n:
                return {}
            try:
                return json.loads(self.rfile.read(n))
            except ValueError:
                return {}

        def do_GET(self):
            u = urlparse(self.path)
            q = {k: v[0] for k, v in parse_qs(u.query).items()}
            try:
                if u.path in ("/", "/index.html"):
                    with open(PAGE, "rb") as f:
                        return self._send(200, f.read(), "text/html; charset=utf-8")
                if u.path == "/api/state":
                    return self._send(200, app.state())
                if u.path == "/api/walk":
                    return self._send(200, app.store.walk_for_review(q["id"], q.get("suggest") == "1"))
                if u.path == "/api/bundle":
                    return self._send(200, bundle(app.store, q["id"]))
                if u.path == "/api/stairs_bundle":
                    return self._send(200, stairs_bundle(app.store, q["id"]))
                if u.path == "/api/model":
                    m = app.store.policy() if q.get("name") == "policy" else app.store.stairs_model()
                    return self._send(200, m.d or {})
            except Exception as e:
                traceback.print_exc()
                return self._send(500, {"ok": False, "error": str(e)})
            self._send(404, {"ok": False, "error": "not found"})

        def do_POST(self):
            u = urlparse(self.path)
            q = {k: v[0] for k, v in parse_qs(u.query).items()}
            b = self._body()
            live = app.live
            try:
                if u.path == "/api/person":
                    app.store.set_person(b["name"].strip().lower(), float(b["step_length_m"]))
                    if live:
                        live.set_person(b["name"].strip().lower())
                    return self._send(200, {"ok": True})
                if u.path == "/api/labels":
                    app.store.save_labels(b["id"], {"events": b["events"], "use": b.get("use", True)})
                    return self._send(200, {"ok": True})
                if u.path == "/api/delete":
                    if b.get("stairs_id"):
                        app.store.delete_stairs(b["stairs_id"])
                    else:
                        app.store.delete_walk(b["id"])
                    return self._send(200, {"ok": True})
                if u.path == "/api/train":
                    return self._send(200, app.train(b.get("what", "policy")))
                if u.path == "/api/bundle":
                    return self._send(200, unbundle(app.store, b))
                if u.path == "/api/stairs_bundle":
                    return self._send(200, unbundle_stairs(app.store, b))
                if u.path == "/api/model":
                    name = "policy" if q.get("name") == "policy" else "stairs"
                    st._write(os.path.join(app.store.root, "models", name + ".json"), b)
                    if live:
                        live.reload_models()
                    return self._send(200, {"ok": True})
                if live is None:
                    return self._send(400, {"ok": False, "error": "no sensor here (laptop studio)"})
                if u.path == "/api/start":
                    return self._send(200, live.start(b))
                if u.path == "/api/stop":
                    return self._send(200, live.stop(bool(b.get("save", True))))
                if u.path == "/api/tap":
                    return self._send(200, live.tap(b.get("kind", "mark"), b.get("label")))
            except Exception as e:
                traceback.print_exc()
                return self._send(500, {"ok": False, "error": str(e)})
            self._send(404, {"ok": False, "error": "not found"})

    return Handler


# --- Moving walks between the board and laptops (sync.py) ---------------------

def bundle(store, wid):
    d = store._wdir(wid)
    with open(os.path.join(d, "raw.csv")) as f:
        raw = f.read()
    return {"id": wid, "meta": store.meta(wid), "track": store.track(wid),
            "labels": store.labels(wid), "raw": raw}


def unbundle(store, b):
    wid = os.path.basename(b["id"])
    d = store._wdir(wid)
    os.makedirs(d, exist_ok=True)
    with open(os.path.join(d, "raw.csv"), "w") as f:
        f.write(b["raw"])
    st._write(os.path.join(d, "track.json"), b["track"])
    st._write(os.path.join(d, "meta.json"), b["meta"])
    if b.get("labels"):
        st._write(os.path.join(d, "labels.json"), b["labels"])
    return {"ok": True, "id": wid}


def stairs_bundle(store, sid):
    d = os.path.join(store.root, "stairs", os.path.basename(sid))
    with open(os.path.join(d, "raw.csv")) as f:
        raw = f.read()
    return {"id": sid, "meta": st._read(os.path.join(d, "meta.json")), "raw": raw}


def unbundle_stairs(store, b):
    sid = os.path.basename(b["id"])
    d = os.path.join(store.root, "stairs", sid)
    os.makedirs(d, exist_ok=True)
    with open(os.path.join(d, "raw.csv"), "w") as f:
        f.write(b["raw"])
    st._write(os.path.join(d, "meta.json"), b["meta"])
    return {"ok": True, "id": sid}


def serve(app, port=S.WEB_PORT):
    server = ThreadingHTTPServer(("0.0.0.0", port), make_handler(app))
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server
