"""
Live dashboard for the step counter, in your browser.

    python dashboard.py            # opens http://localhost:8035
    python dashboard.py --text     # no browser: one line per update in the terminal
    python dashboard.py --port 9000

Listens to the board over MQTT and serves dashboard.html, which shows the
thigh swing signal from the gyro live with the spikes and dips the board
counts as steps, the step count, distance, cadence, activity, a chart of the last 5
minutes and time spent in each activity. Its buttons send commands back to
the board (pause, resume, reset).

Calibrating ("training"): on the page, press Start, walk 5 m, press Stop and
type how many steps you took. From that it works out your step length, and
shows how close the board's count was to yours. Each walk is saved in
calibration.json under your name and sent to the board. 2-3 walks is plenty.

Ctrl+C to quit; the board keeps counting.
"""

import argparse
import json
import os
import queue
import sys
import threading
import webbrowser
from collections import deque
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import paho.mqtt.client as mqtt

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "unoq_app", "python"))
import settings as S                              # noqa: E402

PAGE = os.path.join(HERE, "dashboard.html")
CALIB_FILE = os.path.join(HERE, "calibration.json")
RUN_OVER_WALK = 1.45                             # running steps are ~45% longer
COMMANDS = {"count", "idle", "reset"}            # what the page may send
HISTORY = deque(maxlen=int(5 * 60 / S.STEP_S))   # last 5 min of results, for new tabs
clients = []                                     # one queue per open browser tab
lock = threading.Lock()
mqtt_client = None
text_mode = False
person = "me"


# --- Calibration (the 5 m walks) --------------------------------------------

def read_walks():
    try:
        with open(CALIB_FILE) as f:
            return json.load(f).get(person, [])
    except (OSError, ValueError):
        return []


def save_walks(walks):
    try:
        with open(CALIB_FILE) as f:
            everyone = json.load(f)
    except (OSError, ValueError):
        everyone = {}
    everyone[person] = walks
    with open(CALIB_FILE, "w") as f:
        json.dump(everyone, f, indent=1)


def calibration(walks):
    """What the board needs, from the saved walks (None = use its defaults):
    step length = metres / steps you counted, for walking and running."""
    if not walks:
        return None
    lengths = {}
    for kind in ("walk", "run"):
        mine = [w for w in walks if w["kind"] == kind]
        if mine:
            lengths[kind] = sum(w["meters"] for w in mine) / sum(w["steps"] for w in mine)
    if "walk" not in lengths:
        lengths["walk"] = lengths["run"] / RUN_OVER_WALK
    if "run" not in lengths:
        lengths["run"] = lengths["walk"] * RUN_OVER_WALK
    return {"step_length_m": {k: round(v, 3) for k, v in lengths.items()}}


def send_calibration():
    """Tell the board (retained, so it gets it whenever it connects) and the page."""
    walks = read_walks()
    calib = calibration(walks)
    mqtt_client.publish(S.TOPIC_CALIB, json.dumps(calib) if calib else "", qos=1, retain=True)
    broadcast({"type": "calibration", "person": person, "walks": walks, "calibration": calib})


def broadcast(event):
    data = json.dumps(event)
    with lock:
        if event["type"] == "result":
            HISTORY.append(event)
        for q in clients:
            q.put(data)


def print_line(event):
    if event["type"] == "status":
        print(f"  [board] {event['text']}")
        return
    print(f"  {event['activity']:5s}  {event['steps']:7.0f} steps  {event['distance_m']:7.1f} m"
          f"  {event['cadence_spm']:4.0f} steps/min")


# --- MQTT -------------------------------------------------------------------

def on_connect(client, userdata, flags, reason_code, properties):
    if reason_code.is_failure:
        print(f"Could not connect to {S.MQTT_BROKER}: {reason_code}")
        return
    client.subscribe([(S.TOPIC_RESULT, 0), (S.TOPIC_STATUS, 0), (S.TOPIC_LIVE, 0)])
    print(f"Connected to {S.MQTT_BROKER}, listening on {S.MQTT_BASE}/...")
    send_calibration()


def on_message(client, userdata, msg):
    text = msg.payload.decode(errors="replace")
    if msg.topic == S.TOPIC_STATUS:
        event = {"type": "status", "text": text, "old": bool(msg.retain)}
    elif msg.topic == S.TOPIC_LIVE:
        if not text_mode:
            broadcast({"type": "live", **json.loads(text)})
        return
    else:
        try:
            event = {"type": "result", **json.loads(text)}
        except ValueError:
            return
    if text_mode:
        print_line(event)
    else:
        broadcast(event)


# --- Web page ---------------------------------------------------------------

class Handler(BaseHTTPRequestHandler):
    def log_message(self, *args):
        pass

    def do_GET(self):
        if self.path == "/":
            with open(PAGE, "rb") as f:
                body = f.read()
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        elif self.path == "/events":
            self.stream()
        else:
            self.send_error(404)

    def stream(self):
        """Server-sent events: the recent history, then every new message."""
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.end_headers()
        q = queue.Queue()
        with lock:
            backlog = [json.dumps(e) for e in HISTORY]
            clients.append(q)
        walks = read_walks()
        intro = [{"type": "history", "items": backlog},
                 {"type": "calibration", "person": person, "walks": walks,
                  "calibration": calibration(walks)}]
        try:
            for event in intro:
                self.wfile.write(f"data: {json.dumps(event)}\n\n".encode())
            self.wfile.flush()
            while True:
                try:
                    data = q.get(timeout=15)
                    self.wfile.write(f"data: {data}\n\n".encode())
                except queue.Empty:
                    self.wfile.write(b": keep-alive\n\n")
                self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError):
            pass
        finally:
            with lock:
                clients.remove(q)

    def do_POST(self):
        body = self.rfile.read(int(self.headers.get("Content-Length", 0))).decode().strip()
        if self.path == "/calibrate":
            self.calibrate(body)
            return
        if self.path != "/cmd":
            self.send_error(404)
            return
        cmd = body
        if cmd not in COMMANDS:
            self.send_error(400, "unknown command")
            return
        mqtt_client.publish(S.TOPIC_CMD, cmd, qos=1)
        self.send_response(204)
        self.end_headers()

    def calibrate(self, body):
        """{"add": {"kind", "meters", "steps", "detected"}}, {"remove": index} or {"clear": true}.
        steps = what you counted, detected = what the board counted."""
        try:
            req = json.loads(body)
            walks = read_walks()
            if "add" in req:
                w = req["add"]
                walk = {"kind": w["kind"], "meters": float(w["meters"]),
                        "steps": int(w["steps"]), "detected": int(w["detected"])}
                if walk["kind"] not in ("walk", "run") or walk["steps"] < 1 or walk["meters"] <= 0:
                    raise ValueError
                walks.append(walk)
            elif "remove" in req:
                walks.pop(int(req["remove"]))
            elif req.get("clear"):
                walks = []
        except (ValueError, KeyError, IndexError, TypeError):
            self.send_error(400, "bad calibration request")
            return
        save_walks(walks)
        send_calibration()
        self.send_response(204)
        self.end_headers()


def main():
    global mqtt_client, text_mode, person
    ap = argparse.ArgumentParser(description="Live step counter dashboard")
    ap.add_argument("--port", type=int, default=8035)
    ap.add_argument("--text", action="store_true", help="print to the terminal instead")
    ap.add_argument("--no-browser", action="store_true", help="don't open the browser")
    ap.add_argument("--person", default="me", help="whose calibration to use and save")
    args = ap.parse_args()
    text_mode = args.text
    person = args.person.lower()

    mqtt_client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2)
    mqtt_client.on_connect = on_connect
    mqtt_client.on_message = on_message
    mqtt_client.connect_async(S.MQTT_BROKER, S.MQTT_PORT, keepalive=30)
    mqtt_client.loop_start()

    try:
        if text_mode:
            print("Type a command and Enter (count, idle, reset), or q to quit.")
            while (cmd := input().strip()) not in ("q", "quit"):
                if cmd:
                    mqtt_client.publish(S.TOPIC_CMD, cmd, qos=1)
            return
        server = ThreadingHTTPServer(("127.0.0.1", args.port), Handler)
        server.daemon_threads = True
        url = f"http://localhost:{args.port}"
        print(f"Dashboard at {url}  (Ctrl+C to quit; the board keeps counting)")
        if not args.no_browser:
            webbrowser.open(url)
        server.serve_forever()
    except (KeyboardInterrupt, EOFError):
        pass
    finally:
        mqtt_client.loop_stop()
        mqtt_client.disconnect()


if __name__ == "__main__":
    main()
