"""
Gait identifier, UNO Q Linux side.

Pulls IMU samples from the sketch, and depending on the mode:
  record <label>  streams the raw samples over MQTT (collect.py on the laptop
                  saves them to a CSV)
  classify        every 0.5 s, runs both models on the last 2 s and publishes
                  {"peak": ..., "full": ...} (smoothed over the last 3 answers)
  idle            just keeps reading

Commands arrive on ME193/Rogers/Fiona/gait/cmd:
  "record walk", "stop", "classify", "idle", "reload" (re-read the models)
"""

import json
import os
import sys
import time
from collections import Counter, deque

import numpy as np
import paho.mqtt.client as mqtt
from arduino.app_utils import App, Bridge

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import models                                     # noqa: E402
import settings as S                              # noqa: E402

MODEL_DIR = os.path.join(HERE, "models")
SCALE = np.array([1 / S.ACCEL_COUNTS_PER_G] * 3 + [1 / S.GYRO_COUNTS_PER_DPS] * 3)


class State:
    def __init__(self):
        self.models = models.load_all(MODEL_DIR)
        self.mode = "classify" if self.models else "idle"
        self.label = None
        self.recent = deque(maxlen=S.WINDOW_N)     # (t, 6 values) in g / deg/s
        self.new_since_classify = 0
        self.votes = {name: deque(maxlen=S.VOTE_WINDOWS) for name in ("peak", "full")}
        self.count = 0
        self.rate_start = time.monotonic()
        self.pending_cmd = None


state = State()


# --- MQTT -------------------------------------------------------------------

def on_connect(client, userdata, flags, reason_code, properties):
    if not reason_code.is_failure:
        client.subscribe(S.TOPIC_CMD, qos=1)
        status(f"connected; mode={state.mode}; models={sorted(state.models)}")


def on_message(client, userdata, msg):
    # Handled in the main loop so the loop never sees a half-changed mode.
    state.pending_cmd = msg.payload.decode(errors="replace").strip()


client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2)
client.on_connect = on_connect
client.on_message = on_message
client.connect_async(S.MQTT_BROKER, S.MQTT_PORT, keepalive=30)
client.loop_start()


def status(text):
    print(text)
    client.publish(S.TOPIC_STATUS, text, retain=True)


def handle(cmd):
    parts = cmd.lower().split()
    if not parts:
        return
    if parts[0] == "record" and len(parts) == 2 and parts[1] in S.LABELS:
        state.mode, state.label = "record", parts[1]
    elif parts[0] == "stop":
        state.mode, state.label = ("classify" if state.models else "idle"), None
    elif parts[0] in ("classify", "idle"):
        state.mode, state.label = parts[0], None
    elif parts[0] == "reload":
        state.models = models.load_all(MODEL_DIR)
    else:
        status(f"unknown command: {cmd!r}")
        return
    if state.mode == "classify" and not state.models:
        state.mode = "idle"
        status("no models in python/models/ yet; train on the laptop first")
    for v in state.votes.values():
        v.clear()
    status(f"mode={state.mode}" + (f" label={state.label}" if state.label else "")
           + f"; models={sorted(state.models)}")


# --- Reading the sketch -----------------------------------------------------

def read_samples():
    """New samples since the last call: list of (t_seconds, np.array of 6)."""
    text = Bridge.call("read_batch")
    out = []
    for chunk in text.split(";"):
        parts = chunk.split(",")
        if len(parts) != 7:
            continue
        raw = np.array([int(p) for p in parts[1:]], dtype=float)
        out.append((int(parts[0]) / 1000.0, raw * SCALE))
    return out


def report_sensor():
    """Say where the sketch found the MPU-6050 (or that it didn't)."""
    try:
        status("sensor: " + Bridge.call("sensor_info"))
    except Exception as e:
        status(f"sensor: couldn't ask the sketch ({e})")


def classify():
    window = np.array([v for _, v in state.recent])
    result = {"t": time.time()}
    for name, model in state.models.items():
        state.votes[name].append(model.predict(window))
        result[name] = Counter(state.votes[name]).most_common(1)[0][0]
        result[name + "_raw"] = state.votes[name][-1]
    if "peak" in state.models:
        result["peak_g"] = round(state.models["peak"].feature(window), 3)
    client.publish(S.TOPIC_RESULT, json.dumps(result))
    shown = result.get("full", result.get("peak"))
    try:
        Bridge.call("show", S.LABELS.index(shown))
    except Exception:
        pass
    print(f"peak={result.get('peak', '-'):6s} full={result.get('full', '-'):6s}")


def loop():
    if state.pending_cmd:
        cmd, state.pending_cmd = state.pending_cmd, None
        handle(cmd)

    try:
        samples = read_samples()
    except Exception as e:
        status(f"bridge error: {e}")
        time.sleep(1.0)
        return

    for t, v in samples:
        state.recent.append((t, v))
    state.new_since_classify += len(samples)
    state.count += len(samples)

    if state.mode == "record" and samples:
        lines = "\n".join(f"{t:.3f}," + ",".join(f"{x:.4f}" for x in v) for t, v in samples)
        client.publish(S.TOPIC_RAW, f"{state.label}\n{lines}", qos=1)

    if (state.mode == "classify" and len(state.recent) == S.WINDOW_N
            and state.new_since_classify >= S.STEP_N):
        state.new_since_classify = 0
        classify()

    # Report the real sample rate every 10 s (should be ~100 Hz).
    elapsed = time.monotonic() - state.rate_start
    if elapsed >= 10:
        status(f"mode={state.mode}; {state.count / elapsed:.0f} samples/s")
        if state.count == 0:
            report_sensor()
        state.count, state.rate_start = 0, time.monotonic()

    time.sleep(0.1)


App.run(user_loop=loop)
