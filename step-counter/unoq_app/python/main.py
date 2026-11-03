"""
Step counter, UNO Q Linux side.

Pulls IMU samples from the sketch, and depending on the mode:
  count           counts steps from the spikes and dips in the gyro
                  (spikes.py). About 10 times a second it publishes the thigh
                  swing signal and its steps (for the dashboard's live
                  graph), and every 0.5 s
                  the running step count, distance, cadence and activity
  record <label>  streams the raw samples over MQTT (collect.py on the laptop
                  saves them to a CSV)
  idle            just keeps reading (the totals are kept, not counted on)

Commands arrive on ME193/Rogers/Fiona/steps/cmd:
  "count", "idle", "reset" (totals back to 0), "record walk", "stop"

The calibration from the 5 m walks (step length) arrives as
a retained message on .../calibration, so the board gets it whenever it
connects. It's also saved to python/calibration.json in case the broker
forgets it.
"""

import json
import os
import sys
import time

import numpy as np
import paho.mqtt.client as mqtt
from arduino.app_utils import App, Bridge

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import settings as S                              # noqa: E402
import spikes                                     # noqa: E402

CALIB_PATH = os.path.join(HERE, "calibration.json")
SCALE = np.array([1 / S.ACCEL_COUNTS_PER_G] * 3 + [1 / S.GYRO_COUNTS_PER_DPS] * 3)
LIVE_SPIKES_S = 12          # live messages carry the spikes from the last 12 s


def saved_calibration():
    try:
        with open(CALIB_PATH) as f:
            return json.load(f)
    except (OSError, ValueError):
        return None


class State:
    def __init__(self):
        self.counter = spikes.SpikeCounter(saved_calibration())
        self.mode = "count"
        self.label = None
        self.count = 0
        self.rate_start = time.monotonic()
        self.next_result = 0.0
        self.pending_cmd = None
        self.pending_calib = None


state = State()


# --- MQTT -------------------------------------------------------------------

def on_connect(client, userdata, flags, reason_code, properties):
    if not reason_code.is_failure:
        client.subscribe([(S.TOPIC_CMD, 1), (S.TOPIC_CALIB, 1)])
        status(f"connected; mode={state.mode}")


def on_message(client, userdata, msg):
    # Handled in the main loop so the loop never sees a half-changed state.
    text = msg.payload.decode(errors="replace").strip()
    if msg.topic == S.TOPIC_CALIB:
        state.pending_calib = text
    else:
        state.pending_cmd = text


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
        state.mode, state.label = "count", None
    elif parts[0] in ("count", "idle"):
        state.mode, state.label = parts[0], None
    elif parts[0] == "reset":
        state.counter.reset()
    else:
        status(f"unknown command: {cmd!r}")
        return
    status(f"mode={state.mode}" + (f" label={state.label}" if state.label else ""))


def apply_calibration(text):
    try:
        calib = json.loads(text) if text else None
    except ValueError:
        status(f"bad calibration: {text!r}")
        return
    state.counter.calibrate(calib)
    try:
        with open(CALIB_PATH, "w") as f:
            json.dump(calib, f)
    except OSError:
        pass
    c = state.counter.cal
    status(f"calibration: {c['step_length_m']['walk']:.2f} m per step walking, "
           f"{c['step_length_m']['run']:.2f} m running")


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


def count(samples):
    c = state.counter
    ts, vals = [], []
    for t, v in samples:
        s = c.add(t, v[3:])
        ts.append(round(t, 2))
        vals.append(round(s, 1))
    now = samples[-1][0]
    summary = c.summary(now)
    # Every other sample (50 Hz) is plenty for the graph.
    client.publish(S.TOPIC_LIVE, json.dumps({
        "t": ts[::2], "s": vals[::2], "now": now,
        "spikes": [sp for sp in c.spikes if now - sp["t"] < LIVE_SPIKES_S],
        "threshold": spikes.SPIKE_DPS, "dip": spikes.DIP_DPS,
        **summary,
    }))
    if time.monotonic() >= state.next_result:
        state.next_result = time.monotonic() + S.STEP_S
        result = {**summary, "t": time.time()}
        client.publish(S.TOPIC_RESULT, json.dumps(result))
        try:
            Bridge.call("show", S.LABELS.index(result["activity"]))
        except Exception:
            pass
        print(f"{result['activity']:5s} steps={result['steps']:7.0f} "
              f"distance={result['distance_m']:7.1f} m  cadence={result['cadence_spm']:5.0f}/min")


def loop():
    if state.pending_cmd:
        cmd, state.pending_cmd = state.pending_cmd, None
        handle(cmd)
    if state.pending_calib is not None:
        text, state.pending_calib = state.pending_calib, None
        apply_calibration(text)

    try:
        samples = read_samples()
    except Exception as e:
        status(f"bridge error: {e}")
        time.sleep(1.0)
        return
    state.count += len(samples)

    if state.mode == "record" and samples:
        lines = "\n".join(f"{t:.3f}," + ",".join(f"{x:.4f}" for x in v) for t, v in samples)
        client.publish(S.TOPIC_RAW, f"{state.label}\n{lines}", qos=1)

    if state.mode == "count" and samples:
        count(samples)

    # Report the real sample rate every 10 s (should be ~100 Hz).
    elapsed = time.monotonic() - state.rate_start
    if elapsed >= 10:
        status(f"mode={state.mode}; {state.count / elapsed:.0f} samples/s")
        if state.count == 0:
            report_sensor()
        state.count, state.rate_start = 0, time.monotonic()

    time.sleep(0.1)


App.run(user_loop=loop)
