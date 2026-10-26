"""
Watch the wearable live, and send it commands.

    python live_view.py

Shows both models' answers side by side as they arrive, the board's
status messages, and (while recording) the live sensor values. Type a
command and press Enter to send it to the board:
    classify | idle | stop | reload | record <label> | q (quit)
"""

import json
import os
import sys
import time

import paho.mqtt.client as mqtt

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "unoq_app", "python"))
import settings as S                              # noqa: E402


def on_connect(client, userdata, flags, reason_code, properties):
    if reason_code.is_failure:
        print(f"Could not connect to {S.MQTT_BROKER}: {reason_code}")
        return
    client.subscribe([(S.TOPIC_RESULT, 0), (S.TOPIC_STATUS, 0), (S.TOPIC_RAW, 0)])
    print(f"Connected. Listening on {S.MQTT_BASE}/...")


last_raw = 0.0


def show_raw(text):
    """While recording: the newest sample, at most twice a second."""
    global last_raw
    if time.monotonic() - last_raw < 0.5:
        return
    label, _, body = text.partition("\n")
    lines = body.splitlines()
    parts = lines[-1].split(",") if lines else []
    if len(parts) != 7:
        return
    last_raw = time.monotonic()
    ax, ay, az, gx, gy, gz = (float(v) for v in parts[1:])
    print(f"  [{label}] accel {ax:+5.2f} {ay:+5.2f} {az:+5.2f} g   "
          f"gyro {gx:+7.1f} {gy:+7.1f} {gz:+7.1f} deg/s")


def on_message(client, userdata, msg):
    text = msg.payload.decode(errors="replace")
    if msg.topic == S.TOPIC_STATUS:
        print(f"  [board] {text}")
        return
    if msg.topic == S.TOPIC_RAW:
        show_raw(text)
        return
    try:
        r = json.loads(text)
    except ValueError:
        return
    peak = r.get("peak", "-")
    full = r.get("full", "-")
    agree = "" if peak == full else "   <- models disagree"
    g = f"  (peak {r['peak_g']:.2f} g)" if "peak_g" in r else ""
    print(f"  model 1 (peak): {peak:6s}   model 2 (full): {full:6s}{g}{agree}")


def main():
    client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2)
    client.on_connect = on_connect
    client.on_message = on_message
    client.connect_async(S.MQTT_BROKER, S.MQTT_PORT, keepalive=30)
    client.loop_start()
    print(__doc__.strip().split("\n\n", 2)[2])
    try:
        while True:
            cmd = input().strip()
            if cmd in ("q", "quit"):
                break
            if cmd:
                client.publish(S.TOPIC_CMD, cmd, qos=1)
    except (KeyboardInterrupt, EOFError):
        pass
    finally:
        client.loop_stop()
        client.disconnect()


if __name__ == "__main__":
    main()
