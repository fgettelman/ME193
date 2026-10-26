"""
Record one labelled session from the wearable over MQTT.

    python collect.py walk --person fiona            # 60 s of walking
    python collect.py bike --person fiona --seconds 90

Tells the board "record <label>", saves every sample it streams back to
data/<label>_<person>_<YYYYmmdd-HHMMSS>.csv, then sends "stop" (which puts
the board back in classify mode if it has models). Start the activity during
the countdown; the first and last 3 s are trimmed off in training anyway.

guided_train.py uses record() to walk through a whole dataset step by step.
"""

import argparse
import os
import sys
import threading
import time

import paho.mqtt.client as mqtt

HERE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(HERE, "data")
sys.path.insert(0, os.path.join(HERE, "unoq_app", "python"))
import settings as S                              # noqa: E402


def record(label, person, seconds=60, countdown=5, save=True, quiet=False):
    """Record one session. Returns a dict with the rows, rate, gaps and (if
    save) the CSV path. Raises RuntimeError if the board can't be reached or
    sends nothing."""
    person = person.replace("_", "-")
    samples = {}                       # t -> row; dict drops QoS-1 duplicates
    connected = threading.Event()

    def on_connect(client, userdata, flags, reason_code, properties):
        if not reason_code.is_failure:
            client.subscribe([(S.TOPIC_RAW, 1), (S.TOPIC_STATUS, 0)])
            connected.set()

    def on_message(client, userdata, msg):
        text = msg.payload.decode(errors="replace")
        if msg.topic == S.TOPIC_STATUS:
            if not quiet:
                print(f"  board: {text}")
            return
        lab, _, body = text.partition("\n")
        if lab != label:
            return
        for line in body.splitlines():
            parts = line.split(",")
            if len(parts) == 7:
                samples[float(parts[0])] = parts

    client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2)
    client.on_connect = on_connect
    client.on_message = on_message
    client.connect_async(S.MQTT_BROKER, S.MQTT_PORT, keepalive=30)
    client.loop_start()
    if not connected.wait(10):
        client.loop_stop()
        raise RuntimeError(f"Could not reach {S.MQTT_BROKER}.")

    try:
        for i in range(countdown, 0, -1):
            print(f"Start {label.upper()} in {i}...\a")
            time.sleep(1)
        client.publish(S.TOPIC_CMD, f"record {label}", qos=1)
        print(f"RECORDING {label} for {seconds:.0f} s  (Ctrl+C to stop early)\a")
        end = time.monotonic() + seconds
        while time.monotonic() < end:
            time.sleep(1)
            print(f"\r  {len(samples):6d} samples, {end - time.monotonic():4.0f} s left ", end="", flush=True)
    except KeyboardInterrupt:
        print("\nStopped early.")
    finally:
        client.publish(S.TOPIC_CMD, "stop", qos=1)
        time.sleep(1.5)                # let the last batches arrive
        client.loop_stop()
        client.disconnect()
    print("\a")

    if not samples:
        raise RuntimeError("No samples arrived. Is the board running the app and online? "
                           "(python live_view.py shows its status)")
    rows = [samples[t] for t in sorted(samples)]
    ts = [float(r[0]) for r in rows]
    duration = ts[-1] - ts[0]
    rate = (len(rows) - 1) / duration if duration > 0 else 0
    gaps = sum(1 for a, b in zip(ts, ts[1:]) if b - a > 3 / S.SAMPLE_HZ)
    out = {"rows": rows, "duration": duration, "rate": rate, "gaps": gaps, "path": None}

    if save:
        os.makedirs(DATA_DIR, exist_ok=True)
        path = os.path.join(DATA_DIR, f"{label}_{person}_{time.strftime('%Y%m%d-%H%M%S')}.csv")
        with open(path, "w") as f:
            f.write("t," + ",".join(S.COLUMNS) + "\n")
            for r in rows:
                f.write(",".join(r) + "\n")
        out["path"] = path
    print(f"Got {len(rows)} samples ({duration:.1f} s, {rate:.0f} Hz, {gaps} gaps)"
          + (f", saved to {out['path']}" if save else ""))
    if rate < 0.9 * S.SAMPLE_HZ or gaps:
        print("  Warning: samples are missing. Check the 'samples/s' status from the board.")
    return out


def main():
    ap = argparse.ArgumentParser(description="Record a labelled gait session")
    ap.add_argument("label", choices=S.LABELS)
    ap.add_argument("--person", default="me", help="who is wearing it (no underscores)")
    ap.add_argument("--seconds", type=float, default=60)
    ap.add_argument("--countdown", type=int, default=5)
    args = ap.parse_args()
    try:
        record(args.label, args.person, args.seconds, args.countdown)
    except RuntimeError as e:
        raise SystemExit(str(e))


if __name__ == "__main__":
    main()
