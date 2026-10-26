"""
Start the working gait identifier for one person.

    python run.py                 # asks whose recordings to use
    python run.py --person fiona  # skip the question

1. Lists the people with recordings in data/ and asks which one (or everyone).
2. Trains both models on that person's recordings (a few seconds).
3. If those models aren't on the board yet, uploads them and restarts the app
   (deploy.py; asks for the board's password; about a minute). If they are,
   just checks the board is online, and starts the app if it isn't.
4. Opens the live view: both models' answers twice a second.

For a new person, record them first:  python guided_train.py
"""

import argparse
import glob
import os
import subprocess
import sys
import threading
from collections import Counter

import paho.mqtt.client as mqtt

import deploy
import live_view
from collect import DATA_DIR, HERE, S


def people():
    """{person: Counter of recordings per activity} from data/."""
    out = {}
    for path in glob.glob(os.path.join(DATA_DIR, "*.csv")):
        parts = os.path.basename(path).split("_")
        if len(parts) >= 3 and parts[0] in S.LABELS:
            out.setdefault(parts[1], Counter())[parts[0]] += 1
    return out


def choose_person(found):
    names = sorted(found)
    print("Whose gait should it identify?")
    for i, name in enumerate(names, 1):
        c = found[name]
        print(f"  {i}. {name:12s} " + ", ".join(f"{lab} {c[lab]}" for lab in S.LABELS))
    print(f"  {len(names) + 1}. everyone     (one model trained on all of them)")
    while True:
        try:
            ans = input(f"Pick 1-{len(names) + 1} [1]: ").strip() or "1"
        except EOFError:
            raise SystemExit()
        if ans in names or ans == "everyone":
            return None if ans == "everyone" else ans
        if ans.isdigit() and 1 <= int(ans) <= len(names) + 1:
            return None if int(ans) == len(names) + 1 else names[int(ans) - 1]
        print("  Type a number from the list.")


def board_online(seconds=12):
    """True if the board sends a status message over MQTT within `seconds`
    (it reports every 10 s while the app runs)."""
    heard = threading.Event()

    def on_connect(client, userdata, flags, reason_code, properties):
        client.subscribe(S.TOPIC_STATUS)

    def on_message(client, userdata, msg):
        if not msg.retain:
            heard.set()

    client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2)
    client.on_connect = on_connect
    client.on_message = on_message
    client.connect_async(S.MQTT_BROKER, S.MQTT_PORT, keepalive=30)
    client.loop_start()
    ok = heard.wait(seconds)
    client.loop_stop()
    client.disconnect()
    return ok


def main():
    ap = argparse.ArgumentParser(description="Start the gait identifier for one person")
    ap.add_argument("--person", help="whose recordings to train on ('everyone' for all)")
    ap.add_argument("--host", default=deploy.BOARD_HOST, help="board IP (default %(default)s)")
    args = ap.parse_args()

    found = people()
    if not found:
        raise SystemExit("No recordings in data/ yet. Record some first:  python guided_train.py")
    if args.person:
        person = None if args.person == "everyone" else args.person.lower()
        if person and person not in found:
            raise SystemExit(f"No recordings for '{person}'. People with data: "
                             f"{', '.join(sorted(found))}.\n"
                             f"Record them first:  python guided_train.py")
    else:
        person = choose_person(found)

    if person:
        missing = [lab for lab in S.LABELS if not found[person][lab]]
        if missing:
            raise SystemExit(f"{person} has no recordings of {', '.join(missing)} yet.\n"
                             "Record them first:  python guided_train.py")

    who = person or "everyone"
    print(f"\n--- Training on {who}'s recordings ---")
    cmd = [sys.executable, os.path.join(HERE, "train.py")] + (["--person", person] if person else [])
    result = subprocess.run(cmd, cwd=HERE, capture_output=True, text=True)
    if result.returncode:
        print(result.stdout + result.stderr)
        raise SystemExit("Training failed (see above).")
    for line in result.stdout.splitlines():           # just the accuracy lines
        if "-> using" in line:
            print(line.strip())

    print("\n--- Board ---")
    last = deploy.last_deployed()
    if last.get("fingerprint") == deploy.app_fingerprint() and last.get("host") == args.host:
        print(f"The board already has {who}'s models. Checking it's online...")
        if not board_online():
            print("No word from the board.")
            if not deploy.start(args.host):
                raise SystemExit(f"Couldn't reach the board at {args.host}. Is it powered on "
                                 "and on Tufts_Robot? (python deploy.py --host <IP> if its IP changed)")
    else:
        print(f"Putting {who}'s models on the board (enter the board's password if asked).")
        if not deploy.deploy(args.host, who):
            raise SystemExit(f"Couldn't reach the board at {args.host}. Is it powered on "
                             "and on Tufts_Robot? (python run.py --host <IP> if its IP changed)")

    print(f"\n--- Live: identifying {who}'s gait ---")
    live_view.main()


if __name__ == "__main__":
    main()
