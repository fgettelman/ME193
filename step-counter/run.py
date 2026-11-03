"""
Start the step counter for one person.

    python run.py                 # asks who's wearing it
    python run.py --person fiona  # skip the question

1. Asks who's wearing it (their 5 m calibration walks are kept under their name).
2. If the board doesn't have the latest app yet, uploads it and restarts it
   (deploy.py; asks for the board's password; about a minute). If it does,
   just checks the board is online, and starts the app if it isn't.
3. Opens the live dashboard in your browser: the gyro's spikes live, the
   step count, and the 5 m calibration ("training"), which takes a minute.
"""

import argparse
import json
import sys
import threading

import paho.mqtt.client as mqtt

import dashboard
import deploy
from collect import S


def people():
    """Names with saved 5 m walks."""
    try:
        with open(dashboard.CALIB_FILE) as f:
            return sorted(json.load(f))
    except (OSError, ValueError):
        return []


def choose_person(names):
    if not names:
        ans = input("Who's wearing it? (a name, e.g. fiona): ").strip().lower()
        return ans or "me"
    print("Who's wearing it?")
    for i, name in enumerate(names, 1):
        print(f"  {i}. {name}")
    print("  Or type a new name.")
    while True:
        try:
            ans = input(f"Pick 1-{len(names)} or a name [1]: ").strip().lower() or "1"
        except EOFError:
            raise SystemExit()
        if ans.isdigit():
            if 1 <= int(ans) <= len(names):
                return names[int(ans) - 1]
            print("  Type a number from the list, or a name.")
            continue
        return ans.replace(" ", "-")


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
    ap = argparse.ArgumentParser(description="Start the step counter for one person")
    ap.add_argument("--person", help="who's wearing it")
    ap.add_argument("--host", default=deploy.BOARD_HOST, help="board IP (default %(default)s)")
    args = ap.parse_args()
    person = args.person.lower() if args.person else choose_person(people())

    print("\n--- Board ---")
    last = deploy.last_deployed()
    if last.get("fingerprint") == deploy.app_fingerprint() and last.get("host") == args.host:
        print("The board has the latest app. Checking it's online...")
        if not board_online():
            print("No word from the board.")
            if not deploy.start(args.host):
                raise SystemExit(f"Couldn't reach the board at {args.host}. Is it powered on "
                                 "and on Tufts_Robot? (python run.py --host <IP> if its IP changed)")
    else:
        print("Putting the latest app on the board (enter the board's password if asked).")
        if not deploy.deploy(args.host, person):
            raise SystemExit(f"Couldn't reach the board at {args.host}. Is it powered on "
                             "and on Tufts_Robot? (python run.py --host <IP> if its IP changed)")

    print(f"\n--- Live: counting {person}'s steps ---")
    print("To calibrate, use 'Calibrate: walk 5 m' on the page.")
    sys.argv = [sys.argv[0], "--person", person]
    dashboard.main()


if __name__ == "__main__":
    main()
