"""
Step-by-step trainer: tells you what to do at each point, from checking the
board to recording every activity to training and evaluating both models.

    python guided_train.py

1. Checks the board is online and streaming at ~100 Hz.
2. Checks the sensor is strapped on the right way (+X along the thigh).
3. Walks through the recording plan: 3 recordings of each activity, each at
   a different pace or style. Before each one it says what to do; after each
   one you keep it or redo it. Recordings already in data/ for you count, so
   you can quit and pick up where you left off.
4. Runs train.py and evaluate.py, then says how to put the models on the board.

Redone recordings are moved to data_discarded/, not deleted.
"""

import glob
import os
import shutil
import subprocess
import sys
import time

import numpy as np
import paho.mqtt.client as mqtt

import collect
from collect import DATA_DIR, HERE, S

DISCARD_DIR = os.path.join(HERE, "data_discarded")

# What to do for each recording. Recording k of an activity uses PLAN[label][k].
PLAN = {
    "still": [
        "Stand still, relaxed, arms at your sides.",
        "Stand in place but shift your weight from foot to foot and fidget a bit.",
        "Stand still again (e.g. as if waiting in line, looking around).",
    ],
    "walk": [
        "Walk at a SLOW, easy pace.",
        "Walk at your NORMAL pace.",
        "Walk BRISKLY, like you're late for class.",
    ],
    "run": [
        "JOG slowly.",
        "Run at a NORMAL running pace.",
        "Run FAST (but keep it up for the whole recording).",
    ],
    "bike": [
        "Ride a REAL BIKE (or a stationary bike) at a steady pace.",
        "Lie on your BACK and AIR-BIKE (pedal in the air).",
        "Bike again, real or air-bike, at a DIFFERENT cadence than before.",
    ],
}
# Extra seconds of countdown for activities you can't start from the laptop.
EXTRA_COUNTDOWN = {"still": 0, "walk": 5, "run": 10, "bike": 15}
MIN_SECONDS = 10                  # shorter than this gives almost no windows


# --- Small helpers ----------------------------------------------------------

def ask(prompt, default=""):
    try:
        ans = input(prompt).strip()
    except EOFError:
        raise SystemExit("\nBye.")
    return ans or default


def heading(text):
    print("\n" + "=" * 70 + f"\n{text}\n" + "=" * 70)


def recordings(person, label):
    return sorted(glob.glob(os.path.join(DATA_DIR, f"{label}_{person}_*.csv")))


def discard(path):
    os.makedirs(DISCARD_DIR, exist_ok=True)
    shutil.move(path, os.path.join(DISCARD_DIR, os.path.basename(path)))
    print(f"  Moved to {DISCARD_DIR}/")


# --- Step 1: board online ---------------------------------------------------

def check_board():
    heading("Step 1: is the board online?")
    print("Turn on the wearable (power bank plugged in). The Gait Identifier app\n"
          "starts by itself once it's been deployed (python deploy.py).")
    ask("Press Enter when the app is running... ")

    msgs = []

    def on_connect(client, userdata, flags, reason_code, properties):
        client.subscribe(S.TOPIC_STATUS)

    def on_message(client, userdata, msg):
        msgs.append((msg.retain, msg.payload.decode(errors="replace")))

    client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2)
    client.on_connect = on_connect
    client.on_message = on_message
    client.connect_async(S.MQTT_BROKER, S.MQTT_PORT, keepalive=30)
    client.loop_start()
    print("Listening for the board (it reports every 10 s)...")
    fresh = None
    end = time.monotonic() + 15
    while time.monotonic() < end and fresh is None:
        time.sleep(0.5)
        fresh = next((t for r, t in msgs if not r), None)
    client.loop_stop()
    client.disconnect()

    if fresh is None:
        last = msgs[-1][1] if msgs else "nothing"
        print(f"  No live message from the board in 15 s (last known status: {last}).")
        print("  Check: is the app running in App Lab? Is the UNO Q on Wi-Fi?")
        return ask("Try anyway? [y/N] ", "n").lower().startswith("y")
    print(f"  Board says: {fresh}")
    if "samples/s" in fresh:
        rate = float(fresh.split(";")[-1].split()[0])
        if rate < 0.9 * S.SAMPLE_HZ:
            print(f"  Warning: only {rate:.0f} samples/s (want {S.SAMPLE_HZ}). "
                  "Check the sensor wiring.")
    return True


# --- Step 2: sensor orientation ---------------------------------------------

def check_orientation(person):
    heading("Step 2: is the sensor on the right way?")
    print("Strap the sensor to the FRONT of your thigh, just above the knee,\n"
          "with the +X arrow pointing DOWN toward your knee. Strap it firmly so\n"
          "it can't rotate, and mount it the same way every time.\n"
          "Then stand up straight and still.")
    while True:
        ask("Press Enter to take a 5 s test reading... ")
        try:
            r = collect.record("still", person, seconds=5, countdown=3, save=False, quiet=True)
        except RuntimeError as e:
            print(f"  {e}")
            if ask("Retry? [Y/n] ", "y").lower().startswith("n"):
                return
            continue
        mean = np.array([[float(x) for x in row[1:4]] for row in r["rows"]]).mean(axis=0)
        print(f"  Average accel: ax={mean[0]:+.2f} g  ay={mean[1]:+.2f} g  az={mean[2]:+.2f} g")
        # At rest an accelerometer reads +1 g along whichever axis points UP,
        # so +X pointing down the thigh reads about -1 g when standing.
        if mean[0] < -0.8:
            print("  Good: +X points down your thigh (about -1 g when standing).")
            return
        if mean[0] > 0.8:
            print("  The sensor is UPSIDE DOWN: +X points up toward your hip. Flip it.")
        else:
            print("  +X isn't along your thigh: the sensor is rotated on the strap.\n"
                  "  Rotate it so the +X arrow points straight down toward the knee.")
        if ask("Fix it and test again? [Y/n] ", "y").lower().startswith("n"):
            return


# --- Step 3: record ---------------------------------------------------------

def record_all(person, seconds):
    heading("Step 3: record every activity")
    print(f"Plan: {len(S.LABELS)} activities x 3 recordings of {seconds:.0f} s each.\n"
          "Before each one I'll say what to do. The laptop beeps when the\n"
          "countdown starts and when recording starts and stops. Keep doing the\n"
          "activity until the final beep; the first and last 3 s are cut off\n"
          "anyway. You don't need to be near the laptop: data goes over Wi-Fi.")
    steps = [(lab, k) for k in range(3) for lab in S.LABELS]   # mix activities up
    for i, (label, k) in enumerate(steps, 1):
        if len(recordings(person, label)) > k:
            continue                                          # done in an earlier run
        while True:
            countdown = 5 + EXTRA_COUNTDOWN[label]
            print(f"\n--- Recording {i}/{len(steps)}: {label.upper()} #{k + 1} ---")
            print(f"  {PLAN[label][k]}")
            print(f"  You'll get a {countdown} s countdown to get into position.")
            choice = ask("  [Enter] record   [s] skip   [q] stop recording and train: ").lower()
            if choice == "q":
                return
            if choice == "s":
                break
            try:
                r = collect.record(label, person, seconds, countdown, quiet=True)
            except RuntimeError as e:
                print(f"  {e}")
                continue
            problems = []
            if r["duration"] < MIN_SECONDS:
                problems.append(f"only {r['duration']:.0f} s long")
            if r["rate"] < 0.9 * S.SAMPLE_HZ or r["gaps"]:
                problems.append("samples went missing")
            if problems:
                print("  Problem: " + ", ".join(problems) + ". Redoing is recommended.")
            keep = ask("  [Enter] keep it   [r] redo it: ").lower()
            if keep == "r":
                discard(r["path"])
                continue
            break


# --- Step 4: train ----------------------------------------------------------

def counts(person=None):
    pattern = "{label}_*.csv" if person is None else "{label}_" + person + "_*.csv"
    return {lab: len(glob.glob(os.path.join(DATA_DIR, pattern.format(label=lab))))
            for lab in S.LABELS}


def train():
    heading("Step 4: train and evaluate")
    c = counts()
    people = {os.path.basename(p).split("_")[1] for p in glob.glob(os.path.join(DATA_DIR, "*.csv"))}
    print("Recordings in data/ (all people): "
          + ", ".join(f"{lab} {n}" for lab, n in c.items()))
    print(f"People: {', '.join(sorted(people)) or 'none'}")
    missing = [lab for lab, n in c.items() if n == 0]
    if missing:
        print(f"Can't train yet: no recordings of {', '.join(missing)}.")
        return False
    if min(c.values()) < 2:
        print("Note: some activity has only 1 recording, so the accuracy check is\n"
              "less trustworthy. Record each at least twice for a fair test.")
    if len(people) < 2:
        print("Note: only one person so far. Add a second person later to test\n"
              "whether it works on someone it never saw.")
    ask("Press Enter to train (takes up to a minute)... ")
    for script in ("train.py", "evaluate.py"):
        print(f"\n$ python {script}")
        if subprocess.run([sys.executable, os.path.join(HERE, script)], cwd=HERE).returncode:
            print(f"{script} failed (see above).")
            return False
    return True


def main():
    heading("Gait identifier: guided training")
    print("This walks you through checking the wearable, recording data,\n"
          "and training both models. Ctrl+C quits at any time; recordings\n"
          "that were kept stay in data/ and are picked up next time.")
    person = ask("\nWho is wearing the sensor? (one word, e.g. fiona): ", "me")
    person = person.lower().replace("_", "-").replace(" ", "-")
    done = counts(person)
    if any(done.values()):
        print(f"Already recorded for {person}: "
              + ", ".join(f"{lab} {n}/3" for lab, n in done.items()))

    if sum(min(n, 3) for n in done.values()) < 3 * len(S.LABELS):
        if ask("Record now? [Y/n] ", "y").lower().startswith("y"):
            secs = ask("Seconds per recording [60]: ", "60")
            seconds = max(float(secs), MIN_SECONDS + 5)
            if not check_board():
                raise SystemExit("Get the board online, then run this again.")
            check_orientation(person)
            record_all(person, seconds)

    if not train():
        raise SystemExit("Record the missing activities (run this again), then train.")

    heading("Step 5: put the models on the board")
    print("The new models were copied to unoq_app/python/models/. To use them:\n"
          "  1. Run   python deploy.py   (uploads them to the UNO Q over SSH and\n"
          "     restarts the app; asks for the board's password).\n"
          "  2. Run   python live_view.py   to watch both models' answers live.\n"
          "Plots are in results/: peak_histogram.png, confusion.png,\n"
          "feature_scatter.png.")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nQuit. Kept recordings are in data/; run again to continue.")
