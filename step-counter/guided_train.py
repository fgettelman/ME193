"""
Step-by-step trainer: tells you what to do at each point, from checking the
board to recording to training and testing the step-counting net.

    python guided_train.py

1. Checks the board is online and streaming at ~100 Hz.
2. Checks the sensor is strapped on the right way (+X along the thigh).
3. Asks your height (for distance, until you calibrate it in step 5).
4. Activity recordings: 3 each of still, walk and run, at different paces.
   The net learns from these; their steps are labelled automatically.
5. Counted recordings: you walk (or run) a number of steps while counting
   them yourself, and type the count in. Optionally over a distance you
   measured, which calibrates your step length. These test the counter
   against the truth.
6. Runs train.py and evaluate.py, then says how to put the net on the board.

Recordings already in data/ for you count, so you can quit and pick up where
you left off. Redone recordings are moved to data_discarded/, not deleted.
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
import learn
from collect import DATA_DIR, HERE, S

DISCARD_DIR = os.path.join(HERE, "data_discarded")

# What to do for each activity recording. Recording k uses PLAN[label][k].
PLAN = {
    "still": [
        "Stand still, relaxed, arms at your sides.",
        "Stand in place but shift your weight from foot to foot and fidget a bit.",
        "Sit down and stay seated (fidgeting is fine).",
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
}
# Counted recordings: (activity, steps to aim for, seconds to record, what to do).
COUNTED = [
    ("walk", 40, 45, "WALK at your normal pace"),
    ("walk", 40, 50, "WALK SLOWLY"),
    ("run", 50, 40, "JOG"),
]
# Extra seconds of countdown for activities you can't start from the laptop.
EXTRA_COUNTDOWN = {"still": 0, "walk": 5, "run": 10}
MIN_SECONDS = 10                  # shorter than this gives almost no windows


# --- Small helpers ----------------------------------------------------------

def ask(prompt, default=""):
    try:
        ans = input(prompt).strip()
    except EOFError:
        raise SystemExit("\nBye.")
    return ans or default


def ask_number(prompt, default=None):
    while True:
        ans = ask(prompt, "" if default is None else str(default))
        if not ans:
            return None
        try:
            return float(ans)
        except ValueError:
            print("  Type a number.")


def heading(text):
    print("\n" + "=" * 70 + f"\n{text}\n" + "=" * 70)


def recordings(person, label, counted):
    """This person's recordings of label: the counted ones or the others."""
    have = learn.read_counts()
    paths = sorted(glob.glob(os.path.join(DATA_DIR, f"{label}_{person}_*.csv")))
    return [p for p in paths if (os.path.splitext(os.path.basename(p))[0] in have) == counted]


def discard(path):
    os.makedirs(DISCARD_DIR, exist_ok=True)
    shutil.move(path, os.path.join(DISCARD_DIR, os.path.basename(path)))
    print(f"  Moved to {DISCARD_DIR}/")


def check_recording(r):
    problems = []
    if r["duration"] < MIN_SECONDS:
        problems.append(f"only {r['duration']:.0f} s long")
    if r["rate"] < 0.9 * S.SAMPLE_HZ or r["gaps"]:
        problems.append("samples went missing")
    if problems:
        print("  Problem: " + ", ".join(problems) + ". Redoing is recommended.")


# --- Step 1: board online ---------------------------------------------------

def check_board():
    heading("Step 1: is the board online?")
    print("Turn on the wearable (power bank plugged in). The Step Counter app\n"
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
        print("  Check: is the app deployed and running (python deploy.py --start)?\n"
              "  Is the UNO Q on Wi-Fi?")
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


# --- Step 3: height ---------------------------------------------------------

def ask_height(person):
    heading("Step 3: your height (for distance)")
    known = learn.read_people().get(person, {}).get("height_cm")
    print("Distance = steps x step length. Until you calibrate it in step 5, step\n"
          "length is estimated from your height (about 0.41 x height walking).")
    while True:
        ans = ask(f"Your height in cm, e.g. 168, or feet'inches, e.g. 5'6"
                  f"{f' [{known:.0f} cm]' if known else ''}: ",
                  str(known) if known else "").replace('"', "")
        if not ans:
            print(f"  Skipped; using {S.DEFAULT_HEIGHT_CM} cm.")
            return
        try:
            if "'" in ans:
                feet, _, inches = ans.partition("'")
                h = (float(feet) * 12 + float(inches or 0)) * 2.54
            else:
                h = float(ans)
        except ValueError:
            h = 0
        if 100 <= h <= 230:
            break
        print("  That doesn't look like a height. Try e.g. 168 or 5'6.")
    learn.save_height(person, round(h))
    print(f"  Saved {h:.0f} cm in people.json.")


# --- Step 4: activity recordings --------------------------------------------

def record_activities(person, seconds):
    heading("Step 4: record each activity")
    print(f"Plan: {len(S.LABELS)} activities x 3 recordings of {seconds:.0f} s each.\n"
          "Before each one I'll say what to do. The laptop beeps when the\n"
          "countdown starts and when recording starts and stops. Keep doing the\n"
          "activity until the final beep; the first and last 3 s are cut off\n"
          "anyway. You don't need to be near the laptop: data goes over Wi-Fi.")
    steps = [(lab, k) for k in range(3) for lab in S.LABELS]   # mix activities up
    for i, (label, k) in enumerate(steps, 1):
        if len(recordings(person, label, counted=False)) > k:
            continue                                          # done in an earlier run
        while True:
            countdown = 5 + EXTRA_COUNTDOWN[label]
            print(f"\n--- Recording {i}/{len(steps)}: {label.upper()} #{k + 1} ---")
            print(f"  {PLAN[label][k]}")
            print(f"  You'll get a {countdown} s countdown to get into position.")
            choice = ask("  [Enter] record   [s] skip   [q] go to the next step: ").lower()
            if choice == "q":
                return
            if choice == "s":
                break
            try:
                r = collect.record(label, person, seconds, countdown, quiet=True)
            except RuntimeError as e:
                print(f"  {e}")
                continue
            check_recording(r)
            if ask("  [Enter] keep it   [r] redo it: ").lower() == "r":
                discard(r["path"])
                continue
            break


# --- Step 5: counted recordings ---------------------------------------------

def record_counted(person):
    heading("Step 5: count your own steps (the real test)")
    print("Now you count, so we can check the net against the truth.\n"
          "For each one: stand still during the countdown. When recording\n"
          "starts, do the activity while counting EVERY step (both feet), stop\n"
          "after about the number shown, and stand still until the final beep.\n"
          "Then type how many steps you actually took.\n\n"
          "For distance: if you can, walk along something you can measure\n"
          "(a hallway, a track, a sidewalk you paced out) and type the metres.\n"
          "That calibrates your step length. Otherwise just press Enter.")
    for i, (label, target, seconds, what) in enumerate(COUNTED, 1):
        k = sum(1 for a, *_ in COUNTED[:i - 1] if a == label)
        if len(recordings(person, label, counted=True)) > k:
            continue
        while True:
            print(f"\n--- Counted {i}/{len(COUNTED)}: {what}, about {target} steps ---")
            choice = ask("  [Enter] record   [s] skip   [q] go to training: ").lower()
            if choice == "q":
                return
            if choice == "s":
                break
            try:
                r = collect.record(label, person, seconds, countdown=5 + EXTRA_COUNTDOWN[label],
                                   quiet=True)
            except RuntimeError as e:
                print(f"  {e}")
                continue
            check_recording(r)
            steps = ask_number(f"  How many steps did you take? [{target}] ", target)
            if ask("  [Enter] keep it   [r] redo it: ").lower() == "r":
                discard(r["path"])
                continue
            meters = ask_number("  Distance covered in metres (Enter to skip): ")
            name = os.path.splitext(os.path.basename(r["path"]))[0]
            learn.save_count(DATA_DIR, name, int(steps), meters)
            break


# --- Step 6: train ----------------------------------------------------------

def train(person):
    heading("Step 6: train and test the net")
    have = {lab: len(recordings(person, lab, counted=False)) for lab in S.LABELS}
    print("Your activity recordings: " + ", ".join(f"{lab} {n}" for lab, n in have.items()))
    missing = [lab for lab, n in have.items() if n == 0]
    if missing:
        print(f"Can't train yet: no recordings of {', '.join(missing)}.")
        return False
    ask("Press Enter to train (about 15 s) and test (a couple of minutes)... ")
    for script in ("train.py", "evaluate.py"):
        print(f"\n$ python {script} --person {person}")
        cmd = [sys.executable, os.path.join(HERE, script), "--person", person]
        if subprocess.run(cmd, cwd=HERE).returncode:
            print(f"{script} failed (see above).")
            return False
    return True


def main():
    heading("Step counter: guided training")
    print("This walks you through checking the wearable, recording data, and\n"
          "training the step-counting net. Ctrl+C quits at any time; recordings\n"
          "that were kept stay in data/ and are picked up next time.")
    person = ask("\nWho is wearing the sensor? (one word, e.g. fiona): ", "me")
    person = person.lower().replace("_", "-").replace(" ", "-")
    done = {lab: len(recordings(person, lab, counted=False)) for lab in S.LABELS}
    n_counted = sum(len(recordings(person, lab, counted=True)) for lab in S.LABELS)
    if any(done.values()) or n_counted:
        print(f"Already recorded for {person}: "
              + ", ".join(f"{lab} {n}/3" for lab, n in done.items())
              + f", counted {n_counted}/{len(COUNTED)}")

    todo = sum(min(n, 3) for n in done.values()) < 3 * len(S.LABELS) or n_counted < len(COUNTED)
    if todo and ask("Record now? [Y/n] ", "y").lower().startswith("y"):
        secs = ask("Seconds per activity recording [30]: ", "30")
        seconds = max(float(secs), MIN_SECONDS + 5)
        if not check_board():
            raise SystemExit("Get the board online, then run this again.")
        check_orientation(person)
        ask_height(person)
        record_activities(person, seconds)
        record_counted(person)
    elif person not in learn.read_people():
        ask_height(person)

    if not train(person):
        raise SystemExit("Record the missing activities (run this again), then train.")

    heading("Step 7: put the net on the board")
    print("The new net was copied to unoq_app/python/models/. To use it:\n"
          f"  python run.py --person {person}\n"
          "uploads it to the UNO Q (asks for the board's password) and opens\n"
          "the live dashboard. Plots are in results/: step_counts.png and\n"
          "step_trace.png.")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nQuit. Kept recordings are in data/; run again to continue.")
