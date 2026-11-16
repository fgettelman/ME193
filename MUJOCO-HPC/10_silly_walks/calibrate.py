"""Measure your real robot so the simulation and the policy match it. Run once (and again if you rebuild the robot):

    python3 calibrate.py           # the real Double Motor, over Bluetooth (turn it on first)
    python3 calibrate.py --sim     # practice on the simulated motor (sim_motor.py): it does the moves for you

It asks you to hold, tip, turn and spin the robot, and works out from the readings:
  * which way the motion sensor inside the motor faces (so tilt and turning are read in robot directions),
  * the units of the yaw and gyro readings,
  * which way each output counts, and the motor's real top speed at 100% (the simulation then uses it),
  * where each beam's angle sensor reads "long end straight down" (the simulation's beam angle 0).

It writes real_robot.json next to this file (sim_robot.json with --sim). Keep the robot built the same way afterwards:
if you take a beam off and put it back at a different angle, run this again.
"""
import argparse
import json
import math
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
NOTIFY_MS = 25          # ask the motor for new readings every 25 ms (the fastest it allows is 15)


def snapshot(hub):
    """Copy the latest raw readings (they are replaced every time a notification arrives)."""
    imu = hub.imu_device
    return {
        "acc": np.array([imu.accelerometerX, imu.accelerometerY, imu.accelerometerZ], dtype=float),
        "gyro": np.array([imu.gyroscopeX, imu.gyroscopeY, imu.gyroscopeZ], dtype=float),
        "yaw": float(imu.yaw),
        "position": np.array([hub.motor[0].position, hub.motor[1].position], dtype=float),
        "absolute": np.array([hub.motor[0].absolutePosition, hub.motor[1].absolutePosition], dtype=float),
    }


def record(hub, seconds):
    """Readings every 10 ms for `seconds`: a list of (time, snapshot)."""
    out, start = [], time.monotonic()
    while (t := time.monotonic() - start) < seconds:
        out.append((t, snapshot(hub)))
        time.sleep(0.01)
    return out


def average(samples, key):
    return np.mean([s[key] for _, s in samples], axis=0)


def unwrap_degrees(values):
    """Turn a list of 0..359 angle readings into one that keeps counting past a full turn."""
    steps = (np.diff(values) + 180) % 360 - 180
    return np.concatenate([[values[0]], values[0] + np.cumsum(steps)])


def circular_mean_deg(values):
    rad = np.radians(values)
    return math.degrees(math.atan2(np.mean(np.sin(rad)), np.mean(np.cos(rad)))) % 360


def ask(hub, step, text):
    """Tell the person what to do and wait for Enter; the simulated motor just does it."""
    print("\n" + text)
    if hasattr(hub, "act_out"):
        print("   (simulated: done for you)")
        hub.act_out(step)
        time.sleep(0.5)
    else:
        input("   Press Enter when ready... ")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--sim", action="store_true", help="use the simulated Double Motor instead of the real one")
    parser.add_argument("--card", help='the motor\'s Connection Card, e.g. --card "red 1131"')
    args = parser.parse_args()
    out_file = HERE / ("sim_robot.json" if args.sim else "real_robot.json")

    import robot_io
    hub = robot_io.connect(sim=args.sim, card=args.card, notify_ms=NOTIFY_MS)
    try:
        cal = calibrate(hub)
    finally:
        hub.movement_stop()
        hub.disconnect()
    out_file.write_text(json.dumps(cal, indent=2) + "\n")
    print(f"\nwrote {out_file.name}")
    if not args.sim:
        print("Next: train again with this file (see README: --with 10_silly_walks/real_robot.json), so the\n"
              "simulated motor has your motor's real top speed. Then run the policy: python3 run_real.py policy.npz")


def calibrate(hub) -> dict:
    # 0. Are readings arriving, and how often?
    changes, last, start = 0, hub.imu_device, time.monotonic()
    while time.monotonic() - start < 1.0:            # each notification brings a new reading object
        if hub.imu_device is not last:
            changes, last = changes + 1, hub.imu_device
        time.sleep(0.002)
    if np.isnan(snapshot(hub)["acc"]).any():
        raise SystemExit("No readings from the motor's motion sensor. Update its firmware in the LEGO Education app.")
    print(f"Connected. New sensor readings about {changes} times a second.")

    # 1. Level, in the air: which way is up for the sensor chip?
    ask(hub, "hold_level", "1/5  Pick the robot up and hold it LEVEL in the air, beams hanging down and free to spin.\n"
                           "     Keep it still.")
    level = record(hub, 1.5)
    acc_level = average(level, "acc")
    up_chip = acc_level / np.linalg.norm(acc_level)
    counts_per_g = float(np.linalg.norm(acc_level))
    print(f"     gravity reads {counts_per_g:.0f} counts")

    # 2. Still in the air: spin the beams forward at 50% and 100%.
    ask(hub, "hold_level", "2/5  Keep holding it level in the air. The beams will spin for 4 seconds: watch them.\n"
                           "     The bottom ends swing toward the BACK of the robot. The other side is its FRONT.")
    rates = {}
    for percent in (50, 100):
        hub.movement_move_tank(percent, percent, blocking=False)
        time.sleep(0.7)                                   # let it get up to speed
        spin = record(hub, 1.3)
        dt = spin[-1][0] - spin[0][0]
        position = np.array([s["position"] for _, s in spin])
        absolute = np.array([s["absolute"] for _, s in spin])
        rates[percent] = {
            "position": (position[-1] - position[0]) / dt,                       # degrees/s, each motor
            "absolute": np.array([unwrap_degrees(absolute[:, i])[-1] - absolute[0, i] for i in range(2)]) / dt,
        }
    hub.movement_stop(blocking=False)
    position_sign = [int(np.sign(r)) or 1 for r in rates[100]["position"]]
    absolute_sign = [int(np.sign(r)) or 1 for r in rates[100]["absolute"]]
    speed_100 = np.radians(np.abs(rates[100]["position"]))
    speed_50 = np.radians(np.abs(rates[50]["position"]))
    max_speed = float(np.mean(speed_100))
    print(f"     at 100%: left {speed_100[0]:.1f} rad/s, right {speed_100[1]:.1f} rad/s"
          f"   (at 50%: {speed_50[0]:.1f} and {speed_50[1]:.1f})")
    if abs(speed_100[0] - speed_100[1]) > 0.15 * max_speed:
        print("     NOTE: the two sides differ by more than 15%. Is something rubbing? The policy can cope a bit.")
    if hasattr(hub, "act_out"):
        long_end = "back"
    else:
        answer = ""
        while answer not in ("f", "b"):
            answer = input("     The motor sticks out much further on one side of the beams. Is that long end at the\n"
                           "     robot's FRONT or BACK? [f/b] ").strip().lower()[:1]
        long_end = {"f": "front", "b": "back"}[answer]

    # 3. Tip the front down: which way is forward for the sensor chip?
    ask(hub, "nose_down", "3/5  Still in the air, tip the robot's FRONT down, about 45 to 90 degrees. Hold it still.")
    acc_down = average(record(hub, 1.5), "acc")
    sideways = acc_down - np.dot(acc_down, up_chip) * up_chip
    tilt = math.degrees(math.atan2(np.linalg.norm(sideways), np.dot(acc_down, up_chip)))
    if tilt < 25:
        raise SystemExit(f"     It only tipped {tilt:.0f} degrees. Run again and tip the front further down.")
    forward_chip = -sideways / np.linalg.norm(sideways)       # tipping forward, the chip feels "up" toward the back
    left_chip = np.cross(up_chip, forward_chip)
    chip_axes = np.stack([forward_chip, left_chip, up_chip], axis=1)   # rows: chip x, y, z in robot axes
    print(f"     tipped {tilt:.0f} degrees")

    # 4. On the floor, turn it left a quarter turn: yaw direction and the gyro's units.
    ask(hub, "on_floor", "4/5  Put the robot down on the floor.")
    hub.imu_reset_yaw_axis(0)
    time.sleep(0.3)
    ask(hub, "turn_left", "     Next, after Enter: within 2 seconds turn it LEFT (anticlockwise seen from above)\n"
                          "     by about a quarter turn, then let go and keep your hands off.")
    if hasattr(hub, "act_out"):             # the simulated hand turns for 2 seconds, then stops
        turn = record(hub, 2.0)
        hub.act_out("stop_turning")
        turn += [(t + 2.0, s) for t, s in record(hub, 1.0)]
    else:
        turn = record(hub, 3.0)
    yaw_counts = turn[-1][1]["yaw"] - turn[0][1]["yaw"]
    yaw_rad_per_count = math.copysign(math.radians(0.1), yaw_counts)   # the motor reports yaw in tenths of a degree
    turned = yaw_counts * yaw_rad_per_count                              # radians, anticlockwise positive
    if not math.radians(40) < turned < math.radians(140):
        raise SystemExit(f"     It measured a turn of {math.degrees(turned):.0f} degrees. Run again and turn it "
                         "LEFT by about a quarter turn.")
    gyro_up = [np.dot(s["gyro"], up_chip) for _, s in turn]
    times = [t for t, _ in turn]
    gyro_integral = float(np.sum(np.diff(times) * np.array(gyro_up[:-1])))   # counts * seconds
    gyro_rad_s_per_count = turned / gyro_integral
    print(f"     turned {math.degrees(turned):.0f} degrees; one gyro count = "
          f"{math.degrees(gyro_rad_s_per_count):.3f} degrees/s")

    # 5. Beams pointing straight down: the beam angle sensors' zero.
    ask(hub, "beams_down", "5/5  Pick it up again. Turn each beam by hand until its LONG end points straight down\n"
                           "     (a beam held in its middle: either end). Hold still.")
    absolute = np.array([s["absolute"] for _, s in record(hub, 1.0)])
    beam_zero = [circular_mean_deg(absolute[:, i]) for i in range(2)]
    ask(hub, "on_floor", "     Done. Put the robot down.")

    return {
        "max_speed_rad_s": round(max_speed, 3),
        "motor_long_end": long_end,
        "position_sign": position_sign,
        "absolute_sign": absolute_sign,
        "beam_zero_deg": [round(z, 1) for z in beam_zero],
        "chip_axes": np.round(chip_axes, 4).tolist(),
        "yaw_rad_per_count": yaw_rad_per_count,
        "gyro_rad_s_per_count": gyro_rad_s_per_count,
        "measured": {
            "speed_100_rad_s": np.round(speed_100, 3).tolist(),
            "speed_50_rad_s": np.round(speed_50, 3).tolist(),
            "gravity_counts": round(counts_per_g, 1),
            "readings_per_second": changes,
            "date": time.strftime("%Y-%m-%d %H:%M"),
        },
    }


if __name__ == "__main__":
    main()
