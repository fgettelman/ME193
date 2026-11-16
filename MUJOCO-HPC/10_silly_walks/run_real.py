"""Walk the real robot with a trained policy, over Bluetooth, and (optionally) the simulation alongside it, live.

    mjpython run_real.py policy.npz --card "red 1131" --countdown 10 --with-sim   # real robot + live simulation
    python3  run_real.py policy.npz --card "red 1131"          # real robot only (run calibrate.py once first)
    python3  run_real.py policy.npz --sim                      # simulated motor only, to try it without the robot
    python3  run_real.py --fixed 60 --card "red 1131"          # no policy: both beams at 60% (a grip test)

(On a Mac the live simulation window needs mjpython, which comes with MuJoCo, instead of python3.)

Put the robot on the floor, pointing the way it should walk: that direction becomes "straight ahead" (heading 0).
Every 50 ms it reads the motor's sensors, turns them into the same 16 numbers the policy saw in training (robot_io.py),
asks the policy for two beam speeds, and sends them with movement_move_tank. Ctrl-C stops the motors at once.

--with-sim runs the same policy at the same moment on the simulated robot (sim_motor.py: MuJoCo with your CAD and your
motor's measured top speed), starting with its beams at the real beams' angles, in a 3D window that follows it.
Each second it prints the real and the simulated heading side by side.

Every step of the real robot is saved to real_run-<time>.csv next to this file.
"""
import argparse
import csv
import math
import os
import sys
import time
from pathlib import Path

import numpy as np

import robot_io
import silly_robot

HERE = Path(__file__).resolve().parent


class Walker:
    """One robot (real or simulated) being walked by the policy: reads its sensors, sends its beam speeds."""

    def __init__(self, hub, cal, percent_per_action):
        self.hub, self.cal, self.percent_per_action = hub, cal, percent_per_action

    def begin(self):
        self.hub.imu_reset_yaw_axis(0)               # heading 0 = the way it points now
        self.last_action = np.zeros(2)
        self.prev = robot_io.read_sensors(self.hub, self.cal)
        self.prev_t = time.monotonic()

    def step(self, policy, now):
        s = robot_io.read_sensors(self.hub, self.cal)
        beam_speed = (s["beam_position"] - self.prev["beam_position"]) / max(now - self.prev_t, 1e-3)
        obs = robot_io.observation(s["gravity"], s["gyro"], s["yaw"], s["beam_angle"], beam_speed, self.last_action)
        action = policy(obs)
        command = [int(max(-100, min(100, round(a * self.percent_per_action)))) for a in action]  # same rad/s as trained
        self.hub.movement_move_tank(command[0], command[1], blocking=False)
        self.last_action, self.prev, self.prev_t = action, s, now
        return {"heading_deg": round(math.degrees(s["yaw"]), 1),
                "left_speed": round(beam_speed[0], 2), "right_speed": round(beam_speed[1], 2),
                "left_cmd": command[0], "right_cmd": command[1],
                **{f"obs{i}": round(float(v), 4) for i, v in enumerate(obs)}}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("policy", nargs="?", help="policy.npz (from make_video.py) or a model_<n>.pt checkpoint")
    parser.add_argument("--fixed", type=int, metavar="PERCENT",
                        help="no policy: both beams at this fixed speed (a grip test, to compare with the simulation)")
    parser.add_argument("--seconds", type=float, default=10.0, help="how long to walk (default 10)")
    parser.add_argument("--sim", action="store_true", help="use the simulated Double Motor instead of the real one")
    parser.add_argument("--with-sim", action="store_true", help="also walk the simulated robot, live, in a 3D window")
    parser.add_argument("--card", help='the motor\'s Connection Card, e.g. --card "red 1131"')
    parser.add_argument("--countdown", type=int, metavar="SECONDS",
                        help="start after a spoken countdown instead of waiting for Enter")
    args = parser.parse_args()

    # ---- the policy, and how its outputs become motor percentages -------------------------------------------
    if args.fixed is not None:
        policy, trained_speed = (lambda obs: np.full(2, args.fixed / 100)), None
    elif args.policy:
        policy = robot_io.Policy(args.policy)
        trained_speed = policy.max_speed or silly_robot.MAX_SPEED
    else:
        parser.error("give a policy file (or --fixed PERCENT)")

    def percent_per_action(cal):
        # The policy asks for beam speeds as a fraction of the top speed it was trained with. Send the percentage
        # that gives the same rad/s on this motor, so the beams turn exactly as fast as in the simulation.
        return 100.0 if trained_speed is None else 100 * trained_speed / cal["max_speed_rad_s"]

    # ---- the robot(s) -----------------------------------------------------------------------------------------
    viewer = None
    if args.sim:
        from sim_motor import SimDoubleMotor
        hub = SimDoubleMotor()
        hub.connect()
        cal = hub.calibration()
    else:
        cal = robot_io.load_calibration(HERE / "real_robot.json")
    if trained_speed is not None:
        print(f"policy trained with a top speed of {trained_speed:.1f} rad/s; this motor does "
              f"{cal['max_speed_rad_s']:.1f} at 100%, so full speed is sent as {min(percent_per_action(cal), 100):.0f}%")
    twin = None
    if args.with_sim:
        from sim_motor import SimDoubleMotor
        twin = SimDoubleMotor(max_speed=cal["max_speed_rad_s"])   # the simulated robot, with your motor's speed
        twin.connect()
        viewer = open_viewer(twin)                                  # before the real robot moves: fail early
    if not args.sim:
        hub = robot_io.connect(card=args.card)

    walkers = [Walker(hub, cal, percent_per_action(cal))]
    if twin:
        walkers.append(Walker(twin, twin.calibration(), percent_per_action(twin.calibration())))
    try:
        start(hub, args)
        if twin:     # the simulated robot starts like the real one: same beam angles, on the floor
            twin.place(robot_io.read_sensors(hub, cal)["beam_angle"])
            time.sleep(0.3)
        rows = walk(walkers, policy, args.seconds, viewer, show_position=args.sim)
    except KeyboardInterrupt:
        rows = None
        print("\nstopped")
    finally:
        for w in walkers:
            w.hub.movement_stop()
            w.hub.disconnect()
    save(rows)
    if viewer is not None:
        print("(the simulation window closes in 5 seconds)")
        time.sleep(5)
        viewer.close()


def open_viewer(twin):
    """A live 3D window that follows the simulated robot."""
    import mujoco
    import mujoco.viewer
    try:
        viewer = mujoco.viewer.launch_passive(twin.model, twin.data, show_left_ui=False, show_right_ui=False)
    except RuntimeError as e:
        sys.exit(f"Could not open the simulation window ({e}).\nOn a Mac, run it with mjpython instead of python3:\n"
                 f"  mjpython run_real.py {' '.join(sys.argv[1:])}")
    viewer.cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
    viewer.cam.trackbodyid = twin.model.body("motor").id
    viewer.cam.distance, viewer.cam.azimuth, viewer.cam.elevation = 0.45, 130, -20
    return viewer


def start(hub, args):
    """Wait until the robot is on the floor and hands are off."""
    if args.sim:
        hub.act_out("on_floor")
        time.sleep(1.0)                                  # let it settle on the floor
    elif args.countdown:
        say("Put the robot on the floor, pointing the way it should walk. Hands off.")
        time.sleep(max(0, args.countdown - 3))
        say("3. 2. 1. Go.")
    else:
        input("Put the robot on the floor, pointing the way it should walk. Hands off, then press Enter... ")


def walk(walkers, policy, seconds, viewer=None, show_position=False):
    period = 1.0 / silly_robot.CONTROL_HZ
    for w in walkers:
        w.begin()
    time.sleep(0.1)
    rows = []
    start_t = next_t = time.monotonic()
    twin = walkers[1].hub if len(walkers) > 1 else None
    header = "  second  heading (deg)  beam speeds (rad/s)  commands (%)"
    if twin:
        header += "   | simulation: heading   x (m)   y (m)"
    elif show_position:
        header += "     x (m)   y (m)"
    print(header)
    for step in range(int(seconds * silly_robot.CONTROL_HZ)):
        next_t += period
        time.sleep(max(0.0, next_t - time.monotonic()))
        now = time.monotonic()
        row = {"t": round(now - start_t, 3), **walkers[0].step(policy, now)}
        sim_row = walkers[1].step(policy, now) if twin else None
        if show_position:                         # only the simulation knows where the robot really is
            row["x"], row["y"] = (round(float(v), 3) for v in walkers[0].hub.data.body("motor").xpos[:2])
        if twin:
            row["sim_heading_deg"] = sim_row["heading_deg"]
            row["sim_x"], row["sim_y"] = (round(float(v), 3) for v in twin.data.body("motor").xpos[:2])
        if viewer is not None and viewer.is_running():
            with twin._lock:
                viewer.sync()
        rows.append(row)
        if (step + 1) % silly_robot.CONTROL_HZ == 0:      # once a second
            line = (f"  {row['t']:6.1f}  {row['heading_deg']:12.0f}  {row['left_speed']:8.1f} {row['right_speed']:8.1f}"
                    f"     {row['left_cmd']:5d} {row['right_cmd']:5d}")
            if twin:
                line += f"   | {row['sim_heading_deg']:18.0f}  {row['sim_x']:6.2f}  {row['sim_y']:6.2f}"
            elif show_position:
                line += f"    {row['x']:7.2f} {row['y']:7.2f}"
            print(line)
        if now - next_t > 3 * period:
            print("     (running late: is the Bluetooth connection weak?)")
            next_t = now
    return rows


def save(rows):
    if rows:
        out = HERE / time.strftime("real_run-%Y%m%d-%H%M%S.csv")
        with open(out, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        print(f"saved every step to {out.name}")


def say(text):
    """Print, and on a Mac also speak it (so you can follow along with your hands on the robot)."""
    print(text)
    if sys.platform == "darwin":
        os.system(f'say "{text}"')


if __name__ == "__main__":
    main()
