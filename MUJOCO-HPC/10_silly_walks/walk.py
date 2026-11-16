"""Film the silly walker in MuJoCo: before training (fixed beam speeds) or after (the trained mjlab policy).

    python3 10_silly_walks/walk.py                          # before: both beams at the same fixed speed
    python3 10_silly_walks/walk.py --policy policy.npz      # after: the trained policy steers it in a straight line

(run from the mujoco-hpc folder; policy.npz comes from make_video.py, or use a model_<n>.pt checkpoint if PyTorch is
installed). It prints where the robot is each second: walking straight keeps y (sideways) and the heading near 0.

Without a policy, the beams just turn at the same speed: the robot does go forward, but it drifts off to one side,
because nothing corrects it. The policy watches the motor's sensors 20 times a second (the same readings it gets in
training and on the real robot) and changes the two beam speeds to keep it on the line.
"""
import argparse
import math
import os
import sys
from pathlib import Path

# Offscreen rendering on the cluster (Linux has no screen). Must be set before importing mujoco.
if sys.platform.startswith("linux"):
    os.environ.setdefault("MUJOCO_GL", "egl")

import imageio
import mujoco
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import robot_io  # noqa: E402
import silly_robot  # noqa: E402

HERE = Path(__file__).resolve().parent
SECONDS = 10.0
FIXED_SPEED = 0.6                    # without a policy: both beams at 60% of top speed
LEFT_START, RIGHT_START = 0.0, math.pi / 2   # rad: where each beam points at the start (0 = long end down)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--policy", help="policy.npz or model_<n>.pt: let the trained policy walk")
    parser.add_argument("--seconds", type=float, default=SECONDS)
    args = parser.parse_args()
    policy = robot_io.Policy(args.policy) if args.policy else None
    out = HERE / ("walk_policy.mp4" if policy else "walk.mp4")
    # A policy output of 1.0 means the top speed it was trained with (the simulated motor can go that fast).
    top_speed = (policy.max_speed if policy and policy.max_speed else silly_robot.MAX_SPEED)

    # ---- the robot, with an accelerometer where the training task has one ----------------------------------
    spec = silly_robot.add_scene(silly_robot.make_spec())
    spec.add_sensor(name="acc", type=mujoco.mjtSensor.mjSENS_ACCELEROMETER, objtype=mujoco.mjtObj.mjOBJ_SITE,
                    objname="imu")
    model = spec.compile()
    data = mujoco.MjData(model)
    data.joint(silly_robot.LEFT).qpos = LEFT_START
    data.joint(silly_robot.RIGHT).qpos = RIGHT_START
    mujoco.mj_forward(model, data)
    actuators = [model.actuator(silly_robot.LEFT).id, model.actuator(silly_robot.RIGHT).id]
    joints = [data.joint(silly_robot.LEFT), data.joint(silly_robot.RIGHT)]

    def heading():
        w, x, y, z = data.body("motor").xquat
        return math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))

    def observe(last_action):
        """The policy's inputs, the way the training task computes them (mjlab_task/walk_task.py)."""
        acc = data.sensor("acc").data
        return robot_io.observation(
            gravity=-acc / max(np.linalg.norm(acc), 1e-6),            # tilt, as the accelerometer feels it
            gyro=data.qvel[3:6],                                        # turning rate, in the robot's own axes
            yaw=heading(),
            beam_angle=np.array([j.qpos[0] for j in joints]),
            beam_speed=np.array([j.qvel[0] for j in joints]),
            last_action=last_action,
        )

    # ---- simulate and film -------------------------------------------------------------------------------
    renderer = mujoco.Renderer(model, 480, 640)
    camera = mujoco.MjvCamera()
    camera.type = mujoco.mjtCamera.mjCAMERA_TRACKING       # follow the robot
    camera.trackbodyid = model.body("motor").id
    camera.distance, camera.azimuth, camera.elevation = 0.45, 130, -20

    FPS = 30
    decide_every = 1.0 / silly_robot.CONTROL_HZ
    frames, action, next_decision = [], np.zeros(2), 0.0
    print(("trained policy" if policy else f"no policy: both beams at {FIXED_SPEED:.0%}") + f", {args.seconds:g} s")
    print("  second    x (m)    y (m)   heading (deg)")
    while data.time < args.seconds:
        while data.time < len(frames) / FPS:
            if data.time >= next_decision:                  # 20 times a second, like on the real robot
                action = policy(observe(action)) if policy else np.full(2, FIXED_SPEED)
                data.ctrl[actuators] = action * top_speed               # rad/s, as in training
                next_decision += decide_every
            mujoco.mj_step(model, data)
        renderer.update_scene(data, camera=camera)
        frames.append(renderer.render())
        if len(frames) % FPS == 0:
            x, y = data.body("motor").xpos[:2]
            print(f"  {data.time:6.1f}  {x:7.2f}  {y:7.2f}  {math.degrees(heading()):12.0f}")

    renderer.close()
    imageio.mimwrite(out, frames, fps=FPS)
    print(f"wrote {out}: {len(frames)} frames")
    if sys.platform == "darwin":       # VS Code can't play .mp4 videos, so on a Mac open it in QuickTime
        os.system(f'open "{out}"')


if __name__ == "__main__":
    main()
