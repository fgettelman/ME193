"""Make the silly walker walk with a fixed plan, and film it. Runs on your laptop or on the cluster.

    python3 walk.py                                                 (on your laptop, in this folder)
    python3 runIt.py --with 10_silly_walks/silly_robot.py 10_silly_walks/walk.py   (on the cluster, from mujoco-hpc)

Everything you may want to change is in the first block of constants. The robot itself is in silly_robot.py.
"""
import math
import os
import sys

# Offscreen rendering on the cluster (Linux has no screen). Must be set before importing mujoco.
if sys.platform.startswith("linux"):
    os.environ.setdefault("MUJOCO_GL", "egl")

import imageio
import mujoco

import silly_robot

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "walk.mp4")
LEFT_START, RIGHT_START = 0.0, math.pi / 2   # rad: where each beam points at the start (0 = long end down)

# What the beams do, in order: (until this second, left speed, right speed), in radians/second.
PLAN = [
    (1.0, 0, 0),       # stand still and settle
    (5.0, 8, 8),       # both beams forward at the same speed
    (8.0, 8, -8),      # opposite directions: turn on the spot
    (12.0, 8, 6),      # left faster
]

# ---- load the robot ----------------------------------------------------------------------------------
spec = silly_robot.add_scene(silly_robot.make_spec())
model = spec.compile()
data = mujoco.MjData(model)
data.joint(silly_robot.LEFT).qpos = LEFT_START
data.joint(silly_robot.RIGHT).qpos = RIGHT_START
mujoco.mj_forward(model, data)
left, right = model.actuator(silly_robot.LEFT).id, model.actuator(silly_robot.RIGHT).id


def heading_deg():
    w, x, y, z = data.body("motor").xquat
    return math.degrees(math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z)))


# ---- simulate and film -------------------------------------------------------------------------------
renderer = mujoco.Renderer(model, 480, 640)
camera = mujoco.MjvCamera()
camera.type = mujoco.mjtCamera.mjCAMERA_TRACKING       # follow the robot
camera.trackbodyid = model.body("motor").id
camera.distance, camera.azimuth, camera.elevation = 0.5, 130, -20

FPS = 30
frames = []
print("  second    x (m)    y (m)   heading (deg)")
for end, left_speed, right_speed in PLAN:
    while data.time < end:
        while data.time < len(frames) / FPS:
            data.ctrl[left], data.ctrl[right] = left_speed, right_speed
            mujoco.mj_step(model, data)
        renderer.update_scene(data, camera=camera)
        frames.append(renderer.render())
    x, y = data.body("motor").xpos[:2]
    print(f"  {data.time:6.1f}  {x:7.2f}  {y:7.2f}  {heading_deg():12.0f}")

renderer.close()
imageio.mimwrite(OUT, frames, fps=FPS)
print(f"wrote {OUT}: {len(frames)} frames")

if sys.platform == "darwin":       # VS Code can't play .mp4 videos, so on a Mac open it in QuickTime
    os.system(f'open "{OUT}"')
