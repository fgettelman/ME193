"""Record a video of a trained Silly-Walk policy. Made for the cluster, which has no screen.

    python3 runIt.py --gpu --time 00:10:00 --mem 16G --cpus 4 \
      --with 10_silly_walks/mjlab_task/walk_task.py --with 10_silly_walks/silly_robot.py \
      --with runs/<your run>/logs/rsl_rl/silly_walk/<date folder>/model_999.pt \
      10_silly_walks/mjlab_task/make_video.py model_999.pt

runIt.py brings the .mp4 home to ./runs/ and opens it. (mjlab's own play.py also records a video, but then opens a
web viewer that never exits on a machine without a screen.)

It also writes policy.npz: the trained policy in a small file that run_real.py uses to walk the real robot.
"""
import math
import sys
from dataclasses import asdict
from pathlib import Path

import torch

import walk_task  # noqa: F401  (registers "Silly-Walk"; also puts silly_robot.py and robot_io.py on the path)
import robot_io
import silly_robot
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls
from mjlab.utils.torch import configure_torch_backends
from mjlab.utils.wrappers import VideoRecorder

TASK = "Silly-Walk"
HZ = silly_robot.CONTROL_HZ
STEPS = 10 * HZ      # 10 seconds of walking

if len(sys.argv) != 2:
  sys.exit("usage: make_video.py model_999.pt")
checkpoint = Path(sys.argv[1])
print("wrote", robot_io.export(checkpoint, "policy.npz"), "(for run_real.py)")

configure_torch_backends()
device = "cuda:0" if torch.cuda.is_available() else "cpu"
env_cfg = load_env_cfg(TASK, play=True)
env_cfg.scene.num_envs = 1
env_cfg.viewer.width, env_cfg.viewer.height = 640, 480
agent_cfg = load_rl_cfg(TASK)

env = ManagerBasedRlEnv(cfg=env_cfg, device=device, render_mode="rgb_array")
env = VideoRecorder(env, video_folder=".", step_trigger=lambda step: step == 0, video_length=STEPS, disable_logger=True)
env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

runner = (load_runner_cls(TASK) or MjlabOnPolicyRunner)(env, asdict(agent_cfg), device=device)
runner.load(str(checkpoint), load_cfg={"actor": True}, strict=True, map_location=device)
policy = runner.get_inference_policy(device=device)

robot = env.unwrapped.scene["robot"]
print("  second   x (m)   y (m)   heading (deg)   speed (m/s)")
obs = env.get_observations()
with torch.no_grad():
  for step in range(STEPS):
    obs = env.step(policy(obs))[0]
    if step % HZ == HZ - 1:   # once a second
      x, y = robot.data.root_link_pos_w[0, :2].tolist()
      w, qx, qy, qz = robot.data.root_link_quat_w[0].tolist()
      yaw = math.atan2(2 * (w * qz + qx * qy), 1 - 2 * (qy * qy + qz * qz))
      vx = robot.data.root_link_lin_vel_w[0, 0].item()
      print(f"  {(step + 1) / HZ:5.0f}  {x:7.2f} {y:7.2f}  {math.degrees(yaw):11.1f}  {vx:12.2f}")
env.close()
print("wrote the video (an .mp4 next to this script)")
