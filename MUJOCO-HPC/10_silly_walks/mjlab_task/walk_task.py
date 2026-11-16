"""mjlab task: teach the silly walker (LEGO Double Motor with two beams) to walk in a straight line down +x.

The robot starts at the origin pointing a little off to one side (random yaw), with its two beams at random angles.
It must walk forward while staying on the line y = 0 and keeping its heading along +x. The policy outputs one speed
for each beam.

The policy only sees what the real Double Motor can measure itself (so the same policy could later run on the real
robot): its tilt, its turning rate (gyro), its heading (yaw), each beam's angle and speed, and what it did last.
The critic, used only during training, also sees how fast the robot is moving and how far it is off the line.

Importing this file registers the task "Silly-Walk" (see train.py / play.py next to it).
Everything you will want to tune is in the constants block and the reward weights. The robot is in ../silly_robot.py.
"""
from __future__ import annotations

import math
import sys
from pathlib import Path
from typing import TYPE_CHECKING

import torch

# silly_robot.py is next to this file when run by runIt.py, one folder up in the repo
_HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(_HERE), str(_HERE.parent)]
import silly_robot  # noqa: E402

from mjlab.actuator import BuiltinPdActuatorCfg  # noqa: E402
from mjlab.entity import EntityArticulationInfoCfg, EntityCfg  # noqa: E402
from mjlab.envs import ManagerBasedRlEnvCfg  # noqa: E402
from mjlab.envs.mdp import (  # noqa: E402
  action_rate_l2,
  base_ang_vel,
  base_lin_vel,
  joint_vel_rel,
  last_action,
  projected_gravity,
  reset_joints_by_offset,
  reset_root_state_uniform,
  time_out,
)
from mjlab.envs.mdp.actions import JointVelocityActionCfg  # noqa: E402
from mjlab.managers.action_manager import ActionTermCfg  # noqa: E402
from mjlab.managers.event_manager import EventTermCfg  # noqa: E402
from mjlab.managers.observation_manager import ObservationGroupCfg, ObservationTermCfg  # noqa: E402
from mjlab.managers.reward_manager import RewardTermCfg  # noqa: E402
from mjlab.managers.scene_entity_config import SceneEntityCfg  # noqa: E402
from mjlab.managers.termination_manager import TerminationTermCfg  # noqa: E402
from mjlab.rl import RslRlModelCfg, RslRlOnPolicyRunnerCfg, RslRlPpoAlgorithmCfg  # noqa: E402
from mjlab.scene import SceneCfg  # noqa: E402
from mjlab.sim import MujocoCfg, SimulationCfg  # noqa: E402
from mjlab.tasks.registry import register_mjlab_task  # noqa: E402
from mjlab.terrains import TerrainEntityCfg  # noqa: E402
from mjlab.viewer import ViewerConfig  # noqa: E402

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv

# ---- constants you may want to change ----------------------------------------------------------------
LEFT, RIGHT = silly_robot.LEFT, silly_robot.RIGHT
TARGET_SPEED = 0.15         # m/s the robot is rewarded for walking forward (both beams at 8 rad/s by hand: ~0.13)
MAX_BEAM_SPEED = silly_robot.MAX_SPEED   # rad/s; a policy output of 1.0 means this beam speed (motor at 100%)
START_YAW = 0.4             # rad; the robot starts pointing up to this far off +x
MAX_OFF_LINE = 1.0          # m; the episode ends if the robot wanders this far from y = 0
EPISODE_S = 10.0
# -------------------------------------------------------------------------------------------------------

_ROBOT = SceneEntityCfg("robot")


def _robot_cfg() -> EntityCfg:
  return EntityCfg(
    spec_fn=lambda: silly_robot.make_spec(motors=False),   # mjlab adds its own motors (below)
    init_state=EntityCfg.InitialStateCfg(
      pos=(0.0, 0.0, silly_robot.beam_long_end() + 0.005), joint_pos={".*": 0.0}, joint_vel={".*": 0.0}
    ),
    articulation=EntityArticulationInfoCfg(
      actuators=(
        # stiffness 0 = no position target, so the motor only chases the speed we ask for (like the real motor)
        BuiltinPdActuatorCfg(
          target_names_expr=(LEFT, RIGHT),
          stiffness=0.0,
          damping=silly_robot.MOTOR_KV,
          effort_limit=silly_robot.STALL_TORQUE,
          armature=silly_robot.GEARBOX_INERTIA,
        ),
      ),
    ),
  )


# ---- observations and rewards: how the robot is doing relative to "straight down +x" -------------------


def _yaw(env: ManagerBasedRlEnv) -> torch.Tensor:
  w, x, y, z = env.scene["robot"].data.root_link_quat_w.unbind(-1)
  return torch.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))


def _off_line(env: ManagerBasedRlEnv) -> torch.Tensor:
  """Sideways distance from this env's start line (its y axis), in metres."""
  robot = env.scene["robot"]
  return robot.data.root_link_pos_w[:, 1] - env.scene.env_origins[:, 1]


def heading(env: ManagerBasedRlEnv) -> torch.Tensor:
  """[sin(yaw), cos(yaw)]: which way the robot faces. The real Double Motor measures this (its Yaw). Shape [num_envs, 2]."""
  yaw = _yaw(env)
  return torch.stack([torch.sin(yaw), torch.cos(yaw)], dim=-1)


def off_line(env: ManagerBasedRlEnv) -> torch.Tensor:
  """Sideways distance from the line, for the critic only (the real robot cannot measure it). Shape [num_envs, 1]."""
  return _off_line(env).unsqueeze(-1)


def beam_angles(env: ManagerBasedRlEnv) -> torch.Tensor:
  """[sin, cos] of each beam's angle (the real motor measures these as its Position). Shape [num_envs, 4]."""
  q = env.scene["robot"].data.joint_pos
  return torch.cat([torch.sin(q), torch.cos(q)], dim=-1)


def forward_speed(env: ManagerBasedRlEnv) -> torch.Tensor:
  """Reward for going along +x at TARGET_SPEED: 0 when standing still, 1 at the target, less beyond it, negative backwards.

  It slopes everywhere (a narrow bump around the target would give a robot that starts out still nothing to learn from).
  """
  vx = env.scene["robot"].data.root_link_lin_vel_w[:, 0]
  return torch.clamp(1.0 - torch.square((vx - TARGET_SPEED) / TARGET_SPEED), min=-1.0)


def stay_on_line(env: ManagerBasedRlEnv) -> torch.Tensor:
  return -torch.square(_off_line(env) / 0.1)    # 10 cm off the line costs as much as standing still


def face_forward(env: ManagerBasedRlEnv) -> torch.Tensor:
  return -torch.square(_yaw(env))


def too_far_off_line(env: ManagerBasedRlEnv) -> torch.Tensor:
  return _off_line(env).abs() > MAX_OFF_LINE


# ---- environment ----------------------------------------------------------------------------------------


def silly_walk_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  # What the policy sees: only things the real Double Motor measures.
  actor_terms = {
    "tilt": ObservationTermCfg(func=projected_gravity),    # Pitch and Roll
    "gyro": ObservationTermCfg(func=base_ang_vel),         # GyroX/Y/Z
    "heading": ObservationTermCfg(func=heading),           # Yaw
    "beam_angle": ObservationTermCfg(func=beam_angles),    # LeftPosition, RightPosition
    "beam_speed": ObservationTermCfg(func=joint_vel_rel),  # LeftSpeed, RightSpeed
    "last_action": ObservationTermCfg(func=last_action),
  }
  # The critic (training only) also gets to see how fast it moves and where it is.
  critic_terms = {
    **actor_terms,
    "lin_vel": ObservationTermCfg(func=base_lin_vel),
    "off_line": ObservationTermCfg(func=off_line),
  }
  observations = {
    "actor": ObservationGroupCfg(actor_terms, enable_corruption=not play),
    "critic": ObservationGroupCfg(critic_terms),
  }

  actions: dict[str, ActionTermCfg] = {
    "beams": JointVelocityActionCfg(
      entity_name="robot", actuator_names=(LEFT, RIGHT), scale=MAX_BEAM_SPEED, use_default_offset=True
    ),
  }

  events = {
    "reset_pose": EventTermCfg(
      func=reset_root_state_uniform,
      mode="reset",
      params={"pose_range": {"yaw": (-START_YAW, START_YAW)}, "velocity_range": {}},
    ),
    # start each beam at a random angle, so the policy learns to walk from any position
    "reset_beams": EventTermCfg(
      func=reset_joints_by_offset,
      mode="reset",
      params={"position_range": (-math.pi, math.pi), "velocity_range": (0.0, 0.0)},
    ),
  }

  rewards = {
    "forward_speed": RewardTermCfg(func=forward_speed, weight=1.0),
    "stay_on_line": RewardTermCfg(func=stay_on_line, weight=0.5),
    "face_forward": RewardTermCfg(func=face_forward, weight=0.5),
    "action_rate": RewardTermCfg(func=action_rate_l2, weight=-0.02),
  }

  # No "tipped over" ending: rocking and tumbling is how this robot walks.
  terminations = {
    "time_out": TerminationTermCfg(func=time_out, time_out=True),
    "off_line": TerminationTermCfg(func=too_far_off_line),
  }

  return ManagerBasedRlEnvCfg(
    scene=SceneCfg(
      terrain=TerrainEntityCfg(terrain_type="plane"),
      entities={"robot": _robot_cfg()},
      num_envs=1,
      env_spacing=3.0,
    ),
    observations=observations,
    actions=actions,
    events=events,
    rewards=rewards,
    terminations=terminations,
    viewer=ViewerConfig(
      origin_type=ViewerConfig.OriginType.ASSET_BODY,
      entity_name="robot",
      body_name="motor",
      distance=0.6,
      elevation=-20.0,
      azimuth=130.0,
    ),
    # small, light parts need small physics steps: 0.002 s, and the policy acts every 10 steps (50 times a second)
    sim=SimulationCfg(mujoco=MujocoCfg(timestep=0.002, integrator="implicitfast")),
    decimation=10,
    episode_length_s=1e10 if play else EPISODE_S,
  )


def silly_walk_ppo_runner_cfg() -> RslRlOnPolicyRunnerCfg:
  return RslRlOnPolicyRunnerCfg(
    actor=RslRlModelCfg(
      hidden_dims=(128, 128),
      activation="elu",
      obs_normalization=True,
      distribution_cfg={"class_name": "GaussianDistribution", "init_std": 1.0, "std_type": "scalar"},
    ),
    critic=RslRlModelCfg(hidden_dims=(128, 128), activation="elu", obs_normalization=True),
    algorithm=RslRlPpoAlgorithmCfg(
      value_loss_coef=1.0,
      use_clipped_value_loss=True,
      clip_param=0.2,
      entropy_coef=0.005,
      num_learning_epochs=5,
      num_mini_batches=4,
      learning_rate=1.0e-3,
      schedule="adaptive",
      gamma=0.99,
      lam=0.95,
      desired_kl=0.01,
      max_grad_norm=1.0,
    ),
    experiment_name="silly_walk",
    logger="tensorboard",           # the default, wandb, needs an account and a login
    save_interval=50,
    num_steps_per_env=24,
    max_iterations=1000,
  )


register_mjlab_task(
  task_id="Silly-Walk",
  env_cfg=silly_walk_env_cfg(),
  play_env_cfg=silly_walk_env_cfg(play=True),
  rl_cfg=silly_walk_ppo_runner_cfg(),
)
