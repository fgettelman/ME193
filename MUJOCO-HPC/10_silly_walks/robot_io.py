"""The link between the real Double Motor and the trained policy. Used by calibrate.py and run_real.py.

  * read_sensors(): turns the motor's raw readings into the same units and directions the simulation uses
    (radians, rad/s, robot axes x forward / y left / z up), with the numbers calibrate.py measured.
  * observation(): puts them in the same order the policy saw them in training (mjlab_task/walk_task.py).
  * Policy: the trained network, run with numpy only (no mjlab, no GPU, no PyTorch needed on your laptop).

Run it on its own to turn a checkpoint into a small policy file:  python3 robot_io.py model_999.pt  ->  policy.npz
(needs PyTorch; make_video.py also writes policy.npz on the cluster, so usually you do not need this).
"""
import json
import math
import sys
from pathlib import Path

import numpy as np

CALIBRATION_FILE = Path(__file__).resolve().parent / "real_robot.json"
NUM_OBS = 16   # tilt 3, gyro 3, heading 2, beam angles 4, beam speeds 2, last action 2


# ---- connecting ------------------------------------------------------------------------------------------

def connect(sim=False, card=None, notify_ms=25):
    """The Double Motor, connected: the simulated one (sim=True) or the real one over Bluetooth.

    card: the Connection Card on the motor, as "red 1131" (colour and number), so it connects to your motor and not
    a neighbour's. Without it, it connects to the first Double Motor it finds.
    """
    if sim:
        from sim_motor import SimDoubleMotor
        hub = SimDoubleMotor()
        hub.connect(device_notification_delay=notify_ms)
        return hub
    import legoeducation as le
    hub = le.DoubleMotor()
    options = {}
    if card:
        colour, _, serial = card.strip().partition(" ")
        options = {"card_color": getattr(le, f"LEGO_COLOR_{colour.upper()}"), "card_serial": serial.strip()}
    print("Looking for the Double Motor over Bluetooth" + (f" with card {card}" if card else "") + "...")
    hub.connect(device_notification_delay=notify_ms, **options)
    if not hub.connected:
        sys.exit("Could not find the Double Motor. Is it on (light blinking), close to the laptop, and not connected to\n"
                 "anything else (the LEGO Education app or a web page)? Check the card's colour and number.")
    print("connected")
    return hub


# ---- raw readings -> simulation units ----------------------------------------------------------------------

def load_calibration(path=CALIBRATION_FILE) -> dict:
    path = Path(path)
    if not path.exists():
        sys.exit(f"No {path.name} yet: run  python3 calibrate.py  first (it measures your robot).")
    return json.loads(path.read_text())


def read_sensors(hub, cal: dict) -> dict:
    """The motor's latest readings, in simulation units. Beam angle 0 = long end down, positive = forward."""
    imu = hub.imu_device
    chip_to_robot = np.array(cal["chip_axes"])                    # rows: chip x, y, z written in robot axes
    acc_chip = np.array([imu.accelerometerX, imu.accelerometerY, imu.accelerometerZ], dtype=float)
    gyro_chip = np.array([imu.gyroscopeX, imu.gyroscopeY, imu.gyroscopeZ], dtype=float)
    acc = chip_to_robot.T @ acc_chip                              # same reading, in robot axes
    gyro = chip_to_robot.T @ gyro_chip * cal["gyro_rad_s_per_count"]
    gravity = -acc / max(np.linalg.norm(acc), 1e-9)               # an accelerometer feels "up" when gravity pulls down
    yaw = imu.yaw * cal["yaw_rad_per_count"]
    angles, positions = [], []
    for i in range(2):
        m = hub.motor[i]
        angles.append(math.radians(cal["absolute_sign"][i] * (m.absolutePosition - cal["beam_zero_deg"][i])))
        positions.append(math.radians(cal["position_sign"][i] * m.position))   # keeps counting past a turn: speeds
    return {"gravity": gravity, "gyro": gyro, "yaw": yaw, "beam_angle": np.array(angles),
            "beam_position": np.array(positions)}


def observation(gravity, gyro, yaw, beam_angle, beam_speed, last_action) -> np.ndarray:
    """The policy's 16 inputs, in training order (actor_terms in walk_task.py)."""
    return np.concatenate([
        gravity,                                       # tilt
        gyro,                                          # gyro
        [math.sin(yaw), math.cos(yaw)],                # heading
        np.sin(beam_angle), np.cos(beam_angle),        # beam_angle: sin left, sin right, cos left, cos right
        beam_speed,                                    # beam_speed (rad/s)
        last_action,                                   # last_action
    ]).astype(np.float32)


# ---- the trained policy -----------------------------------------------------------------------------------

class Policy:
    """The trained actor network: inputs scaled the way training scaled them, two ELU layers, two outputs."""

    def __init__(self, path):
        path = Path(path)
        if path.suffix == ".pt":
            weights = checkpoint_to_arrays(path)
        else:
            weights = dict(np.load(path))
        self.mean, self.std, self.eps = weights["mean"], weights["std"], float(weights["eps"])
        # The top speed the simulated motor had in training: a policy output of 1.0 means this many rad/s.
        self.max_speed = float(weights["max_speed"]) if "max_speed" in weights else None
        self.layers = [(weights[f"w{i}"], weights[f"b{i}"]) for i in range(int(weights["num_layers"]))]
        if self.mean.shape[-1] != NUM_OBS:
            sys.exit(f"{path.name} expects {self.mean.shape[-1]} inputs, but this robot gives {NUM_OBS}. "
                     "Was it trained with this version of walk_task.py?")

    def __call__(self, obs: np.ndarray) -> np.ndarray:
        x = (obs - self.mean) / (self.std + self.eps)
        for i, (w, b) in enumerate(self.layers):
            x = x @ w.T + b
            if i < len(self.layers) - 1:
                x = np.where(x > 0, x, np.expm1(np.minimum(x, 0)))   # ELU
        return np.clip(x.reshape(-1), -1.0, 1.0)                     # training capped actions at +/-1 (100%)


def checkpoint_to_arrays(path) -> dict:
    """Read the actor out of an mjlab/rsl_rl checkpoint (model_<n>.pt). Needs PyTorch."""
    import torch
    actor = torch.load(path, map_location="cpu", weights_only=False)["actor_state_dict"]
    linear = sorted({int(k.split(".")[1]) for k in actor if k.startswith("mlp.") and k.endswith(".weight")})
    arrays = {
        "mean": actor["obs_normalizer._mean"].numpy().reshape(-1),
        "std": actor["obs_normalizer._std"].numpy().reshape(-1),
        "eps": np.array(0.01),                         # rsl_rl's EmpiricalNormalization default
        "num_layers": np.array(len(linear)),
    }
    for i, n in enumerate(linear):
        arrays[f"w{i}"] = actor[f"mlp.{n}.weight"].numpy()
        arrays[f"b{i}"] = actor[f"mlp.{n}.bias"].numpy()
    return arrays


def export(checkpoint, out="policy.npz", max_speed=None):
    """Write policy.npz. max_speed: the motor top speed (rad/s) it was trained with; default: silly_robot's now."""
    if max_speed is None:
        import silly_robot
        max_speed = silly_robot.MAX_SPEED
    np.savez(out, max_speed=np.array(max_speed), **checkpoint_to_arrays(checkpoint))
    return out


if __name__ == "__main__":
    if len(sys.argv) not in (2, 3):
        sys.exit("usage: python3 robot_io.py model_999.pt [top speed it was trained with, rad/s]\n"
                 "       (leave the speed out if real_robot.json has not changed since training)")
    print("wrote", export(sys.argv[1], max_speed=float(sys.argv[2]) if len(sys.argv) == 3 else None))
