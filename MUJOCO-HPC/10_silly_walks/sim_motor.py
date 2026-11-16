"""A simulated LEGO Double Motor, for trying calibrate.py and run_real.py without the real robot (their --sim option).

It answers the same commands as the real one in the legoeducation library (connect, movement_move_tank,
movement_stop, imu_reset_yaw_axis, disconnect) and fills in the same readings (motor[0], motor[1], imu_device),
while the MuJoCo robot from silly_robot.py runs in real time underneath.

Its raw readings are deliberately awkward, the way a real motor's might be: the sensor chip is turned 90 degrees inside
the motor, yaw counts the other way, the left output counts backwards, the beams' zero is somewhere random, and its top
speed is not the one in silly_robot.py. calibrate.py has to work all of that out, just as it does on the real robot.
"""
import math
import threading
import time

import mujoco
import numpy as np

import silly_robot

TRUE_MAX_SPEED = 13.0              # rad/s at 100%: what calibrate.py should measure
NOTIFY_S = 0.025                   # new readings every 25 ms, like the real motor's fastest Bluetooth setting
POSITION_SIGN = (-1, 1)            # the left output faces the other way, so it counts backwards
ABSOLUTE_OFFSET_DEG = (123, 40)    # where each output's angle sensor reads 0 (random-looking on purpose)
# The sensor chip axes (one per row), written in robot axes (x forward, y left, z up): chip x = robot left,
# chip y = robot back, chip z = robot up.
CHIP_AXES = np.array([[0, 1, 0], [-1, 0, 0], [0, 0, 1]], dtype=float)
YAW_UNIT = -0.1                    # degrees per yaw count; negative: turning left makes it go down
GYRO_UNIT = 1.0                    # degrees/s per gyro count
ACCEL_UNIT = 9.81 / 1000           # m/s^2 per count (milli-g)


class _Reading:
    pass


class SimDoubleMotor:
    """Looks like legoeducation.DoubleMotor to calibrate.py and run_real.py."""

    def __init__(self, max_speed=TRUE_MAX_SPEED):
        self.max_speed = max_speed                 # rad/s at 100%
        spec = silly_robot.add_scene(silly_robot.make_spec())
        for act in spec.actuators:                 # the simulated motor's own top speed
            act.ctrlrange = [-max_speed, max_speed]
        motor = spec.body("motor")
        chip = motor.add_site(name="chip")
        rot = np.zeros(4)
        mujoco.mju_mat2Quat(rot, CHIP_AXES.T.flatten())   # rows of CHIP_AXES = chip axes in robot axes
        chip.quat = rot
        spec.add_sensor(name="acc", type=mujoco.mjtSensor.mjSENS_ACCELEROMETER, objtype=mujoco.mjtObj.mjOBJ_SITE, objname="chip")
        spec.add_sensor(name="gyro", type=mujoco.mjtSensor.mjSENS_GYRO, objtype=mujoco.mjtObj.mjOBJ_SITE, objname="chip")
        # An invisible hand that can hold the robot (switched on and off), for the moves calibrate.py asks for.
        spec.worldbody.add_body(name="hand", mocap=True, pos=motor.pos)
        spec.add_equality(name="hold", type=mujoco.mjtEq.mjEQ_WELD, objtype=mujoco.mjtObj.mjOBJ_BODY,
                          name1="hand", name2="motor", active=False)
        self.model = spec.compile()
        self.data = mujoco.MjData(self.model)
        self.left = self.model.actuator(silly_robot.LEFT).id
        self.right = self.model.actuator(silly_robot.RIGHT).id
        self.joints = [self.data.joint(silly_robot.LEFT), self.data.joint(silly_robot.RIGHT)]
        self.motor = [_Reading(), _Reading()]
        self.imu_device = _Reading()
        self.connected = False
        self._lock = threading.Lock()
        self._held = None            # None = on the floor; else (height, pitch, yaw rate) held in a hand
        self._yaw_hold = 0.0
        self._yaw_zero = 0.0
        self._running = False

    # ---- the legoeducation commands ------------------------------------------------------------------
    def connect(self, *args, device_notification_delay=100, **kwargs):
        mujoco.mj_forward(self.model, self.data)
        self._notify()                    # like the real one, connect() returns once the first readings are in
        self._running = True
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        time.sleep(2 * NOTIFY_S)
        self.connected = True

    def disconnect(self):
        self._running = False
        self.connected = False

    def movement_move_tank(self, speed_left, speed_right, *, blocking=True):
        with self._lock:
            self.data.ctrl[self.left] = max(-100, min(100, speed_left)) / 100 * self.max_speed
            self.data.ctrl[self.right] = max(-100, min(100, speed_right)) / 100 * self.max_speed

    def movement_stop(self, *, blocking=True):
        self.movement_move_tank(0, 0)

    def imu_reset_yaw_axis(self, value=0, *, blocking=True):
        with self._lock:
            self._yaw_zero = self._world_yaw() - math.radians(value * YAW_UNIT)

    def calibration(self) -> dict:
        """What calibrate.py would measure on this simulated motor, exactly (its true settings)."""
        return {
            "max_speed_rad_s": self.max_speed,
            "motor_long_end": silly_robot.MOTOR_LONG_END,
            "position_sign": list(POSITION_SIGN),
            "absolute_sign": list(POSITION_SIGN),
            "beam_zero_deg": list(ABSOLUTE_OFFSET_DEG),
            "chip_axes": CHIP_AXES.tolist(),
            "yaw_rad_per_count": math.radians(YAW_UNIT),
            "gyro_rad_s_per_count": math.radians(GYRO_UNIT),
        }

    def place(self, beam_angles=(0.0, 0.0)):
        """Put the robot back at the start, on the floor facing +x, with its beams at these angles (rad)."""
        with self._lock:
            self._held = None
            self.data.qpos[:7] = [0, 0, silly_robot.beam_long_end() + 0.002, 1, 0, 0, 0]
            self.data.qvel[:] = 0
            for joint, angle in zip(self.joints, beam_angles):
                joint.qpos = angle
            mujoco.mj_forward(self.model, self.data)

    # ---- acting out what calibrate.py asks a person to do ------------------------------------------------
    def act_out(self, step):
        """Do with the simulated robot what calibrate.py asks you to do with the real one."""
        lift = silly_robot.beam_long_end() + 0.05
        with self._lock:
            if step == "hold_level":
                self._held = (lift, 0.0, 0.0)
            elif step == "nose_down":
                self._held = (lift, math.radians(60), 0.0)
            elif step == "beams_down":
                self._held = (lift, 0.0, 0.0)
                for j in self.joints:
                    j.qpos = 0.0
                    j.qvel = 0.0
            elif step == "turn_left":            # a quarter turn to the left, over 2 seconds
                self._held = (silly_robot.beam_long_end() + 0.005, 0.0, math.radians(45))
            elif step == "stop_turning":
                self._held = (silly_robot.beam_long_end() + 0.005, 0.0, 0.0)
            elif step == "on_floor":
                self._held = None
            else:
                raise ValueError(step)

    # ---- inside ----------------------------------------------------------------------------------------
    def _world_yaw(self):
        w, x, y, z = self.data.qpos[3:7]
        return math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))

    def _run(self):
        start = time.monotonic()
        next_notify = 0.0
        while self._running:
            with self._lock:
                while self.data.time < time.monotonic() - start:
                    if self._held is not None:       # the hand holds it: at this height, tipped, turning
                        height, pitch, yaw_rate = self._held
                        self._yaw_hold += yaw_rate * self.model.opt.timestep
                        turn, tip = np.zeros(4), np.zeros(4)
                        mujoco.mju_axisAngle2Quat(turn, [0, 0, 1], self._yaw_hold)
                        mujoco.mju_axisAngle2Quat(tip, [0, 1, 0], pitch)       # + = front down
                        mujoco.mju_mulQuat(self.data.mocap_quat[0], turn, tip)
                        self.data.mocap_pos[0] = [*self.data.qpos[:2], height]
                        self.data.eq_active[0] = 1
                    else:
                        self.data.eq_active[0] = 0
                        self._yaw_hold = self._world_yaw()
                    mujoco.mj_step(self.model, self.data)
                if self.data.time >= next_notify:
                    self._notify()
                    next_notify = self.data.time + NOTIFY_S
            time.sleep(0.001)

    def _notify(self):
        """Fill in the readings the way the real motor's Bluetooth notifications do (raw whole numbers)."""
        for i, (joint, sign, offset) in enumerate(zip(self.joints, POSITION_SIGN, ABSOLUTE_OFFSET_DEG)):
            m = _Reading()
            m.position = int(round(sign * math.degrees(joint.qpos[0])))
            m.absolutePosition = (m.position + offset) % 360
            m.speed = int(round(sign * joint.qvel[0] / self.max_speed * 100))
            self.motor[i] = m
        acc = self.data.sensor("acc").data / ACCEL_UNIT
        gyro = np.degrees(self.data.sensor("gyro").data) / GYRO_UNIT
        yaw = math.degrees(math.atan2(math.sin(self._world_yaw() - self._yaw_zero),
                                      math.cos(self._world_yaw() - self._yaw_zero))) / YAW_UNIT
        r = _Reading()
        r.yaw = int(round(yaw))
        r.accelerometerX, r.accelerometerY, r.accelerometerZ = (int(round(a)) for a in acc)
        r.gyroscopeX, r.gyroscopeY, r.gyroscopeZ = (int(round(g)) for g in gyro)
        self.imu_device = r
