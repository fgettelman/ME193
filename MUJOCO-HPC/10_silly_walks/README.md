# Silly walks: a LEGO Double Motor that walks on two beams

Take the LEGO Education **Double Motor**, put a Technic **beam** on each output instead of a wheel, and it walks: the
beams swing round, push off the floor, and the motor rocks and tumbles along. This folder simulates that robot in
MuJoCo, trains a policy with **mjlab** to make it walk in a **straight line**, and runs that same policy on the
**real Double Motor** over Bluetooth. Everything is Python.

It uses the same cluster setup as the rest of `mujoco-hpc` (`python3 setup.py` all PASS; mjlab is already installed by it).

| File | What it is |
|---|---|
| `silly_robot.py` | **The robot**, built in Python from your CAD: motor mass, motor speed and torque, friction, which end of the motor leads. **Weigh your motor and change the numbers at the top.** Run it to write `silly_robot.xml` for MuJoCo's viewer. |
| `sillywalks.stl` | Your robot from Onshape (the whole robot, in millimetres). |
| `import_cad.py` | Splits `sillywalks.stl` into motor, beams, hubs and pins, and writes `silly_cad.npz`, the shapes the simulation uses. |
| `walk.py` | Walks the simulated robot with a fixed plan and films it. Runs on your laptop. |
| `calibrate.py` | **Measures your real robot** (sensor directions and units, motor directions and top speed, beam angles) and writes `real_robot.json`. |
| `run_real.py` | **Walks the real robot** with a trained policy, over Bluetooth. |
| `robot_io.py` | Turns the real motor's readings into the policy's inputs, and runs the policy with numpy. |
| `sim_motor.py` | A simulated Double Motor that answers the same commands as the real one: `--sim` for `calibrate.py` and `run_real.py`. |
| `mjlab_task/walk_task.py` | The mjlab task "Silly-Walk": rewards, resets, what the policy sees, how messy the simulation is. **Tune this one.** |
| `mjlab_task/train.py`, `play.py` | mjlab's own train/play commands, with our task registered first |
| `mjlab_task/make_video.py` | Records a movie of a trained policy on the cluster, and writes `policy.npz` for `run_real.py` |
| `mjlab_task/train.sbatch` | A Slurm job that trains on a GPU, for running by hand on the cluster |

All commands below are run from the **`mujoco-hpc` folder** (VPN on if off campus), except the real-robot ones, which
run from **this folder** (`cd 10_silly_walks`).

## The whole path, in short

1. `walk.py`: see the simulated robot walk (laptop).
2. `calibrate.py`: measure your real robot once (laptop + robot).
3. `train.py`: train on the cluster, with your robot's measurements.
4. `make_video.py`: watch the trained policy in simulation, and get `policy.npz` (cluster).
5. `run_real.py policy.npz --with-sim`: walk the real robot, with the simulation playing alongside.

Install on your laptop once: `pip install mujoco imageio imageio-ffmpeg legoeducation` (Python 3.11 or newer).

## 1. See the robot walk (on your laptop, 1 minute)

```
python3 10_silly_walks/walk.py
```
The video `10_silly_walks/walk.mp4` opens in QuickTime. This is **before training**: both beams at the same speed, no
steering. In 10 seconds it goes 0.44 m forward but drifts 0.49 m sideways. After training (step 4), the same command
with `--policy policy.npz` walks it straight and writes `walk_policy.mp4`.

## 2. Measure your real robot (on your laptop, 3 minutes)

Build the robot, turn the Double Motor on, and in this folder:
```
cd 10_silly_walks
python3 calibrate.py
```
It asks you to hold it level, spin the beams, tip it forward, turn it left a quarter turn, and point the beams down,
and writes `real_robot.json`. Run it again if you rebuild the robot or put a beam back at a different angle.
To see what it does without the robot, `python3 calibrate.py --sim` does the same on the simulated motor.

If it says the motor's firmware does not match, update the motor in the LEGO Education app (or
`pip install --upgrade legoeducation`).

## 3. Train (on the cluster)

```
python3 runIt.py --gpu --time 01:00:00 --mem 16G --cpus 4 \
  --with 10_silly_walks/mjlab_task/walk_task.py --with 10_silly_walks/silly_robot.py \
  --with 10_silly_walks/silly_cad.npz \
  --with 10_silly_walks/real_robot.json \
  10_silly_walks/mjlab_task/train.py Silly-Walk --env.scene.num-envs 4096
```
`--with 10_silly_walks/real_robot.json` makes the simulated motor as fast as yours; leave that line out if you have not
calibrated yet. First time? Check it starts with a short run: `--time 00:10:00` and
`--env.scene.num-envs 64 --agent.max-iterations 20` at the end. Under "job output" you should see iterations with a
mean reward.

It trains 1000 iterations by default (change with `--agent.max-iterations`). Checkpoints (`model_<n>.pt`, every 50
iterations) come back to `runs/train-<time>/logs/rsl_rl/silly_walk/<date>/`. The last one is `model_999.pt`.

## 4. Watch it in simulation, and get the policy file (on the cluster)

```
python3 runIt.py --gpu --time 00:10:00 --mem 16G --cpus 4 \
  --with 10_silly_walks/mjlab_task/walk_task.py --with 10_silly_walks/silly_robot.py \
  --with 10_silly_walks/silly_cad.npz \
  --with 10_silly_walks/robot_io.py --with 10_silly_walks/real_robot.json \
  --with runs/train-XXXXXXXX-XXXXXX/logs/rsl_rl/silly_walk/XXXX-XX-XX_XX-XX-XX/model_999.pt \
  10_silly_walks/mjlab_task/make_video.py model_999.pt
```
Use the same `real_robot.json` as in training (or leave that `--with` out in both). Do not type the `X`s: type
`runs/train-` and press Tab to complete each folder name. It prints where the robot is each
second (x forward, y sideways, heading; walking straight keeps y and heading near 0), and brings home the movie and
**`policy.npz`** in `runs/make_video-<time>/`.

Quicker, on your laptop: turn the checkpoint into `policy.npz` (needs PyTorch) and film it with `walk.py`:
```
python3 10_silly_walks/robot_io.py runs/train-XXXXXXXX-XXXXXX/logs/rsl_rl/silly_walk/XXXX-XX-XX_XX-XX-XX/model_999.pt
python3 10_silly_walks/walk.py --policy policy.npz
```

## 5. Walk the real robot, with the simulation playing alongside (on your laptop)

`policy.npz` is in this folder (copy in a new one after training). Turn the motor on (nothing else connected to it),
and in this folder:
```
cd 10_silly_walks
mjpython run_real.py policy.npz --card "red 1131" --countdown 10 --with-sim
```
A 3D window opens with the simulated robot. Your Mac says "Put the robot on the floor... Hands off", counts
"3, 2, 1, Go", and **the real robot and the simulated one walk at the same time**, with the same policy, the simulated
one starting with its beams at the real beams' angles. Every second it prints the real heading next to the simulated
one (and where the simulated robot is). It walks for 10 seconds (`--seconds 20` for longer); **Ctrl-C stops the
motors at once**. Every step is saved to `real_run-<time>.csv`.

- `mjpython` (it comes with MuJoCo) is needed on a Mac for the live window; without `--with-sim`, `python3` is fine.
- Use your own card's colour and number in `--card`; leave out `--countdown 10` to start with Enter instead.
- `python3 run_real.py policy.npz --sim` runs the same program on a simulated motor only (no robot needed).
- `python3 run_real.py --fixed 60 --card "red 1131"`: no policy, both beams at 60% (a grip test).

## How the real robot is made to match the simulation

A policy trained in a perfect simulation usually fails on a real robot. This is what keeps them the same:

| Real-life difference | What we do about it |
|---|---|
| The sensors read in their own units and directions | `calibrate.py` measures them; `robot_io.py` converts every reading into the simulation's units (radians, rad/s) and directions (x forward, y left, z up). |
| Your motor's top speed is not the estimate | `calibrate.py` measures it; `silly_robot.py` reads it from `real_robot.json`, so the simulated motor matches. A policy output of 1.0 is 100% on both. |
| Bluetooth is slow and late | The policy decides 20 times a second, the fastest the motor's readings arrive. In training, commands arrive up to 50 ms late and readings can be 50 ms old. |
| Real sensors are noisy, and the accelerometer feels every jolt | Training adds noise to every reading, and tilt comes from a simulated accelerometer (jolts included), not the perfect gravity direction. |
| Floor, weight and motor strength are never exactly what the model says | Each of the 4096 simulated robots gets its own beam and case friction, motor weight (+/-15%) and motor strength (+/-30%). |
| The real motor stops at 100% | Training caps the policy's outputs at +/-1. |

All of these numbers are at the top of `mjlab_task/walk_task.py`.

The policy only uses what the real Double Motor measures itself:

| Policy input | Real Double Motor reading (`legoeducation`) |
|---|---|
| tilt (which way is down) | `imu_device.accelerometerX/Y/Z` |
| turning rate | `imu_device.gyroscopeX/Y/Z` |
| heading | `imu_device.yaw` (0 = the way it pointed at the start) |
| each beam's angle | `motor[0/1].absolutePosition` |
| each beam's speed | change in `motor[0/1].position` every 50 ms |
| what it did last step | remembered by `run_real.py` |

It outputs one speed per beam, sent as `movement_move_tank(left %, right %)`. During training, the critic also sees the
robot's true speed and how far it is from the line, which the real robot cannot measure. That is allowed because the
critic is only used for training.

Rewards (in `walk_task.py`): walk forward at `TARGET_SPEED` (0.15 m/s), stay on the line y = 0, face along +x, and do
not change beam speeds too jerkily. Each robot starts pointing up to 0.4 rad off +x with its beams at random angles.
There is no "tipped over" ending, because rocking and tumbling is how this robot walks.

## Your CAD, and what is still an estimate (in `silly_robot.py`)

The shapes come from your Onshape export: the 80 x 53 x 30 mm Double Motor (its outputs 16 mm from one end) and a bent
liftarm on each output that reaches 33 mm from the axle. Changed the robot in Onshape? Export the whole robot as one
STL in millimetres, save it as `sillywalks.stl` here, and run `pip install trimesh` (once) and `python3 import_cad.py`.
Delete `silly_cad.npz` to go back to simple boxes.

`calibrate.py` measures the motor's speed and sensors. Already measured on our robot: `MOTOR_MASS` 90 g (the Double
Motor with battery), `BEAM_MASS` 2 g per beam (whole robot 94 g), and `MOTOR_LONG_END = "back"` (the 64 mm side
trails). Still estimates:

- `STALL_TORQUE`: how hard the motor pushes. If the simulated robot climbs over things the real one cannot, lower it.
- `BEAM_FRICTION`, `MOTOR_FRICTION`: depend on your floor (table, carpet, tile).
- `GEARBOX_INERTIA`: keep this above about 0.0002, or the very light beams make the simulation blow up.

After changing them, run `walk.py` again and compare the movie with the real robot running the same speeds.

## Tuning (in `mjlab_task/walk_task.py`)

- **It wanders off the line or wobbles:** raise the `stay_on_line` and `face_forward` weights.
- **It walks but slowly:** raise `TARGET_SPEED`, or check `STALL_TORQUE` is right.
- **Beams jitter back and forth:** make the `action_rate` weight more negative (for example -0.05).
- **It flips over and gets stuck:** train longer, or add a reward for staying upright
  (mjlab's `flat_orientation_l2`, with a small negative weight).
- **Works in simulation but not on the real robot:** widen the ranges (`BEAM_FRICTION_RANGE`, `TORQUE_SCALE_RANGE`,
  `COMMAND_DELAY_STEPS`) to cover your robot, check `real_robot.json` is up to date, and compare `real_run-*.csv` with
  the simulation's run to see which reading differs.
- **`run_real.py` says it is running late:** move the laptop closer to the motor; Bluetooth is the slow link.

## What was tested

Trained on the cluster (job on a Pax GPU node, 4096 robots, 1000 iterations, 15 minutes; mjlab 1.6.1) with this
CAD robot, measured masses (90 g motor, 2 g per beam), the long end trailing, and the estimated top speed of 15 rad/s
(no `real_robot.json` yet). Mean reward reached about 7.4.

In simulation (`walk.py`, 10 seconds):

| | Forward | Furthest off the line | Heading |
|---|---|---|---|
| Before training (both beams at 60%) | 0.44 m | 0.49 m | drifts to -54 to -72 degrees |
| Trained policy, 9 starts (random beam angles, start heading up to 19 degrees off) | 0.77 to 0.89 m | 4.6 to 9.8 cm | rocks +/- 20 degrees with each step, but the path stays straight |

The real-robot programs (`calibrate.py`, `run_real.py`) were tested on the simulated motor (`--sim`): calibration finds
its hidden settings (sensor direction within 4 degrees, gyro units within 1%, top speed within 1%, beam zero within
1 degree), and the numpy policy matches mjlab's to within 0.000001. On the **real robot** (card red 1131, on tile, `legoeducation` 1.1.1 because the motor's firmware is RPC 2.0):
`calibrate.py` measured a top speed of 17.6 rad/s, and `run_real.py policy.npz --card "red 1131" --countdown 10` walked
it **in a straight line** for 10 seconds: heading averaged -0.6 degrees (swinging about +/-10 with each step, ending at
0), every step 50 ms apart. It only went about 8 cm (85 cm in simulation): its beams slip on tile. With both beams at a
fixed 60% and no policy (`--fixed 60`) it circles to the left, so the policy is what keeps it straight.

The kept `policy.npz` is that first one (trained with an estimated top speed of 15 rad/s; `run_real.py` sends
85% for full speed on this motor). After it, the simulation's contact friction was fixed (the floor's default 1.0 used
to override the robot's lower values) and training was set up for tile and a lopsided robot (`BEAM_FRICTION_RANGE`
0.1 to 0.5 per beam, `COM_OFFSET`). Training again with these should walk further on tile; that run has not been done.

If `calibrate.py` or `run_real.py` says the motor's firmware does not match: update the motor at
code.legoeducation.com, or install the library that matches it (`pip install legoeducation==1.1.1` for RPC 2.0).
