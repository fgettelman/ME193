# Silly walks: a LEGO Double Motor that walks on two beams

Take the LEGO Education **Double Motor**, put a Technic **beam** on each output instead of a wheel, and it walks: the
beams swing round, push off the floor, and the motor rocks and tumbles along. This folder simulates that robot in
MuJoCo and trains a policy with **mjlab** to make it walk in a **straight line**. Everything is Python.

It uses the same cluster setup as the rest of `mujoco-hpc` (`python3 setup.py` all PASS; mjlab is already installed by it).

| File | What it is |
|---|---|
| `silly_robot.py` | **The robot**, built in Python: motor size and mass, beam length, which hole is on the axle, motor speed and torque. **Measure your parts and change the numbers at the top.** Run it to write `silly_robot.xml` for MuJoCo's viewer. |
| `walk.py` | Walks the robot with a fixed plan (both beams forward, turn, one faster) and films it. Runs on your laptop. |
| `mjlab_task/walk_task.py` | The mjlab task "Silly-Walk": rewards, resets, what the policy sees. **Tune this one.** |
| `mjlab_task/train.py`, `play.py` | mjlab's own train/play commands, with our task registered first |
| `mjlab_task/make_video.py` | Records a movie of a trained policy on the cluster |
| `mjlab_task/train.sbatch` | A Slurm job that trains on a GPU, for running by hand on the cluster |

All commands below are run from the **`mujoco-hpc` folder** (VPN on if off campus).

## 1. See the robot walk (on your laptop, 1 minute)

```
python3 10_silly_walks/walk.py
```
The video `10_silly_walks/walk.mp4` opens in QuickTime. It also prints where the robot is: with both beams at 8 rad/s
it gets about 0.5 m in 4 seconds but turns about 25 degrees, so a fixed plan does not walk straight. That is the
job for the policy. (Needs `pip install mujoco imageio imageio-ffmpeg` once. Or run it on the cluster:
`python3 runIt.py --with 10_silly_walks/silly_robot.py 10_silly_walks/walk.py`.)

## 2. Check the task loads on the cluster (short run)

```
python3 runIt.py --gpu --time 00:10:00 --mem 16G --cpus 4 \
  --with 10_silly_walks/mjlab_task/walk_task.py --with 10_silly_walks/silly_robot.py \
  10_silly_walks/mjlab_task/train.py Silly-Walk --env.scene.num-envs 64 --agent.max-iterations 20
```
Under "job output" you should see training iterations with a mean reward.

## 3. Train

```
python3 runIt.py --gpu --time 00:45:00 --mem 16G --cpus 4 \
  --with 10_silly_walks/mjlab_task/walk_task.py --with 10_silly_walks/silly_robot.py \
  10_silly_walks/mjlab_task/train.py Silly-Walk --env.scene.num-envs 4096
```
It trains 1000 iterations by default (change with `--agent.max-iterations`). This robot needs smaller physics steps than
the wheeled robot in `09_onshape_robot` (2 ms instead of 5 ms), so each iteration takes longer; `--time 00:45:00` leaves
room. Checkpoints (`model_<n>.pt`, every 50 iterations) come back to
`runs/train-<time>/logs/rsl_rl/silly_walk/<date>/`. The last one is `model_999.pt`.

## 4. Watch the result: a movie made on the cluster

```
python3 runIt.py --gpu --time 00:10:00 --mem 16G --cpus 4 \
  --with 10_silly_walks/mjlab_task/walk_task.py --with 10_silly_walks/silly_robot.py \
  --with runs/train-XXXXXXXX-XXXXXX/logs/rsl_rl/silly_walk/XXXX-XX-XX_XX-XX-XX/model_999.pt \
  10_silly_walks/mjlab_task/make_video.py model_999.pt
```
Do not type the `X`s: type `runs/train-` and press Tab to complete each folder name. It prints where the robot is
each second (x forward, y sideways, heading): walking straight keeps y and heading near 0.

## What the policy sees (and why)

The policy only gets what the **real Double Motor measures itself**, so the same policy could later drive the real robot:

| Policy input | On the real Double Motor |
|---|---|
| tilt (which way is down) | `Pitch`, `Roll` |
| turning rate | `GyroX`, `GyroY`, `GyroZ` |
| heading | `Yaw` |
| each beam's angle | `LeftPosition`, `RightPosition` |
| each beam's speed | `LeftSpeed`, `RightSpeed` |
| what it did last step | (remembered by your code) |

It outputs one speed per beam. Output 1.0 means `MAX_SPEED` rad/s, which is the motor at 100%.
During training, the critic also sees the robot's true speed and how far it is from the line, which the real robot
cannot measure. That is allowed because the critic is only used for training.

Rewards (in `walk_task.py`): walk forward at `TARGET_SPEED` (0.15 m/s), stay on the line y = 0, face along +x, and do
not change beam speeds too jerkily. Each robot starts pointing up to 0.4 rad off +x with its beams at random angles.
There is no "tipped over" ending, because rocking and tumbling is how this robot walks.

## Measure your robot (in `silly_robot.py`)

The numbers there are **estimates**. The simulation is only as good as they are:

- `MOTOR_SIZE`, `MOTOR_MASS`, `AXLE_HEIGHT`, `AXLE_X`: measure the motor and where the outputs are; weigh it.
- `BEAM_HOLES`, `AXLE_HOLE`: which beam you use and which hole is on the output. An off-centre hole
  (for example hole 2 of 9) makes a much sillier walk.
- `MAX_SPEED`: run the motor at 100% and time the beam: rad/s = 2π × turns per second.
- `STALL_TORQUE`: how hard the motor pushes. If the simulated robot climbs over things the real one cannot, lower it.
- `BEAM_FRICTION`, `MOTOR_FRICTION`: depend on your floor (table, carpet, tile).
- `GEARBOX_INERTIA`: keep this above about 0.0002, or the very light beams make the simulation blow up.

After changing them, run `walk.py` again and compare the movie with the real robot running the same speeds.

## Tuning (in `mjlab_task/walk_task.py`)

- **It wanders off the line or wobbles:** raise the `stay_on_line` and `face_forward` weights.
- **It walks but slowly:** raise `TARGET_SPEED`, or check `MAX_SPEED` and `STALL_TORQUE` are right.
- **Beams jitter back and forth:** make the `action_rate` weight more negative (for example -0.05).
- **It flips over and gets stuck:** train longer, or add a reward for staying upright
  (mjlab's `flat_orientation_l2`, with a small negative weight).
- **Make it hold up on the real robot:** randomize friction and mass (mjlab's `mjlab.envs.mdp.dr`), so the policy does
  not depend on one exact value.

## What was tested

On a Mac CPU with mjlab 1.6.1 (not the cluster): the task loads, beams start at random angles, and a 60-iteration
training run with 256 robots raised the mean reward from about -4 to +6. That barely-trained policy walked 0.9 m in
8 seconds, staying within 6 cm of the line but wobbling up to 30 degrees in heading, then flipped onto its end.
A full run on the cluster (step 3) has not been done yet.

## On the real robot (not done yet)

Running the trained policy on the real Double Motor needs a small Python program that reads the sensors in the table
above, runs the policy, and sends the two speeds as percentages (`speed / MAX_SPEED * 100`). The repo's AnnaCode
uses the `legoeducation` library for this. Watch the direction: on the real motor the two outputs face opposite ways,
so one side's speed has to be negated to make both beams go "forward" (AnnaCode's `BothSpeed` is `(Right - Left) / 2`).
