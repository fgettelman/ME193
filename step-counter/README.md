# Step Counter

Last week's thigh-worn walker (Arduino UNO Q + MPU-6050, from
[gait-identifier](../gait-identifier/README.md)) now **counts your steps from
the spikes in the gyroscope** and estimates how far you've gone. A browser
dashboard draws the gyro live, with every spike the board counts marked on it.

## How it works

The sensor is on one thigh, and every step you take moves that thigh
([spikes.py](unoq_app/python/spikes.py), which runs on the board):

- a step with the **tracker leg** swings it forward: a **spike** up in the gyro
- a step with the **other leg** happens while the tracker leg is planted and
  the body passes over it, so the tracker thigh swings back: a **dip** down

Each spike and each dip is **1 step**, and they alternate. So it doesn't
matter which foot you start or stop on: start on the other foot and there's a
dip before the first spike; stop on it and there's a dip after the last one.

1. The three gyro channels are smoothed (4 Hz low-pass) and combined along the
   direction the thigh is rotating most (found from the last 2 s, so it
   doesn't matter how the sensor sits on the strap). That's the **thigh
   swing** line on the dashboard.
2. A spike is the swing going above **80 °/s**, a dip below **-35 °/s** (the
   dashed lines). Two spikes (or two dips) in a row is a wobble, not a step.
3. A step alone is probably fidgeting, so a step only counts once another
   follows within 2 s.
4. **Distance = steps x step length**, from your 5 m walks.

## Setup (once)

From the `ME193` folder:

```
python3.12 -m venv my_env_steps_py312
my_env_steps_py312/bin/pip install -r step-counter/requirements.txt
```

## Run it

```
cd step-counter
source ../my_env_steps_py312/bin/activate
python run.py --person fiona
```

`run.py` uploads the app to the board if it changed (asks for the board's
password, about 1 minute; it stops the gait-identifier app first since both
use the sensor), and opens the dashboard at http://localhost:8035.

## Calibrate ("training"): walk 5 m

On the dashboard, under **Calibrate: walk 5 m**:

1. Mark out 5 m and stand at the start.
2. Press **Start**, walk the 5 m counting your steps, stop.
3. Press **Stop**, check the number of steps you counted, **Save this walk**.

Do it 2-3 times (choose **Run 5 m** to calibrate running too). From the walks
it works out your step length (5 m / steps you counted), and it shows how far
off the board's count was from yours. Walks are saved in `calibration.json` under your name and sent to
the board straight away; no retraining or uploading. Until you calibrate it
uses 0.70 m per step (1.0 m running).

**The dashboard** also shows the step count against a daily goal (click the
goal to change it), distance (km/mi toggle), cadence, still/walk/run (running
above 140 steps/min), a 5-minute cadence chart and time spent in each.
**Pause**, **Resume** and **Reset** control the board.

**To end it:** Ctrl+C in the terminal closes the dashboard. The board keeps
counting until you run `python deploy.py --stop` or unplug its battery.

## Other commands

| Command | What it does |
|---|---|
| `python dashboard.py --person X` | Just open the live dashboard (`--text` for terminal output) |
| `python deploy.py` | Upload `unoq_app/` and restart the app on the board |
| `python deploy.py --start` / `--stop` / `--logs` | Start, stop, or follow the app on the board |
| `python collect.py walk --person X` | Record one raw session to `data/` |
| `python uncertainty.py` | Uncertainty analysis: sensor noise, count, step length, distance, the net (`--no-net` to skip retraining); saves `results/uncertainty.json` |

The earlier neural-net version (`net.py`, `train.py`, `evaluate.py`,
`guided_train.py`, `stepnet.py`, `counter.py`) is still here but the board no
longer uses it.

## Network notes

Same as gait-identifier: MQTT over `broker.emqx.io` (topics under
`ME193/Rogers/Fiona/steps`, set in
[settings.py](unoq_app/python/settings.py)), and SSH to the board at
`arduino@10.247.137.128` on **Tufts_Robot**. If its IP changes, pass
`--host <new IP>` to `run.py` / `deploy.py`, or update `BOARD_HOST` in
[deploy.py](deploy.py).

## Wearing it

Front of the thigh, just above the knee, +X arrow pointing down toward the
knee, strapped firmly, the same way as when the recordings were made.
