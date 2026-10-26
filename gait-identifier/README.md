# Gait Identifier

A thigh-worn Arduino UNO Q + MPU-6050 that tells apart **still / walk / run /
bike** with two non-neural models: one using only the peak acceleration on one
axis, and one using all six IMU channels. The board classifies on its own;
the laptop records training data, trains, and shows the live answers.
[PLAN.md](PLAN.md) has the full design.

## Setup (once)

From the `ME193` folder:

```
python3.12 -m venv my_env_gait_py312
my_env_gait_py312/bin/pip install -r gait-identifier/requirements.txt
```

Optional, so SSH stops asking for the board's password:
`ssh-copy-id arduino@10.247.137.128`

## Run it

```
cd gait-identifier
source ../my_env_gait_py312/bin/activate
python run.py                  # pick a person from the list
python run.py --person fiona   # or name them directly
```

`run.py` trains on that person's recordings, uploads the models to the board
if they aren't there yet (asks for the board's password, about 1 minute),
starts the app if needed, and shows both models' answers live. Type `q` to quit.

## Add a person / record more

```
python guided_train.py    # walks you through 12 recordings, then trains
python run.py             # then pick them
```

## Other commands

| Command | What it does |
|---|---|
| `python live_view.py` | Just watch the live answers (and send commands like `record still`, `stop`) |
| `python deploy.py` | Upload `unoq_app/` and restart the app on the board |
| `python deploy.py --start` | Start the app (e.g. after the board rebooted) |
| `python deploy.py --logs` | Follow the board's own output |
| `python train.py [--person X]` | Train both models (copies them to `unoq_app/python/models/`) |
| `python evaluate.py [--person X]` | Accuracy, confusion matrices, plots in `results/` |
| `python collect.py walk --person X` | Record one session by hand |

## Network notes

- The board and laptop talk over MQTT via `broker.emqx.io`
  (`test.mosquitto.org` stopped answering from Tufts). Settings are in
  [unoq_app/python/settings.py](unoq_app/python/settings.py).
- Arduino App Lab can't see the board over Tufts Wi-Fi (mDNS is blocked), so
  everything goes over SSH to the board's IP instead. The board is
  `arduino@10.247.137.128` on **Tufts_Robot**. If its IP changes, pass
  `--host <new IP>` to `run.py` / `deploy.py`, or update `BOARD_HOST` in
  [deploy.py](deploy.py). Over USB-C, App Lab still works.

## Wearing it

Front of the thigh, just above the knee, +X arrow pointing down toward the
knee, strapped firmly. Mount it the same way as when the recordings were made.
