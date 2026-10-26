# Gait Identifier

A thigh-worn Arduino UNO Q + MPU-6050 that tells apart **still / walk / run /
bike** with two non-neural models (peak acceleration on one axis vs. all six
IMU channels). The board classifies on its own; the laptop trains and shows
the live answers. Full design: [PLAN.md](PLAN.md).

## Run it

Power the board, put the laptop on **Tufts_Robot**, then:

```
cd ~/Documents/GitHub/ME193/gait-identifier
source ../my_env_gait_py312/bin/activate
python run.py --person fiona     # or just: python run.py  (pick from a list)
```

It trains on that person's recordings, uploads the models to the board if
needed (asks for the board's password), and shows both models' answers live.
**To end:** `q` closes the live view; `python deploy.py --stop` stops the board.

**New person:** `python guided_train.py` (walks through 12 recordings), then
`python run.py --person <name>`.

## Other commands

| Command | What it does |
|---|---|
| `python deploy.py [--start / --stop / --logs]` | Upload and restart the app / start / stop / show board output |
| `python live_view.py` | Watch live answers; send commands like `record still`, `stop` |
| `python evaluate.py [--person X]` | Accuracy, confusion matrices, plots in `results/` |

## Notes

- Wear it on the front of the thigh, just above the knee, +X arrow toward the knee.
- MQTT broker: `broker.emqx.io` (set in [settings.py](unoq_app/python/settings.py)).
- App Lab can't see the board on Tufts Wi-Fi, so everything uses SSH to
  `arduino@10.247.137.128`. If the IP changes, add `--host <IP>`.
- First-time setup (from `ME193/`): `python3.12 -m venv my_env_gait_py312 &&
  my_env_gait_py312/bin/pip install -r gait-identifier/requirements.txt`
