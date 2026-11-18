# Campus Nav

A thigh-worn guide that walks you to **Nolop** (or any place you've taught it)
using only the UNO Q's IMU: no GPS, no Wi-Fi positioning, no camera. It's built
on top of the [step counter](../step-counter/README.md), adds a **stairs
classifier**, and the navigation itself is **behavior cloning** from at most
10 demonstrated walks (DAgger corrections included), from both partners.
The board hosts a phone-friendly web page for demonstrating, reviewing,
training, guiding and correcting.

## How it works

```
raw IMU, 100 Hz (MPU-6050 on the thigh, same sketch as the step counter)
  -> step counter (spikes.py, unchanged): steps
  -> heading: "up" from the accelerometer (only trusted when the thigh is
     nearly still, mid-stance), carried between steps by the gyro; turn rate =
     gyro . up, integrated. The gyro bias is measured standing still before
     every walk.
  -> turns: heading changed > 50 deg and settled
  -> stairs: one prediction per stride (thigh angle range and peak, swing
     speed, stride time, bounce) -> Gaussian naive Bayes flat / up / down;
     2 strides in a row = a flight
  -> the navigator's state, 5 times a second:
       distance walked, heading, dead-reckoned x/y, turns made so far,
       distance since the last turn, flights of stairs so far, on stairs?
  -> navigator: k-nearest neighbours (k=5) from that state to the
     instruction: straight / left / right / stairs up / stairs down / arrived
```

The navigator never sees raw IMU (the hint: the IMU is noisy). It sees the
step counter's distance plus a few odometry features, so a handful of walks is
enough.

**Labels come from the walk itself, not from tapping while walking.** During a
demonstration you only tap **Arrived**. The turns and stairs are detected from
the IMU, and in **Review** you fix them on a map of the walk (move, add,
delete, retype events), then save. Every moment of the walk then gets the
instruction a guide should have been giving: a turn or stairs instruction from
5 m before it until it's done, "arrived" at the end, "straight" otherwise.
Cleaning the labels like this means fewer walks are needed.

**DAgger:** in **Guide**, when the instruction is wrong you tap what it
should be. That walk is saved too (it counts toward the 10). In Review the
correction marks show where the navigator went wrong. After saving and
retraining, the navigator has examples from the states *it* led you into, not
just the demonstrator's.

**Training** uses leave-one-walk-out cross-validation and picks the feature
weights (how much turns-so-far, distance, heading and position matter) with
the best balanced accuracy, since "straight" is most of every walk. Only
reviewed walks marked "use" are used, at most 10 across both partners. The
page shows the count.

## Setup (once)

From the `ME193` folder (same venv as the step counter works):

```
python3.12 -m venv my_env_steps_py312            # if you don't have it
my_env_steps_py312/bin/pip install -r campus-nav/requirements.txt
```

## Run it

```
cd campus-nav
source ../my_env_steps_py312/bin/activate
python run.py --person fiona
```

`run.py` uploads the app if it changed (about a minute; it stops the
step-counter and gait-identifier apps, which use the same sensor), copies your
step length from `../step-counter/calibration.json`, and prints the page's
address: `http://10.247.137.128:8050`.

Phone and board both stay on **Tufts_Robot**. Run `python board_wifi.py` once
per board: out of the box the board dropped off Tufts_Robot about every 2
minutes (it hopped between 5 GHz access points), which froze the page. That
script locks the board to 2.4 GHz and turns off Wi-Fi power saving.
`python board_wifi.py --check` shows the signal and how many drops in the last
hour. If the Wi-Fi still blips, the page says "Reconnecting…" and catches up
by itself; the board keeps recording and guiding meanwhile.

Wear it like the step counter: front of the thigh, above the knee, strapped
firmly.

## The workflow

The page has two modes, switched at the top:

- **🎓 Training** opens on **Steps**, a checklist that walks you through
  everything below, with the buttons for each step inside it. Steps tick
  themselves off from what's on the board; the green one is next. When you
  add a walk, "Review" and "Train" reopen by themselves. On the other tabs a
  bar shows which step you're on.
- **🧭 Navigate** is just the guide: pick the route and go.

The steps, in short:

1. **Setup tab**: pick who's wearing it (step length). Do the sensor check:
   stand still, Zero, turn a full circle on the spot: about 360°.
2. **Train tab → Stairs**: record ~30 s each of flat walking, stairs up and
   stairs down (stand still 2 s first). **Train stairs**. These don't count
   as walks.
3. **Walk tab**: type the starting point and how you're facing (every walk
   of that route has to start the same way, since heading is relative to the
   start), **Start**, stand still until it says Go, walk to Nolop, tap
   **Arrived**.
4. **Review tab**: open the walk, fix the events, **Save review**.
5. Repeat 3-4 a few times per route, both partners (e.g. 3 demos each).
6. **Train tab → Train navigator.** Check the leave-one-walk-out numbers.
7. **Guide tab**: pick the route, Start, stand still, follow it. Tap a
   correction whenever it's wrong; **Stop & save**. Review that walk, retrain.
   Repeat with the walks you have left.

### Both partners

If you share one board, all walks are on it already. With two boards (or to
work on the laptop):

| Command | What it does |
|---|---|
| `python sync.py` | Copy walks board ↔ laptop (`data/`), both ways |
| `python sync.py --host <A> --host <B>` | Two boards: everything ends up on both |
| `python studio.py` | The same page on the laptop (Review/Train only, no board needed) |
| `python train.py` | Train on the laptop (bigger weight search), plots in `results/` |
| `python train.py --push` | ...and put the models on the board |
| `python deploy.py --logs` / `--stop` / `--start` | Follow, stop, start the board app |

## Files

| File | |
|---|---|
| [unoq_app/python/navcore.py](unoq_app/python/navcore.py) | Heading, turns, stairs, the navigator's state, labels, kNN, live smoothing |
| [unoq_app/python/spikes.py](unoq_app/python/spikes.py) | The step counter (copied unchanged from step-counter) |
| [unoq_app/python/store.py](unoq_app/python/store.py) | Saved walks, review suggestions, training, cross-validation |
| [unoq_app/python/server.py](unoq_app/python/server.py) | The page's JSON API (board and laptop) |
| [unoq_app/python/main.py](unoq_app/python/main.py) | Board: sensor loop, demo / guide / stairs modes |
| [unoq_app/python/web/index.html](unoq_app/python/web/index.html) | The phone page |
| [unoq_app/python/settings.py](unoq_app/python/settings.py) | Thresholds: turn angle, lookahead (5 m), max walks (10), ... |

Walks on the board live in `unoq_app/python/data/` (deploying never
overwrites them). `sync.py` copies them to `data/` here.

## Tested so far (before any campus walks)

Replayed the step-counter recordings through it:

- On the 50 m out-and-back walk, with 3 s of standing still first, the heading
  holds within about ±15° on the straights, shows the ~170° turnaround, and
  dead reckoning ends **1.7 m** from the start. Without measuring the gyro bias
  first it drifted hundreds of degrees, hence standing still before each walk.
- The board app, run on the laptop with a fake bridge, went demo → save →
  review → train → guide with a correction → saved DAgger walk. The live states
  match a replay of the saved raw data exactly.
- The stairs classifier hasn't seen real stairs data yet, and nothing has been
  walked on campus yet.

## Limits

- Heading is relative: start each walk facing the way you wrote in the route's
  note. Drift grows over long walks; the navigator leans on turns-so-far and
  distance-since-turn, which don't drift.
- The step length is fixed per person, so distances on the walk are off by
  however much your stride changes.
- The page needs Tufts_Robot wherever you walk. Indoors it's good; check the
  outdoor parts of your route before relying on it.
