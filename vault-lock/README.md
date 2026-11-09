# Vault Lock: a finger-code safe

Walk up to the safe, **press the button**, the **LEDs light up**, and show the
camera **three different numbers (1-5) with your fingers**, one after the
other. A convolutional neural network, built from scratch in PyTorch, counts
the fingers. Right code: the **servo turns 90°** and the safe opens. Wrong
code: it stays locked, and three wrong codes in a row lock it out for 30 s
(then 60 s, 120 s...).

Hardware: Arduino UNO Q, a USB webcam (on the laptop first, then on the UNO
Q through a USB-C hub), a push button, a hobby servo.

How the neural net works (width, depth, backprop, gradient descent,
convolution, pooling, and Grad-CAM, the new thing) is in
**[CONCEPTS.md](CONCEPTS.md)**.

```
 webcam ─> centre square, 64x64 ─> FingerNet (CNN) ─> P(0..5 fingers) ─> lock.py ─> LEDs, matrix, servo
                                    4 conv layers                          state machine
```

## Quick start: run it

Laptop on Wi-Fi **Tufts_Robot**, webcam on the UNO Q's hub, then from the
`vault-lock` folder:

```
../my_env_vault_py312/bin/python run.py
```

It checks the board, uploads anything that changed (code, model, your code,
sketch), starts the lock, and opens **the live view on the laptop**: the
board's camera with the instructions, the combination boxes, the LEDs, the
servo and the button's wiring. SPACE in that window works as the button too;
`q` closes the window (the lock keeps running on the board).
`run.py --laptop` runs the whole lock on the laptop instead, with the webcam
plugged into the laptop. (Or just tell Claude "run the vault code".)

## Files

| File | What it does |
|---|---|
| [collect.py](collect.py) | Take training pictures with the webcam (keys 0-5, SPACE) |
| [model.py](model.py) | The CNN (width and depth are settings), Grad-CAM, BatchNorm folding, export |
| [train.py](train.py) | Augmentation, the training loop, gradient descent written out by hand, test on held-out recordings, export to the board |
| [gradcam.py](gradcam.py) | Heat maps of where the net looks (the new thing) |
| [sweep.py](sweep.py) | Trains several widths and depths and compares them |
| [backprop_demo.py](backprop_demo.py) | Backprop by hand (dense, conv, max pool), checked against PyTorch; gradient descent by hand |
| [laptop_lock.py](laptop_lock.py) | **Step 1:** the whole lock on the laptop + webcam, live Grad-CAM |
| [set_code.py](set_code.py) | Set the code (stored as a salted hash) |
| [deploy.py](deploy.py) | **Step 2:** put it on the UNO Q over SSH, logs, peek at its camera |
| [board_view.py](board_view.py) | The board's camera and lock status, live on the laptop; SPACE = button |
| [run.py](run.py) | **Everything in one command:** check the board, upload if changed, start, open the live view |
| [unoq_app/python/](unoq_app/python/) | Runs on the board **and** is used by the laptop scripts: [lock.py](unoq_app/python/lock.py) (lock logic), [fingernet.py](unoq_app/python/fingernet.py) (the CNN in numpy), [vision.py](unoq_app/python/vision.py) (frame → net input), [settings.py](unoq_app/python/settings.py), [main.py](unoq_app/python/main.py) (board only) |
| [unoq_app/sketch/sketch.ino](unoq_app/sketch/sketch.ino) | Microcontroller: button, RGB LEDs, LED matrix, servo |

## Setup (once)

From the `ME193-vault` folder:

```
python3.12 -m venv my_env_vault_py312
my_env_vault_py312/bin/pip install -r vault-lock/requirements.txt
```

Then for everything below:

```
cd vault-lock
source ../my_env_vault_py312/bin/activate
```

Optional, to see it working before you collect anything:
`python model.py` prints the network's layers, and `python backprop_demo.py`
checks hand-derived gradients against PyTorch.

## Step 1: the laptop and the USB webcam

Point the webcam **down** at where your hand will go (as it will look inside
the safe). Use the new USB webcam, not the laptop's camera or your phone:
every script shows a thumbnail of each camera, and you press the number of
the USB one (or pass `--camera N` once you know it).

**1. Collect pictures.** `python collect.py`

- Press `1`-`5` for that many fingers, or `0` for "no number" (empty, a
  fist, a flat hand, anything else), then **SPACE** to record one 3 s burst
  (24 pictures). Keep your hand inside the green square.
- **Do lots of short, different bursts**: 20-30 per count. Between bursts
  change hand, which fingers, spread, rotation, height, position, lighting,
  sleeve. On synthetic hands, 5 recordings per count gave 40% on new
  recordings and 30 gave 97%: the variety matters more than the number of
  pictures.

**2. Train.** `python train.py` (a few minutes)

Prints the layer table, then loss and accuracy each epoch, a confusion
matrix on recordings it never trained on, and how often the lock's 80%
confidence threshold is met. Saves `results/training.png` and
`results/confusion.png`, the board's model
`unoq_app/python/models/fingernet.json` (only if the numpy version matches
PyTorch), and `models/fingernet.pt`.

**3. Look inside.** `python gradcam.py` → `results/gradcam.png` (and
`--wrong` for the mistakes). The heat should be on your fingers. If it's on
the background or your sleeve, collect more varied pictures.
`python sweep.py` compares widths and depths (`results/sweep.png`, ~10 min).

**4. Set the code.** `python set_code.py 4 1 3` (three numbers from 1 to 5, no
two in a row the same, so 3 2 3 works but 3 3 2 doesn't; the default is 1 3 5).

**5. Try the lock.** `python laptop_lock.py`

SPACE is the button, `g` turns on live Grad-CAM, `q` quits. The panel on the
right shows what the LEDs, the matrix and the servo will do. Press SPACE, hold
up your first number until the bar fills (0.8 s), change to the second, then
the third.

## Step 2: the UNO Q

### Wiring

| Part | UNO Q |
|---|---|
| Push button | one leg to **D2**, the other to **GND** (internal pull-up, no resistor needed) |
| Servo signal (orange/yellow) | **D9** |
| Servo power (red) / ground (brown) | **5V** / **GND**. A large servo needs its own 5 V supply with the grounds joined, or the board may reset when it moves. |
| USB webcam | USB-C hub on the UNO Q's USB-C port (a powered hub is safest) |
| LEDs | on the board: RGB LEDs 3 and 4 and the 13x8 LED matrix |

Mount the webcam the way it was when you collected the pictures (same height,
looking down). If the board's view is different, collect some more pictures
from there and retrain.

### Deploy

```
python deploy.py                 # upload code + model + code.json, restart (~1 min)
python deploy.py --check-camera  # does the app see /dev/video*?
python deploy.py --logs          # watch it: frames/s, what it sees, each number entered
python deploy.py --peek          # results/peek.jpg: what the board's camera sees right now
python board_view.py             # live view of the board on the laptop (run.py does all of this)
```

The board's IP is `BOARD_HOST` in [deploy.py](deploy.py) (`--host` to
override). Deploying stops the step-counter and gait-identifier apps, since
they also use the microcontroller.

### What the safe does

| | RGB LEDs | LED matrix | Servo |
|---|---|---|---|
| Locked, waiting | off | dim padlock | 0° |
| Button pressed: scanning | **white** (lights your hand) | the count it sees, a hold bar, 3 boxes for the numbers entered | 0° |
| Number entered | LED 4 flashes blue | next box fills | 0° |
| Right code | green | ✓ | **90°**, for 30 s or until the button is pressed |
| Wrong code | red | ✗ | 0° |
| 3 wrong codes | flashing red | countdown | 0°, button ignored |
| Python not running | off | blinking `?` | 0° (fail-secure) |

Scanning gives up after 20 s with nothing entered, and pressing the button
while scanning cancels. All the timings are in
[settings.py](unoq_app/python/settings.py).

### If something doesn't work

- **No camera** (`--check-camera` lists nothing inside the app): check the
  hub and its power. If `/dev/video0` exists on the board but not inside the
  app's container, App Lab isn't passing it through; `main.py` also tries
  Arduino's own `Camera` class.
- **Colours look wrong / it's much worse on the board than the laptop**: some
  cameras give RGB instead of BGR. Set `SWAP_RB = True` in
  [settings.py](unoq_app/python/settings.py) and deploy again.
- **The sketch doesn't compile because of `Servo.h`**: set
  `USE_SERVO_LIB 0` at the top of [sketch.ino](unoq_app/sketch/sketch.ino)
  and the sketch makes the servo pulses itself.
- **LEDs on when they should be off**: swap `LED_ON`/`LED_OFF` in the sketch.

- **Nothing happens when the button is pressed**: the live view shows
  "Button D2: released/PRESSED" straight from the pin. If it never says
  PRESSED, the button isn't connecting D2 to GND (on a 4-leg button, use
  diagonally opposite legs).

## Results on the real board (2026-11-09)

- ~15 frames/s; the CNN takes 25 ms per frame on the UNO Q's CPU.
- Button D2 → scanning; 3-2-3 → unlocked (servo 90°); a wrong code →
  denied; the button relocks.
- Model: 95.2% on held-out recordings, 1-5 perfect. A side-on fist can read
  as "1" (Grad-CAM also caught an earlier "watch = 1" shortcut); a few more
  "0" recordings of side-on fists should fix it.
- Still to check by eye: the LED matrix pictures and the RGB LED colours.
