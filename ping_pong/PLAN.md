# Virtual Ping-Pong — Midterm Plan

Play ping-pong against a virtual wall/opponent using the **LEGO Double Motor
as the paddle**. The webcam watches you, the Double Motor's IMU feels your
swing, AprilTags pick the level and start the game, and your best rally is
posted live over MQTT.

## Requirement → how it's covered

| Requirement | How |
|---|---|
| **Pose** (is the paddle in the right place?) | MediaPipe Pose on the webcam (`camlib.pick_camera()` + MediaPipe VIDEO mode, as in CAMLIB.md). Track the wrist of your paddle hand. A hit only counts if the wrist is within `HIT_RADIUS` of where the ball crosses your plane. |
| **IMU** (are they swinging?) | Double Motor IMU (`imu_device.gyroscopeX/Y/Z`, `accelerometerX/Y/Z`, `roll`). A swing is a spike in gyro magnitude above a calibrated threshold. Hit = **in position AND swinging** within a ±150 ms window of the ball arriving. |
| **AprilTag** (level + start) | `pupil_apriltags` (same as `apriltag_tracker.py`). Tags 1/2/3 = Easy/Medium/Hard (ball speed). Tag 0 = START. Tag has to be seen for ~10 frames in a row before it counts, so a quick flash doesn't trigger it. |
| **Live score over MQTT** | `paho-mqtt` publishes the **current number of continuous hits** as a float (e.g. `"7.0"`) to `ME193/Rogers/Fiona` on `test.mosquitto.org`. Published every time it changes (including dropping to `0.0` on a miss), plus a heartbeat every 5 s. |
| **Everything from class** | `lelib` (BLE + connection card), IMU, PD control (9-17), AprilTag + camera (tracker), pose/MediaPipe (camlib), audio/whistle (miclib + whistle_class) |
| **Something new / fun** | **Haptic feedback**: an off-center weight on one Double Motor output turns it into a rumble motor (see below). Also **paddle tilt changes the return angle** (IMU roll), so you can aim the ball. |

## Hardware setup

- **Paddle** = Double Motor held in your hand (or strapped to a cardboard paddle).
  - Motor A: LEGO beam with an off-center weight → rumble motor.
  - Motor B: short arm that taps your hand → "thunk" on contact, using a
    **PD position move** (Kp/Kd idea from `9-17.py`): snap to +40°, then back to 0.
- **Webcam** on the laptop facing you (about 1.5–2 m away, upper body in frame).
- **AprilTags** (tag36h11) printed on cards: IDs 0 (START), 1, 2, 3 (levels).
- Built-in **hub light** shows the level color; **hub beep** for sound effects.

## Game flow (state machine)

```
CONNECT ─► CALIBRATE ─► LOBBY ─► COUNTDOWN ─► RALLY ─┬─(hit)──► RALLY
                          ▲                          └─(miss)─► MISS ─► COUNTDOWN
                          └──────── 3 misses / GAME OVER ◄──────────────┘
```

1. **CONNECT** – connect the Double Motor by connection card (`lelib.doubleMotor().connect(CARD_SERIAL, CARD_COLOR)`), open the camera, connect to MQTT.
2. **CALIBRATE** – "hold the paddle still for 2 s": record gyro noise → swing threshold = mean + 6·std (with a sensible minimum). Then `reset_heading()` and record the neutral roll.
3. **LOBBY** – show a level tag (1/2/3) → sets ball speed, hub light color, confirmation beep. Show tag 0 → start.
4. **COUNTDOWN** – 3-2-1 on screen + beeps.
5. **RALLY** – the ball flies toward you (gets bigger as it gets closer). When it reaches your plane:
   - wrist near ball **and** swing detected → **HIT**: streak +1, short rumble + arm thunk, ball goes back. The return angle comes from where it hit on the paddle and your paddle tilt (roll). Speed goes up a little each hit.
   - otherwise → **MISS**: streak resets to 0, long buzz pattern + low beep.
   - While the ball is coming in, a **soft rumble that gets stronger as it gets closer** helps you time the swing.
6. **GAME OVER** after 3 misses → back to LOBBY. The record stays for the whole session.
7. **Optional (whistle)** – a high whistle pauses or resumes the game, reusing the band detector from `whistle_class.py` and `miclib.pick_mic()`.

## Ball model (pure math, no hardware)

- Ball state: `x, y` (normalized 0–1 screen coords), `z` (depth: 1 = far wall, 0 = player), `vx, vy, vz`.
- Each frame: `pos += vel * dt`; bounces off the left/right walls; the screen radius scales with `1/(z + 0.2)` so it looks 3D.
- When `z <= 0`, check for a hit; the far wall at `z = 1` always returns the ball with a small random angle.
- Level speeds (`vz`, depth units/s): Easy 0.35, Medium 0.55, Hard 0.8; ×1.05 per hit, with a max.
- Return angle on a hit: `vx = K_OFFSET*(ball.x - wrist.x) + K_TILT*(roll - roll0)`.

## Swing detection (IMU)

- Poll the IMU at about 100 Hz in a background thread.
- `g = sqrt(gx² + gy² + gz²)`, smoothed with a low-pass (EMA, α≈0.4).
- A swing **event** starts when `g` goes above the threshold. After that, it waits 300 ms before counting another swing, so one swing can't count twice.
- Keep a timestamped deque of swing events; `swung_near(t, window=0.15)` checks whether there's a swing close enough to time `t`.
- Debug view: a live bar of `g` with the threshold line, shown in the game window.

## File layout (`ping_pong/`)

| File | Responsibility |
|---|---|
| `ping_pong.py` | Entry point: setup, state machine, main camera loop, drawing the OpenCV overlay |
| `config.py` | Card serial/color, tag IDs, level speeds, thresholds, MQTT broker/topic, name |
| `paddle.py` | `Paddle` class: wraps `doubleMotor`; IMU polling thread, calibration, swing detection, roll |
| `haptics.py` | Non-blocking haptic patterns (`hit()`, `miss()`, `proximity(level)`, `milestone()`) run in a worker thread so they never stall the frame loop; PD thunk on motor B |
| `vision.py` | MediaPipe Pose (wrist landmark) + AprilTag detection on the same frame; tag stability filter |
| `game.py` | `Ball` and `Game` classes: pure logic (physics, hit test, streak/record, states). No hardware imports, so it can be unit-tested |
| `scoreboard.py` | MQTT client (`paho-mqtt` v2, `loop_start()`), publishes the current streak as a float, heartbeat, reconnect |
| `whistle.py` *(optional)* | Pause/resume from a whistle, reusing the band detector |
| `lelib.py`, `camlib.py`, `miclib.py` | Copied from `class/` |
| `test_game.py` | Tests for ball physics, hit test, and streak/record logic (no hardware needed) |

## Threads

- **Main thread**: camera read → vision → game update → draw → `cv2.imshow` (OpenCV windows have to run on the main thread on macOS).
- **IMU thread**: reads `imu_device` and updates the swing detector (all shared state is protected by a lock).
- **Haptics thread**: takes jobs from a queue and runs motor/beep patterns with `blocking=False`.
- **MQTT**: paho's own network thread (`loop_start`).
- Shutdown in a `finally:` block: stop the motors, disconnect BLE, publish the final record, `loop_stop()`, release the camera.

## Build order (each step is testable on its own)

1. `game.py` + `test_game.py`: ball and scoring logic, played with the **mouse as the paddle** and the **space bar as the swing**. No hardware.
2. `scoreboard.py`: publish a fake record and check it shows up with `mosquitto_sub -t 'ME193/Rogers/#'`.
3. `paddle.py`: print `g` live, calibrate, tune the swing threshold.
4. `haptics.py`: feel each pattern; tune the rumble speeds and the PD thunk.
5. `vision.py`: wrist overlay + tag IDs drawn on the frame.
6. `ping_pong.py`: swap the mouse for the wrist and the space bar for the IMU, then add the states and the overlay.
7. Tune: hit radius, swing window, level speeds. Record a demo video.

## Environment & running

The project has its own Python 3.12 virtual environment at the repo root, **`my_env_pingpong_py312/`**: ping-pong project, Python 3.12. It's separate from `my_env_ME193/` (3.12) and `my_env_ME193_py314/` (3.14), which haven't been changed.

```bash
cd ~/Documents/GitHub/ME193
source my_env_pingpong_py312/bin/activate
cd ping_pong
python ping_pong.py              # full game (Double Motor + camera + mic)
python ping_pong.py --sim        # no hardware: mouse = paddle, SPACE = swing
python ping_pong.py --no-whistle # skip the microphone
python -m unittest test_game     # logic tests, no hardware
```

To rebuild the environment from scratch: `python3.12 -m venv my_env_pingpong_py312 && my_env_pingpong_py312/bin/pip install -r ping_pong/requirements.txt`

Watch the score live: `mosquitto_sub -h test.mosquitto.org -t 'ME193/Rogers/#' -v`

## Decisions (confirmed)

- Broker: `test.mosquitto.org:1883`, topic `ME193/Rogers/Fiona`
- Publish **only the current streak** (continuous hits) as a float
- Hardware: **Double Motor** (not the UnoQ)
