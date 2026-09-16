# ME193 — LEGO car + camera scripts

Python scripts for driving LEGO Education SPIKE motors over Bluetooth and
steering a car by tracking an AprilTag with a camera (built-in webcam or an
iPhone over Continuity Camera).

macOS only — the camera and Bluetooth paths both go through Apple frameworks.

## Setup

Use **Python 3.12 from a framework build** (`python.org` installer or
Homebrew). This matters: `enable_continuity_camera.py` patches the
interpreter's `Python.app` bundle, which only framework builds have. Python
3.14 currently has no `pupil-apriltags` wheel.

```bash
python3.12 -m venv my_env_ME193
source my_env_ME193/bin/activate
pip install -r requirements.txt
```

The venv directory is gitignored — each person makes their own.

### One-time: let Python see the iPhone camera

Skip this if you only use the built-in FaceTime camera.

```bash
python enable_continuity_camera.py     # run with the venv activated
```

macOS hides Continuity Camera devices from any app bundle that does not
declare `NSCameraUseContinuityCameraDeviceType`. The script adds that key to
your interpreter's `Python.app/Contents/Info.plist`, keeps a `.bak` beside it,
and rolls back automatically if the interpreter stops launching. It is safe to
re-run — it reports "already set" and exits. Undo with `--undo`.

If the plist is not writable, re-run under `sudo`.

On the phone: Settings → General → AirPlay & Continuity → Continuity Camera on;
same Apple ID as the Mac; Wi-Fi and Bluetooth on; phone locked, still, and near
the Mac with the rear camera facing the scene. iOS only offers the camera when
the phone is stationary.

## Scripts

| Script | What it does |
| --- | --- |
| `camera.py` | Helper module — opens a camera *by name* so nothing hardcodes a fragile index. Imported by the others. |
| `list_cameras.py` | Prints every camera OpenCV can open, with names and resolutions. Run this first to see what you have. |
| `iphone_video.py` | Opens the iPhone feed and shows it live. Sanity check before tracking. Press `q` to quit. |
| `enable_continuity_camera.py` | One-time interpreter patch described above. |
| `test.py` | Minimal SingleMotor check — connects, spins 3 s, stops. |
| `apriltag_tracker.py` | The real one: steers a DoubleMotor car off the on-board iPhone feed to aim it at an AprilTag. Press `q` to stop. |

### Before you run the motor scripts

Edit the connection info at the top of the script to match your Connection
Card — `CARD_COLOR` / `card_color` and `CARD_SERIAL` / `card_serial` (the
4-digit number). Turn the hub on first; the scripts exit if they cannot
connect.

`apriltag_tracker.py` drives off the iPhone riding on the car — the same feed
`iphone_video.py` shows. `CAMERA_NAME` (`'iPhone'`) is matched against the names
`list_cameras.py` prints, so the phone is found whatever index it lands on;
that matters because the index shifts depending on whether the phone was
connected when the script started.

It will not fall back to the Mac's built-in webcam. That camera is bolted to
the desk, not the car, so steering by it would be meaningless — if the phone
isn't found the script stops and says so. To deliberately test with the
built-in camera, set `CAMERA_NAME = 'FaceTime'` and `REQUIRE_NAMED_CAMERA =
False`.

Make sure the tag the car aims at is from the `TAG_FAMILY` family (`tag36h11`
by default). If the car turns the wrong way, flip `REVERSE_STEERING`.

## Troubleshooting

**`list_cameras.py` only shows the FaceTime camera.** The Continuity patch has
not been applied to *this* interpreter, or the phone is not eligible — walk the
checklist above. The `out device of bound` warnings for indices past the last
real camera are normal; the script probes 5 slots and reports what answered.

**Camera opens but returns no frame.** Usually another app holds it, or macOS
has not granted camera access yet. Check System Settings → Privacy & Security →
Camera.

**Motor will not connect.** Confirm the hub is powered on, the card color and
serial match, and Bluetooth is on. `legoeducation` talks BLE through `bleak`,
so the terminal app needs Bluetooth permission too.
