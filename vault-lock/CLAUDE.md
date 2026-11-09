# Vault lock (ME193)

Finger-code safe: a CNN (PyTorch, from scratch) counts fingers under a USB
webcam; three numbers unlock a servo. Runs on an Arduino UNO Q, with a live
view on the laptop. README.md explains the files; CONCEPTS.md explains the
neural-net theory for the assignment.

## "Run the vault code"

When the user says "run the vault code" (or similar), run, from this folder:

```
../my_env_vault_py312/bin/python run.py
```

Run it in the background (it opens a window and stays open until the user
presses q). It prints 4 steps: (1) board reachable over SSH, (2) uploads
`unoq_app/` only if it changed since the last upload, otherwise makes sure the
app is running, (3) waits for the board's camera, (4) opens the laptop view
window "Vault lock (UNO Q)". Check its output got to step 4, then tell the user
the window is open (Cmd+Tab if hidden): SPACE or the D2 button starts, hold
each number 0.8 s, `h` hides numbers, `q` closes the window.

- If the webcam is plugged into the **laptop** instead of the board:
  `../my_env_vault_py312/bin/python run.py --laptop` (camera index 0 = the
  SK-C201 webcam; 1 = FaceTime; 2 = iPhone).
- If step 1 fails: the laptop must be on Wi-Fi **Tufts_Robot** (it has been on
  tufts_eecs before, which can't reach the board); the board is
  `arduino@10.247.137.128`, hostname `ourduino`; the SSH key
  `~/.ssh/id_ed25519` is installed on it (passwordless).
- Watching what the board does: `../my_env_vault_py312/bin/python deploy.py --logs`,
  or poll `http://10.247.137.128:8040/status.json` (lock mode, digits, button
  D2 raw state and press count). `deploy.py --peek` saves the board camera's
  frame to results/peek.jpg. `deploy.py --stop` stops the lock.

## State (as of 2026-11-09)

- Working end to end on the board: button D2 → scanning, 3-2-3 unlocks (servo
  D9 to 90°), wrong code denied, button relocks. ~15 frames/s, CNN 25 ms/frame.
- The user's code is **3-2-3** (salted hash in `unoq_app/python/code.json`,
  git-ignored). Codes must not repeat a number twice in a row.
- Model: width 16, depth 4, 95.2% on held-out recordings (1-5 perfect). Known
  weakness: a side-on fist can read as "1"; fix = record ~4 more "0" bursts of
  side-on fists (`collect.py --camera 0`) and retrain (`train.py`), then
  `run.py` uploads the new model automatically.
- After a fresh sketch flash the board app sometimes exits once (~15 s in);
  run.py restarts it.

## Gotchas

- App Lab needs library versions in `unoq_app/sketch/sketch.yaml`
  (`Servo (1.3.0)`); `Arduino_RouterBridge` is bundled with the core now.
- The board has non-camera /dev/video0-1 (codec); main.py opens the webcam by
  `/dev/v4l/by-id/usb-*-video-index0` first.
- Don't use the synthetic data in the session scratchpad for anything real;
  `docs/synthetic_*.png` are the only synthetic artifacts kept.
