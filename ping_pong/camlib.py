'''
Camera selection and initialisation helper for all ME193 examples.

Usage:
    from camlib import pick_camera
    cap, start_ms = pick_camera()
'''

import subprocess
import json
import time
import cv2
import numpy as np


def _camera_info():
    """Cameras from macOS system_profiler as (name, is_phone), in OpenCV's
    AVFoundation index order -- OpenCV sorts devices by their unique ID."""
    try:
        raw  = subprocess.check_output(
            ['system_profiler', 'SPCameraDataType', '-json'],
            text=True, stderr=subprocess.DEVNULL,
        )
        cams = json.loads(raw).get('SPCameraDataType', [])
    except Exception:
        return []
    cams.sort(key=lambda c: c.get('spcamera_unique-id', ''))
    info = []
    for i, c in enumerate(cams):
        name = c.get('_name', f'Camera {i}')
        ident = f"{name} {c.get('spcamera_model-id', '')}".lower()
        info.append((name, 'iphone' in ident or 'ipad' in ident))   # Continuity Camera
    return info


def pick_camera(width=1280, height=720):
    """
    Scan for connected cameras, show a live preview thumbnail of each one,
    and let the user press the matching number key to choose. iPhone / iPad
    (Continuity) cameras are left out.

    Returns:
        cap      — cv2.VideoCapture, already opened and warmed up
        start_ms — int, epoch milliseconds at the moment the camera opened
                   (use as a timestamp base for mediapipe detect_for_video)

    Raises RuntimeError if no cameras are found or the chosen camera won't open.
    """
    THUMB_W, THUMB_H = 320, 240

    print("Scanning for cameras…")
    info = _camera_info()

    # Probe each index and grab a preview frame
    available = []   # list of (index, label)
    previews  = {}   # index -> BGR thumbnail
    for i in range(6):
        if i < len(info) and info[i][1]:
            print(f"Skipping phone camera: [{i}] {info[i][0]}")
            continue
        cap = cv2.VideoCapture(i, cv2.CAP_AVFOUNDATION)
        if not cap.isOpened():
            cap.release()
            continue
        # Skip the first few frames — they are often black while the sensor warms up
        for _ in range(8):
            ret, frame = cap.read()
        if ret and frame is not None:
            previews[i] = cv2.resize(frame, (THUMB_W, THUMB_H))
        label = info[i][0] if i < len(info) else f"Camera {i}"
        available.append((i, label))
        cap.release()

    if not available:
        raise RuntimeError(
            "No cameras found. Check System Settings → Privacy & Security → Camera."
        )

    # If only one camera, skip the chooser
    if len(available) == 1:
        chosen = available[0][0]
        print(f"One camera found: [{chosen}] {available[0][1]}\n")
    else:
        # Build a side-by-side thumbnail grid
        blank = np.zeros((THUMB_H, THUMB_W, 3), dtype=np.uint8)
        tiles = []
        for idx, label in available:
            tile = previews.get(idx, blank).copy()
            # Dark banner at top for the label
            cv2.rectangle(tile, (0, 0), (THUMB_W, 34), (0, 0, 0), -1)
            cv2.putText(tile, f"[{idx}] {label}", (8, 24),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 80), 2)
            # Instruction at bottom
            cv2.putText(tile, f"press  {idx}  to select",
                        (8, THUMB_H - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1)
            tiles.append(tile)

        grid = np.hstack(tiles)
        cv2.imshow("Select camera — press its number key", grid)

        valid_keys = {ord(str(idx)): idx for idx, _ in available}
        chosen = available[0][0]          # default to first
        while True:
            key = cv2.waitKey(0) & 0xFF
            if key in valid_keys:
                chosen = valid_keys[key]
                break
            elif key in (27, ord('q')):   # ESC or Q = accept default
                break

        cv2.destroyAllWindows()
        print(f"Using camera {chosen}\n")

    # Open the chosen camera for real
    cap = cv2.VideoCapture(chosen, cv2.CAP_AVFOUNDATION)
    if not cap.isOpened():
        cap = cv2.VideoCapture(chosen)
    if not cap.isOpened():
        raise RuntimeError(
            f"Could not open camera {chosen}. "
            "Check System Settings → Privacy & Security → Camera."
        )

    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    time.sleep(0.5)                  # let the sensor stabilise
    start_ms = int(time.time() * 1000)
    return cap, start_ms
