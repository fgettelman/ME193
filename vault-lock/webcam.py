"""
Open the USB webcam on the laptop (not the built-in FaceTime camera or an
iPhone via Continuity Camera) at the same 640x480 the board uses.

    from webcam import open_webcam
    cap = open_webcam()          # shows thumbnails of every camera, press its number
    cap = open_webcam(index=1)   # skip the chooser once you know the number
"""

import os
import sys
import time

import cv2

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "unoq_app", "python"))
sys.path.insert(0, os.path.join(HERE, "..", "class"))

import settings as S       # noqa: E402


def open_webcam(index=None):
    if index is None:
        try:
            from camlib import pick_camera
        except ImportError:
            raise SystemExit("Couldn't find class/camlib.py; pass --camera N instead.")
        print("Pick the USB webcam, NOT the laptop's FaceTime camera or your phone.")
        cap, _ = pick_camera(width=S.CAM_W, height=S.CAM_H)
    else:
        cap = cv2.VideoCapture(index, cv2.CAP_AVFOUNDATION)
        if not cap.isOpened():
            cap = cv2.VideoCapture(index)
        if not cap.isOpened():
            raise SystemExit(f"Couldn't open camera {index}.")
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, S.CAM_W)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, S.CAM_H)
    # macOS webcams often give empty frames for the first moment: keep trying.
    deadline = time.time() + 3.0
    ok, frame = cap.read()
    while not ok and time.time() < deadline:
        time.sleep(0.05)
        ok, frame = cap.read()
    if not ok:
        raise SystemExit("The camera opened but gives no pictures (is another app using it?).")
    h, w = frame.shape[:2]
    if (w, h) != (S.CAM_W, S.CAM_H):
        print(f"Note: the camera gives {w}x{h}, not {S.CAM_W}x{S.CAM_H}. The centre square "
              "is still used, but check that the board's view looks the same.")
    return cap
