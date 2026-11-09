"""
Finger-code safe, UNO Q Linux side.

Every frame from the USB webcam:
    camera frame -> vision.shrink (centre square, 64x64) -> FingerNet (numpy)
    -> probabilities for 0..5 fingers -> lock.step() -> what to show
    -> Bridge.call("show", ...) -> the sketch drives the LEDs, matrix and servo

Button presses come from the sketch: Bridge.call("presses").

For the laptop (board_view.py) it also serves, on port 8040:
    /frame.jpg     the latest camera picture
    /status.json   the lock's state, what the net sees, recent events, the button wiring
    /press         (POST) a press of the button, from the laptop's space bar

The same lock.py and fingernet.py ran on the laptop (laptop_lock.py) first.
"""

import glob
import json
import os
import sys
import threading
import time
from collections import deque
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import numpy as np
from arduino.app_utils import App, Bridge

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import settings as S                      # noqa: E402
import fingernet                          # noqa: E402
import lock as L                          # noqa: E402

try:
    import cv2
except ImportError:
    cv2 = None

MODEL_PATH = os.path.join(HERE, "models", "fingernet.json")
PEEK_PATH = os.path.join(HERE, "peek.jpg")
VIEW_PORT = 8040


# --- Camera ---------------------------------------------------------------------

class Webcam:
    """The USB webcam (through a USB-C hub on the UNO Q). OpenCV first; if
    that can't open it, Arduino's own Camera class."""

    def __init__(self):
        self.cap = None
        self.arduino_cam = None
        self.name = None

    def open(self):
        if cv2 is not None:
            # USB webcams by name first (the board also has non-camera video
            # devices, e.g. its hardware video codec), then anything else.
            devices = (sorted(glob.glob("/dev/v4l/by-id/usb-*-video-index0"))
                       + sorted(glob.glob("/dev/video*"))) or [0]
            for dev in devices:
                cap = cv2.VideoCapture(dev, cv2.CAP_V4L2) if isinstance(dev, str) else cv2.VideoCapture(dev)
                if not cap.isOpened():
                    cap.release()
                    continue
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, S.CAM_W)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, S.CAM_H)
                cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                ok, frame = cap.read()
                if ok and frame is not None:
                    self.cap, self.name = cap, f"OpenCV {dev} {frame.shape[1]}x{frame.shape[0]}"
                    return True
                cap.release()
        try:
            from arduino.app_peripherals.camera import Camera
            cam = Camera(resolution=(S.CAM_W, S.CAM_H), fps=15)
            cam.start()
            if cam.capture() is not None:
                self.arduino_cam, self.name = cam, "Arduino Camera"
                return True
            cam.stop()
        except Exception as e:
            print(f"Arduino Camera didn't work either: {e}")
        return False

    def read(self):
        if self.cap is not None:
            ok, frame = self.cap.read()
            return frame if ok else None
        if self.arduino_cam is not None:
            frame = self.arduino_cam.capture(latest=True)
            return None if frame is None else np.asarray(frame)
        return None

    def close(self):
        if self.cap is not None:
            self.cap.release()
        if self.arduino_cam is not None:
            self.arduino_cam.stop()
        self.cap = self.arduino_cam = None


# --- Shared with the laptop viewer -------------------------------------------------

class Shared:
    """What the web server hands to board_view.py. Written by loop(), read by
    the server's threads; the lock keeps them from seeing half an update."""
    lock = threading.Lock()
    frame = None                 # latest camera frame (BGR)
    status = {"mode": "starting"}
    events = deque(maxlen=8)     # (time.time(), text)
    remote_presses = 0           # space bar presses from the laptop


def log(text):
    print(time.strftime("%H:%M:%S"), text, flush=True)
    with Shared.lock:
        Shared.events.append((time.time(), text))


class ViewHandler(BaseHTTPRequestHandler):
    def _send(self, body, ctype):
        self.send_response(200)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path.startswith("/frame.jpg"):
            with Shared.lock:
                frame = Shared.frame
            if frame is None or cv2 is None:
                self.send_error(503, "no camera picture yet")
                return
            ok, jpg = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 75])
            self._send(jpg.tobytes(), "image/jpeg")
        elif self.path.startswith("/status.json"):
            with Shared.lock:
                st = dict(Shared.status)
                st["events"] = [[t, e] for t, e in Shared.events]
            st["time"] = time.time()
            self._send(json.dumps(st).encode(), "application/json")
        else:
            self.send_error(404)

    def do_POST(self):
        if self.path.startswith("/press"):
            with Shared.lock:
                Shared.remote_presses += 1
            self._send(b"ok", "text/plain")
        else:
            self.send_error(404)

    def log_message(self, *args):        # keep the app's log readable
        pass


def start_server():
    try:
        server = ThreadingHTTPServer(("0.0.0.0", VIEW_PORT), ViewHandler)
    except OSError as e:
        print(f"Laptop view server didn't start: {e}")
        return
    threading.Thread(target=server.serve_forever, daemon=True).start()
    print(f"Laptop view: port {VIEW_PORT} (python board_view.py on the laptop)")


# --- Setup ------------------------------------------------------------------------

net = fingernet.load(MODEL_PATH)
print(f"Model: width {net.spec['width']}, depth {net.spec['depth']}, "
      f"test accuracy {net.spec.get('val_acc', '?')}")
check, is_default = L.load_checker()
if is_default:
    print(f"No code.json: the code is {'-'.join(map(str, L.DEFAULT_CODE))} (set yours with set_code.py)")
lk = L.Lock(check)
cam = Webcam()
start_server()


class Stats:
    frames = 0
    t0 = time.monotonic()
    infer_s = 0.0
    fps = 0.0
    next_peek = 0.0
    next_cam_try = 0.0
    next_diag = 0.0
    button = {"pressed_now": None, "presses_since_power_on": None}


st = Stats()


def show(display):
    mode, arg = display.code_for_sketch()
    Bridge.call("show", mode * 1000 + min(arg, 999))


def read_button_diag():
    """Raw D2 state and total presses, straight from the sketch (wiring check)."""
    try:
        v = int(Bridge.call("button_diag"))
        st.button = {"pressed_now": v >= 1000, "presses_since_power_on": v % 1000}
    except Exception as e:
        st.button = {"pressed_now": None, "presses_since_power_on": None, "error": str(e)}


def loop():
    now = time.monotonic()
    if cam.cap is None and cam.arduino_cam is None:
        if now < st.next_cam_try:
            time.sleep(0.2)
            return                      # no "show": the sketch shows "?" after 5 s
        st.next_cam_try = now + 5
        if not cam.open():
            print("No camera found (is the webcam on the USB-C hub? ls /dev/video*). Trying again in 5 s.")
            return
        print(f"Camera: {cam.name}")

    frame = cam.read()
    if frame is None:
        print("Camera stopped giving pictures; reopening.")
        cam.close()
        return

    try:
        presses = int(Bridge.call("presses") or 0)
    except Exception as e:
        print(f"bridge error: {e}")
        presses = 0
    with Shared.lock:
        remote, Shared.remote_presses = Shared.remote_presses, 0
    if presses:
        log(f"button pressed (D2) x{presses}")
    if remote:
        log("button pressed (laptop space bar)")
    if now >= st.next_diag:
        st.next_diag = now + 0.25
        read_button_diag()

    t = time.monotonic()
    probs = net.probs(frame)
    st.infer_s += time.monotonic() - t

    display = lk.step(time.monotonic(), presses + remote, probs)
    for e in lk.events:
        log(e)
    lk.events.clear()
    try:
        show(display)
    except Exception as e:
        print(f"bridge error: {e}")

    d = display.__dict__.copy()
    d["digits"] = list(d["digits"])
    with Shared.lock:
        Shared.frame = frame
        Shared.status = {"mode": "running", "display": d, "probs": lk.smoothed.round(4).tolist(),
                         "fps": round(st.fps, 1), "button": st.button, "camera": cam.name,
                         "hold_s": S.HOLD_S, "code_length": S.CODE_LENGTH}

    st.frames += 1
    if now >= st.next_peek and cv2 is not None:
        st.next_peek = now + 2
        cv2.imwrite(PEEK_PATH, frame)
    elapsed = now - st.t0
    if elapsed >= 15:
        st.fps = st.frames / elapsed
        sees = int(np.argmax(lk.smoothed))
        print(f"{st.fps:.1f} frames/s, net {st.infer_s / max(st.frames, 1) * 1000:.0f} ms/frame; "
              f"mode {L.MODE_NAMES[display.mode]}; sees {sees} ({lk.smoothed[sees]:.0%}); "
              f"button D2 {'PRESSED' if st.button.get('pressed_now') else 'released'}, "
              f"{st.button.get('presses_since_power_on')} presses since power-on", flush=True)
        st.frames, st.infer_s, st.t0 = 0, 0.0, now


App.run(user_loop=loop)
