"""
See what the UNO Q's lock is doing, on the laptop: its camera, the
instructions, the combination boxes, the LEDs and servo, and a live check
of the button's wiring. Same screen as laptop_lock.py, but everything comes
from the board.

    python board_view.py
    python board_view.py --host 10.247.137.130     # if the board's IP changed

    SPACE   press the safe's button from the laptop (as well as the real one on D2)
    h       hide/show the numbers in the combination boxes
    q       quit (the lock keeps running on the board)

It reads http://<board>:8040 (served by unoq_app/python/main.py). If the
board doesn't let the laptop reach that port directly, it goes through SSH
instead (needs the SSH key from deploy.py's notes).
"""

import argparse
import json
import subprocess
import time
import urllib.request

import cv2
import numpy as np

from deploy import BOARD_HOST, CONTAINER
import laptop_lock as LL
import lock as L
import settings as S

PORT = 8040
TUNNEL_PORT = 18040


def get(url, timeout=1.5):
    with urllib.request.urlopen(url, timeout=timeout) as r:
        return r.read()


def connect(host):
    """Base URL that reaches the board's view server, and the SSH tunnel (if any)."""
    direct = f"http://{host}:{PORT}"
    try:
        get(direct + "/status.json")
        print(f"Connected to {direct}")
        return direct, None
    except Exception:
        pass
    print("Can't reach the board's port directly; going through SSH...")
    ip = subprocess.run(
        ["ssh", "-o", "BatchMode=yes", f"arduino@{host}",
         f"docker inspect -f '{{{{range .NetworkSettings.Networks}}}}{{{{.IPAddress}}}}{{{{end}}}}' {CONTAINER}"],
        capture_output=True, text=True, timeout=20).stdout.strip()
    if not ip:
        raise SystemExit("The lock app isn't running on the board (python deploy.py --start).")
    tunnel = subprocess.Popen(["ssh", "-o", "BatchMode=yes", "-o", "ExitOnForwardFailure=yes", "-N",
                               "-L", f"{TUNNEL_PORT}:{ip}:{PORT}", f"arduino@{host}"])
    url = f"http://127.0.0.1:{TUNNEL_PORT}"
    for _ in range(40):
        try:
            get(url + "/status.json")
            print(f"Connected through SSH ({url} -> board container {ip}:{PORT})")
            return url, tunnel
        except Exception:
            time.sleep(0.25)
    tunnel.terminate()
    raise SystemExit("Couldn't reach the lock's view server. Is the new version deployed? (python deploy.py)")


def message_screen(text1, text2, h=S.CAM_H, w=S.CAM_W + LL.PANEL_W):
    img = np.full((h, w, 3), 25, np.uint8)
    LL.centred(img, text1, w // 2, h // 2 - 10, 1.0, LL.WHITE, 2)
    LL.centred(img, text2, w // 2, h // 2 + 30, 0.55, LL.GREY, 1)
    return img


def button_line(view, b):
    """The button's wiring check, at the bottom of the camera picture."""
    h = view.shape[0]
    if b.get("pressed_now") is None:
        txt, col = "Button D2: no answer from the sketch", LL.RED
    elif b["pressed_now"]:
        txt, col = f"Button D2: PRESSED  ({b['presses_since_power_on']} presses counted)", LL.GREEN
    else:
        txt, col = f"Button D2: released  ({b['presses_since_power_on']} presses counted)", LL.WHITE
    cv2.rectangle(view, (0, h - 58), (view.shape[1], h - 30), (0, 0, 0), -1)
    LL.text(view, txt, (10, h - 38), 0.55, col, 1)


def main():
    ap = argparse.ArgumentParser(description="Watch the UNO Q lock from the laptop")
    ap.add_argument("--host", default=BOARD_HOST)
    args = ap.parse_args()

    base, tunnel = connect(args.host)
    hide = False
    flash = None
    prev = (L.LOCKED, 0)
    frame = np.zeros((S.CAM_H, S.CAM_W, 3), np.uint8)
    last_ok = time.monotonic()
    print(__doc__)
    try:
        while True:
            now = time.monotonic()
            key = cv2.waitKey(30) & 0xFF
            if key == ord("q"):
                break
            if key == ord("h"):
                hide = not hide
            try:
                if key == ord(" "):
                    urllib.request.urlopen(urllib.request.Request(base + "/press", data=b"", method="POST"),
                                           timeout=1.5).read()
                st = json.loads(get(base + "/status.json"))
                jpg = get(base + "/frame.jpg")
                f = cv2.imdecode(np.frombuffer(jpg, np.uint8), cv2.IMREAD_COLOR)
                if f is not None:
                    frame = f
                last_ok = now
            except Exception as e:
                if now - last_ok > 3:
                    cv2.imshow("Vault lock (UNO Q)", message_screen(
                        "Lost the board", f"{type(e).__name__}: retrying... (is the app running? deploy.py --logs)"))
                continue

            if st.get("mode") != "running":
                cv2.imshow("Vault lock (UNO Q)", message_screen("The board is starting...", "waiting for its camera"))
                continue

            dd = st["display"]
            dd["digits"] = tuple(dd["digits"])
            d = L.Display(**dd)
            probs = np.array(st["probs"])
            # The same short messages as the laptop version.
            if prev[0] == L.LOCKED and d.mode == L.SCANNING:
                flash = ("GO!", now + 1.2, f"Show your first number and hold it still for {S.HOLD_S:.1f} s")
            elif d.mode == L.SCANNING and len(d.digits) > prev[1]:
                flash = (f"Got it: {'*' if hide else d.digits[-1]}", now + 1.2,
                         f"Number {len(d.digits)} of {S.CODE_LENGTH} entered - now change to the next one")
            elif d.mode != L.SCANNING:
                flash = None
            prev = (d.mode, len(d.digits))

            server_now = st["time"]
            events = [(now - (server_now - t), time.strftime("%H:%M:%S", time.localtime(t)) + "  " + e)
                      for t, e in st["events"]]

            view = cv2.resize(frame, (S.CAM_W, S.CAM_H)) if frame.shape[:2] != (S.CAM_H, S.CAM_W) else frame.copy()
            x0, y0, side = LL.vision.crop_box(S.CAM_H, S.CAM_W)
            cv2.rectangle(view, (x0, y0), (x0 + side, y0 + side), (0, 200, 0), 2)
            LL.camera_banner(view, d, now, flash)
            button_line(view, st.get("button", {}))
            LL.text(view, f"UNO Q: {st.get('fps', 0):.0f} frames/s   SPACE button   h hide   q quit",
                    (10, S.CAM_H - 12), 0.45, LL.YELLOW, 1)
            cv2.imshow("Vault lock (UNO Q)", np.hstack([view, LL.panel(d, probs, now, S.CAM_H, events, hide)]))
    finally:
        cv2.destroyAllWindows()
        if tunnel:
            tunnel.terminate()


if __name__ == "__main__":
    main()
