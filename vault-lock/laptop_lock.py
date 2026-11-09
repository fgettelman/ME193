"""
The whole lock on the laptop with the USB webcam: step 1, before the UNO Q.

    python laptop_lock.py              # pick the webcam from thumbnails
    python laptop_lock.py --camera 0

    SPACE   the safe's button
    g       Grad-CAM on/off: see what the net is looking at, live
    h       hide/show the numbers in the combination boxes (shoulder-surfers)
    q       quit

It runs the same files the board runs (unoq_app/python/lock.py for the logic,
fingernet.py for the net, with the exported numpy model), so if it works here
it works on the safe. The panel on the right shows what's happening: which
number of the combination you're on, the numbers entered so far, what the
camera sees, and the LEDs and servo the UNO Q will drive.
"""

import argparse
import math
import os
import time
from collections import deque

import cv2
import numpy as np

from webcam import open_webcam, HERE
import settings as S
import vision
import fingernet
import lock as L

BOARD_MODEL = os.path.join(HERE, "unoq_app", "python", "models", "fingernet.json")
FONT = cv2.FONT_HERSHEY_SIMPLEX
PANEL_W = 460

# BGR
WHITE, GREY, DIM, DARK = (255, 255, 255), (170, 170, 170), (90, 90, 90), (32, 32, 32)
YELLOW, GREEN, RED, BLUE = (0, 210, 255), (0, 210, 0), (0, 0, 255), (255, 170, 0)

BANNER = {L.LOCKED: ("LOCKED", (70, 70, 70)), L.SCANNING: ("ENTER THE CODE", (150, 110, 0)),
          L.UNLOCKED: ("UNLOCKED", (0, 150, 0)), L.DENIED: ("WRONG CODE", (0, 0, 190)),
          L.LOCKOUT: ("LOCKED OUT", (0, 0, 150))}
LED_COLORS = {L.LOCKED: DIM, L.SCANNING: WHITE, L.UNLOCKED: GREEN, L.DENIED: RED, L.LOCKOUT: RED}


def text(img, s, org, scale=0.6, color=WHITE, thick=1):
    cv2.putText(img, s, org, FONT, scale, color, thick, cv2.LINE_AA)


def centred(img, s, cx, y, scale, color, thick=2):
    (w, _), _ = cv2.getTextSize(s, FONT, scale, thick)
    text(img, s, (cx - w // 2, y), scale, color, thick)


def camera_banner(view, d, now, flash):
    """Big instructions across the top of the camera view, where the user is
    looking. flash = (text, until) shown for a moment after something happens."""
    h, w = view.shape[:2]
    n = len(d.digits)
    sub, bar, colour = "", None, WHITE
    if flash and now < flash[1]:
        big, sub, colour = flash[0], flash[2], GREEN
    elif d.mode == L.LOCKED:
        big = "Press SPACE to start"
        sub = f"Then show {S.CODE_LENGTH} numbers with your fingers, holding each still for {S.HOLD_S:.1f} s"
        colour = YELLOW if int(now * 1.5) % 2 else WHITE          # gently blinking
    elif d.mode == L.SCANNING:
        big = f"Show number {n + 1} of {S.CODE_LENGTH}"
        if d.hold > 0:
            left = S.HOLD_S * (1 - d.hold)
            sub, bar = f"Holding {d.live}... keep still  {left:.1f} s", d.hold
            colour = YELLOW
        elif n:
            sub = f"Change your fingers to number {n + 1}, then hold still {S.HOLD_S:.1f} s"
        else:
            sub = f"Fingers in the green box, hold still {S.HOLD_S:.1f} s"
        sub += f"   ({d.seconds} s left)" if d.hold == 0 else ""
    elif d.mode == L.UNLOCKED:
        big, sub, colour = "OPEN!", f"Press SPACE to lock again (locks itself in {d.seconds} s)", GREEN
    elif d.mode == L.DENIED:
        big, sub, colour = "Wrong code", f"Still locked. Try {d.fails} of {S.MAX_TRIES}: press SPACE to try again", RED
    else:
        big, sub, colour = f"Locked out: wait {d.seconds} s", "Too many wrong codes. The button does nothing until then.", RED

    overlay = view.copy()
    cv2.rectangle(overlay, (0, 0), (w, 92), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.6, view, 0.4, 0, view)
    centred(view, big, w // 2, 44, 1.2, colour, 3)
    centred(view, sub, w // 2, 76, 0.5, WHITE, 1)
    if bar is not None:
        cv2.rectangle(view, (0, 88), (int(w * bar), 92), YELLOW, -1)


def instructions(d, seen_k, seen_p):
    """Two lines telling the user what to do now."""
    n = len(d.digits)
    if d.mode == L.LOCKED:
        return "Press SPACE (the safe's button) to start,", f"then hold each number still for {S.HOLD_S:.1f} s."
    if d.mode == L.SCANNING:
        if n == 0:
            first = f"Show number 1 of {S.CODE_LENGTH} with your fingers"
        else:
            first = f"Got {d.digits[-1]}! Now show number {n + 1} of {S.CODE_LENGTH}"
        if d.hold > 0:
            return first, f"Holding {d.live}... keep it still"
        if n and seen_k == d.digits[-1] and seen_p >= S.CONF_MIN:
            return first, f"Change to a DIFFERENT number (not {d.digits[-1]})"
        return first, f"Hold it in the green box ({S.HOLD_S:.1f} s). {d.seconds} s left"
    if d.mode == L.UNLOCKED:
        return "Correct code: the safe is open.", f"Relocks in {d.seconds} s, or press SPACE."
    if d.mode == L.DENIED:
        return "That code is wrong. Still locked.", f"Wrong tries: {d.fails} of {S.MAX_TRIES}"
    if d.mode == L.LOCKOUT:
        return "Too many wrong codes.", f"Wait {d.seconds} s (the button is ignored)."
    return "", ""


def panel(d, probs, now, h, events, hide_digits=False, w=PANEL_W):
    p = np.full((h, w, 3), DARK, np.uint8)
    cx = w // 2

    # 1. Banner: the state, in its colour.
    label, bg = BANNER[d.mode]
    cv2.rectangle(p, (0, 0), (w, 52), bg, -1)
    centred(p, label, cx, 36, 1.0, WHITE, 2)

    # 2. What to do now.
    seen_k = int(np.argmax(probs)) if probs is not None else 0
    seen_p = float(probs[seen_k]) if probs is not None else 0.0
    a, b = instructions(d, seen_k, seen_p)
    text(p, a, (14, 82), 0.58, WHITE, 1)
    text(p, b, (14, 106), 0.52, GREY, 1)

    # 3. The combination: one box per number, the current one highlighted.
    box, gap = 90, 22
    x0 = cx - (S.CODE_LENGTH * box + (S.CODE_LENGTH - 1) * gap) // 2
    y0 = 124
    current = len(d.digits) if d.mode == L.SCANNING else -1
    for i in range(S.CODE_LENGTH):
        x = x0 + i * (box + gap)
        decided = d.mode in (L.UNLOCKED, L.DENIED)     # a full code went in, but it's never shown
        filled = decided or i < len(d.digits)
        if d.mode == L.UNLOCKED:
            border = GREEN
        elif d.mode == L.DENIED:
            border = RED
        elif i == current:
            border = YELLOW
        else:
            border = DIM
        cv2.rectangle(p, (x, y0), (x + box, y0 + box), (50, 50, 50), -1)
        cv2.rectangle(p, (x, y0), (x + box, y0 + box), border, 4 if i == current else 2)
        if filled:
            masked = decided or hide_digits
            centred(p, "*" if masked else str(d.digits[i]), x + box // 2, y0 + 66, 2.0, WHITE, 4)
        elif i == current and d.live:
            # The number being held, faint, with a bar filling up underneath.
            centred(p, str(d.live), x + box // 2, y0 + 66, 2.0, (110, 110, 110), 3)
        if i == current:
            cv2.rectangle(p, (x, y0 + box + 6), (x + int(box * d.hold), y0 + box + 14), YELLOW, -1)
            cv2.rectangle(p, (x, y0 + box + 6), (x + box, y0 + box + 14), DIM, 1)
        centred(p, f"#{i + 1}", x + box // 2, y0 + box + 34, 0.5, YELLOW if i == current else GREY, 1)

    # 4. What the camera sees right now.
    y = 300
    if probs is not None:
        sure = seen_p >= S.CONF_MIN
        what = "nothing / not a number" if seen_k == 0 else f"{seen_k} finger{'s' if seen_k > 1 else ''}"
        text(p, "Camera sees:", (14, y), 0.55, GREY, 1)
        text(p, f"{what}  ({seen_p:.0%})", (130, y), 0.6, GREEN if sure else GREY, 2 if sure else 1)
        if not sure:
            text(p, "(not sure enough)", (130, y + 20), 0.45, GREY, 1)
        for k in range(S.N_CLASSES):           # small probability bars
            bx = 14 + k * 72
            bh = int(36 * probs[k])
            cv2.rectangle(p, (bx, y + 70 - bh), (bx + 56, y + 70), GREEN if probs[k] >= S.CONF_MIN else DIM, -1)
            centred(p, S.LABELS[k], bx + 28, y + 88, 0.45, GREY, 1)

    # 5. The UNO Q's outputs: RGB LEDs and servo.
    y = 410
    led = LED_COLORS[d.mode]
    if d.mode == L.LOCKOUT and int(now * 2) % 2:
        led = DIM
    text(p, "LEDs", (14, y + 5), 0.5, GREY, 1)
    for i in range(2):
        cv2.circle(p, (75 + 36 * i, y), 12, led, -1)
    sx, sy, r = 300, y + 8, 30
    a = math.radians(d.servo_deg)
    text(p, f"servo {d.servo_deg} deg", (190, y + 5), 0.5, GREY, 1)
    cv2.circle(p, (sx + 40, sy), 7, GREY, -1)
    cv2.line(p, (sx + 40, sy), (int(sx + 40 + r * math.cos(a)), int(sy - r * math.sin(a))), YELLOW, 5)

    # 6. What just happened.
    y = 455
    for i, (t, e) in enumerate(list(events)[-3:]):
        age = now - t
        text(p, e, (14, y + 18 * i), 0.45, WHITE if age < 3 else GREY, 1)
    return p


def main():
    ap = argparse.ArgumentParser(description="The finger-code lock, on the laptop")
    ap.add_argument("--camera", type=int, help="camera number (skips the chooser)")
    args = ap.parse_args()

    if not os.path.exists(BOARD_MODEL):
        raise SystemExit("No trained model yet: run collect.py, then train.py.")
    net = fingernet.load(BOARD_MODEL)
    check, is_default = L.load_checker()
    if is_default:
        print(f"No code set: using {'-'.join(map(str, L.DEFAULT_CODE))}. Set yours with set_code.py.")
    lk = L.Lock(check)
    events = deque(maxlen=6)

    torch_model = None            # loaded the first time you press g
    show_cam, hide_digits = False, False
    flash = None
    cap = open_webcam(args.camera)
    print(__doc__)
    fps_t, fps = time.time(), 0.0

    while True:
        ok, frame = cap.read()
        if not ok:
            continue
        now = time.monotonic()
        presses = 0
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        if key == ord(" "):
            presses = 1
        if key == ord("g"):
            show_cam = not show_cam
        if key == ord("h"):
            hide_digits = not hide_digits

        small = vision.shrink(frame)
        before = (lk.mode, len(lk.entered))
        d = lk.step(now, presses, net.probs_small(small))
        # A short message right after something happens.
        if before[0] == L.LOCKED and d.mode == L.SCANNING:
            flash = ("GO!", now + 1.2, f"Show your first number and hold it still for {S.HOLD_S:.1f} s")
        elif d.mode == L.SCANNING and len(d.digits) > before[1]:
            flash = (f"Got it: {'*' if hide_digits else d.digits[-1]}", now + 1.2,
                     f"Number {len(d.digits)} of {S.CODE_LENGTH} entered - now change to the next one")
        elif d.mode != L.SCANNING:
            flash = None
        for e in lk.events:
            print(time.strftime("%H:%M:%S"), e)
            events.append((now, f"{time.strftime('%H:%M:%S')}  {e}"))
        lk.events.clear()

        view = frame.copy()
        hh, ww = view.shape[:2]
        x0, y0, side = vision.crop_box(hh, ww)
        if show_cam:
            if torch_model is None:
                from gradcam import load_model, explain, overlay
                torch_model = load_model()
            heat, _ = explain(torch_model, [small])
            view[y0:y0 + side, x0:x0 + side] = overlay(small, heat[0], side)
        cv2.rectangle(view, (x0, y0), (x0 + side, y0 + side), (0, 200, 0), 2)
        camera_banner(view, d, now, flash)
        dt = time.time() - fps_t
        fps_t, fps = time.time(), 0.9 * fps + 0.1 / max(dt, 1e-3)
        text(view, f"{fps:.0f} fps   SPACE button   g heat map ({'on' if show_cam else 'off'})   h hide   q quit",
             (10, hh - 12), 0.45, YELLOW, 1)
        cv2.imshow("Vault lock", np.hstack([view, panel(d, lk.smoothed, now, hh, events, hide_digits)]))

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
