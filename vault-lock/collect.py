"""
Take training pictures with the USB webcam.

    python collect.py              # pick the webcam from thumbnails
    python collect.py --camera 1   # or say which one

Keys:
    1-5     you're going to show this many fingers
    0       "no number": empty scene, a fist, your hand flat/sideways, your arm...
    SPACE   record one short burst: 24 pictures in ~3 s, then it stops by itself
    q       quit

Each recording is saved to data/<count>/<recording>_<n>.png as the 64x64
image the net sees. train.py keeps whole recordings out for testing.

What matters most is how many DIFFERENT recordings there are, not how many
pictures: with 5 recordings per count the net memorises them and gets ~40% on
new ones; with 30 it gets ~95% (tested on synthetic hands). So do many short
bursts, and change something between every one: left/right hand, which
fingers, spread or together, rotation, height, where in the box, lighting,
sleeve up/down, a friend's hand. Aim for 20-30 bursts per count. For "0",
include anything that might be under the camera: nothing, a fist, a flat hand,
the back of your hand, your phone...
"""

import argparse
import glob
import os
import time

import cv2
import numpy as np

from webcam import open_webcam, HERE
import settings as S
import vision

DATA = os.path.join(HERE, "data")
SAVE_EVERY_S = 0.12
BURST = 24                # pictures per recording


def counts():
    return {lab: len(glob.glob(os.path.join(DATA, lab, "*.png"))) for lab in S.LABELS}


def main():
    ap = argparse.ArgumentParser(description="Take training pictures")
    ap.add_argument("--camera", type=int, help="camera number (skips the chooser)")
    args = ap.parse_args()

    cap = open_webcam(args.camera)
    label, recording, session, n, last = "1", False, None, 0, 0.0
    have = counts()
    print(__doc__)

    while True:
        ok, frame = cap.read()
        if not ok:
            continue
        small = vision.shrink(frame)
        now = time.time()
        if recording and now - last >= SAVE_EVERY_S:
            last = now
            os.makedirs(os.path.join(DATA, label), exist_ok=True)
            cv2.imwrite(os.path.join(DATA, label, f"{session}_{n:03d}.png"), small)
            n += 1
            have[label] += 1
            if n >= BURST:
                recording = False
                bursts = len({os.path.basename(p).rsplit("_", 1)[0]
                              for p in glob.glob(os.path.join(DATA, label, "*.png"))})
                print(f"saved {n} pictures of '{label}' ({bursts} recordings of it so far)"
                      " - change something before the next one")

        # Preview: camera with the square the net sees, plus that 64x64 image, enlarged.
        view = frame.copy()
        h, w = view.shape[:2]
        x0, y0, side = vision.crop_box(h, w)
        color = (0, 0, 255) if recording else (0, 200, 0)
        cv2.rectangle(view, (x0, y0), (x0 + side, y0 + side), color, 2)
        big = cv2.resize(small, (h // 2, h // 2), interpolation=cv2.INTER_NEAREST)
        panel = np.zeros((h, h // 2, 3), np.uint8)
        panel[:h // 2] = big
        cv2.putText(panel, "what the net sees", (8, h // 2 + 22), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        for i, lab in enumerate(S.LABELS):
            mark = ">" if lab == label else " "
            cv2.putText(panel, f"{mark} {lab}: {have[lab]}", (8, h // 2 + 50 + 22 * i),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255) if lab == label else (200, 200, 200), 1)
        what = "no number (0)" if label == "0" else f"{label} finger{'s' if label != '1' else ''}"
        state = f"REC {n}" if recording else "SPACE to record"
        cv2.putText(view, f"{what}   {state}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
        cv2.imshow("collect (q to quit)", np.hstack([view, panel]))

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        if key == ord(" ") and not recording:
            recording, session, n = True, time.strftime("%Y%m%d-%H%M%S"), 0
        elif chr(key) in S.LABELS and not recording:
            label = chr(key)

    cap.release()
    cv2.destroyAllWindows()
    print("pictures per count:", counts())
    print("(to delete a bad recording: rm data/<count>/<recording>_*.png)")


if __name__ == "__main__":
    main()
