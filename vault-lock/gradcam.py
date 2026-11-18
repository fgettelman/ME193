"""
Grad-CAM: show WHERE in the picture the net looked to decide its count.

    python gradcam.py            # results/gradcam.png: 4 test pictures per count
    python gradcam.py --wrong    # only the ones it got wrong (what fooled it?)

Red = these pixels pushed the score for the answer up. A good net lights up
the fingers. If it lights up the background, a sleeve or the table edge,
it's cheating, and you need more varied pictures (collect.py).

How it works is in model.gradcam(): it's backpropagation again, but instead of
asking "how should each weight change?" it asks "how much does each spot of
the last feature maps change the answer?".
"""

import argparse
import os

import cv2
import numpy as np
import torch

from model import FingerNet, gradcam, standardize, images_to_batch, HERE, S
from train import load_data, DATA, RESULTS

MODEL_PT = os.path.join(HERE, "models", "fingernet.pt")


def load_model(path=MODEL_PT):
    if not os.path.exists(path):
        raise SystemExit(f"No {os.path.relpath(path)} yet: run train.py first.")
    ck = torch.load(path, map_location="cpu")
    model = FingerNet(ck["width"], ck["depth"])
    model.load_state_dict(ck["state"])
    model.eval()
    return model


def overlay(small_bgr, heat, size=192):
    """64x64 BGR picture + 64x64 heat map (0..1) -> size x size picture."""
    im = cv2.resize(small_bgr, (size, size), interpolation=cv2.INTER_NEAREST)
    hm = cv2.applyColorMap((cv2.resize(heat, (size, size)) * 255).astype(np.uint8), cv2.COLORMAP_JET)
    return cv2.addWeighted(im, 0.55, hm, 0.45, 0)


def explain(model, smalls):
    """List of 64x64 BGR pictures -> (heat maps, predictions)."""
    x = standardize(images_to_batch(smalls))
    return gradcam(model, x)


def main():
    ap = argparse.ArgumentParser(description="Grad-CAM pictures of what the net looks at")
    ap.add_argument("--wrong", action="store_true", help="only show mistakes")
    ap.add_argument("--per-count", type=int, default=4)
    ap.add_argument("--data", default=DATA)
    args = ap.parse_args()

    model = load_model()
    _, val = load_data(args.data)
    if not val:
        raise SystemExit("No test pictures (need 2+ recordings per count).")
    heat, pred = explain(model, [im for im, _ in val])
    rng = np.random.default_rng(0)
    rows = []
    for k in range(S.N_CLASSES):
        idx = [i for i, (_, y) in enumerate(val) if y == k and (not args.wrong or pred[i] != y)]
        if not idx:
            continue
        idx = rng.choice(idx, min(args.per_count, len(idx)), replace=False)
        tiles = []
        for i in idx:
            t = overlay(val[i][0], heat[i])
            ok = pred[i] == k
            cv2.putText(t, f"is {k}, says {pred[i]}", (6, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                        (0, 255, 0) if ok else (0, 0, 255), 2)
            tiles.append(t)
        tiles += [np.zeros_like(tiles[0])] * (args.per_count - len(tiles))
        rows.append(np.hstack(tiles))
    if not rows:
        raise SystemExit("Nothing to show (no mistakes!).")
    os.makedirs(RESULTS, exist_ok=True)
    out = os.path.join(RESULTS, "gradcam_wrong.png" if args.wrong else "gradcam.png")
    cv2.imwrite(out, np.vstack(rows))
    print(f"Saved {os.path.relpath(out)}  (one row per count, red = what drove the answer)")


if __name__ == "__main__":
    main()
