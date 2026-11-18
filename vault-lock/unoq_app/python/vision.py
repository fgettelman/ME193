"""
Camera frame -> what the net sees. Used by collect.py on the laptop, by
training (images are saved already cropped and shrunk) and by the board, so
every image goes through exactly the same steps.

    frame (480x640x3, BGR)
      -> centre square (480x480)          the box drawn on the laptop preview
      -> shrink to 64x64 (area averaging)
      -> net input: (3, 64, 64) float, each channel standardized on its own
         (mean 0, std 1). That throws away overall brightness and colour cast,
         so the net can't rely on the lighting, only on shapes.
"""

import numpy as np

import settings as S

try:
    import cv2
except ImportError:          # the board's container should have it, but just in case
    cv2 = None


def crop_box(h, w):
    """(x0, y0, side) of the centre square of an h x w frame."""
    side = min(h, w)
    return (w - side) // 2, (h - side) // 2, side


def shrink(frame):
    """Camera frame (H, W, 3) uint8 BGR -> centre square at IMG x IMG, uint8 BGR."""
    h, w = frame.shape[:2]
    x0, y0, side = crop_box(h, w)
    sq = frame[y0:y0 + side, x0:x0 + side]
    if S.SWAP_RB:
        sq = sq[:, :, ::-1]
    if cv2 is not None:
        return cv2.resize(sq, (S.IMG, S.IMG), interpolation=cv2.INTER_AREA)
    # No OpenCV: average blocks of pixels (close to INTER_AREA).
    edges = np.linspace(0, side, S.IMG + 1).astype(int)
    rows = np.add.reduceat(sq.astype(np.float32), edges[:-1], axis=0)
    rows /= np.diff(edges)[:, None, None]
    out = np.add.reduceat(rows, edges[:-1], axis=1) / np.diff(edges)[None, :, None]
    return out.round().astype(np.uint8)


def to_input(small):
    """IMG x IMG x 3 uint8 image -> (3, IMG, IMG) float32 net input."""
    x = small.astype(np.float32).transpose(2, 0, 1) / 255.0
    mean = x.mean(axis=(1, 2), keepdims=True)
    std = x.std(axis=(1, 2), keepdims=True) + 0.02
    return (x - mean) / std
