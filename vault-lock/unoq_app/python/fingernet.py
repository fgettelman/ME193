"""
The finger-counting CNN with plain numpy, so it runs on the UNO Q without
PyTorch. It's trained in PyTorch on the laptop (model.py, train.py) and saved
as a list of layers in models/fingernet.json; train.py checks that this file
gives the same answers as PyTorch before it saves it.

The JSON's "layers" is a list, run in order:
    {"type": "conv", "w": (out, in, k, k), "b": (out,), "pad": p}
    {"type": "relu"}
    {"type": "maxpool", "k": 2}
    {"type": "gap"}                 global average pool: (C, H, W) -> (C,)
    {"type": "linear", "w": (out, in), "b": (out,)}
"""

import json

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

import settings as S
import vision


def conv2d(x, w, b, pad):
    """x: (C_in, H, W), w: (C_out, C_in, k, k) -> (C_out, H', W'), like
    torch's Conv2d (which is really cross-correlation: no kernel flip).

    Every output pixel is a dot product between the kernel and the k x k patch
    under it, over all input channels. sliding_window_view lists the patches
    without copying them; tensordot does all the dot products at once."""
    if pad:
        x = np.pad(x, ((0, 0), (pad, pad), (pad, pad)))
    k = w.shape[2]
    patches = sliding_window_view(x, (k, k), axis=(1, 2))     # (C_in, H', W', k, k)
    out = np.tensordot(w, patches, axes=([1, 2, 3], [0, 3, 4]))  # (C_out, H', W')
    return out + b[:, None, None]


def maxpool(x, k):
    """(C, H, W) -> (C, H//k, W//k): keep the biggest value in each k x k block."""
    c, h, w = x.shape
    h, w = h // k * k, w // k * k
    return x[:, :h, :w].reshape(c, h // k, k, w // k, k).max(axis=(2, 4))


def softmax(z):
    e = np.exp(z - z.max())
    return e / e.sum()


class FingerNet:
    def __init__(self, spec):
        self.spec = spec
        self.layers = []
        for L in spec["layers"]:
            L = dict(L)
            for key in ("w", "b"):
                if key in L:
                    L[key] = np.asarray(L[key], dtype=np.float32)
            self.layers.append(L)

    def logits(self, x):
        """x: (3, IMG, IMG) net input -> (N_CLASSES,) scores."""
        for L in self.layers:
            t = L["type"]
            if t == "conv":
                x = conv2d(x, L["w"], L["b"], L["pad"])
            elif t == "relu":
                x = np.maximum(x, 0)
            elif t == "maxpool":
                x = maxpool(x, L["k"])
            elif t == "gap":
                x = x.mean(axis=(1, 2))
            elif t == "linear":
                x = L["w"] @ x + L["b"]
            else:
                raise ValueError(f"unknown layer type {t!r}")
        return x

    def probs_small(self, small):
        """64x64x3 uint8 BGR image -> probabilities for 0..5 fingers."""
        return softmax(self.logits(vision.to_input(small)))

    def probs(self, frame):
        """Camera frame -> probabilities for 0..5 fingers."""
        return self.probs_small(vision.shrink(frame))


def load(path):
    with open(path) as f:
        spec = json.load(f)
    if spec.get("labels") != S.LABELS:
        raise ValueError(f"{path} was trained for labels {spec.get('labels')}, not {S.LABELS}")
    return FingerNet(spec)
