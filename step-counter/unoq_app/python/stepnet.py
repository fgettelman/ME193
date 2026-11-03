"""
The step-counting neural net, run with plain numpy so it works on the UNO Q.
It's trained with PyTorch on the laptop (net.py, train.py) and saved as JSON
(models/step_model.json); net.py checks that this file gives the same answers
as PyTorch.

For a 2 s window of IMU samples the net outputs:
  - how many steps happened in the window (a fraction, e.g. 3.4), and
  - which activity it is (still / walk / run), with probabilities.

How it counts: three 1D convolution layers (with ReLU) turn the window into
50 "step density" values, one per 40 ms, each >= 0 because of a final ReLU.
Their sum is the number of steps in the window. So the net learns to light up
where a step happens, and counting is just adding up.
"""

import json

import numpy as np

from settings import LABELS


def inputs(window):
    """(N, 6) window in g and deg/s -> (8, N) net input: the 6 raw channels
    plus the size of the acceleration and of the rotation."""
    w = np.asarray(window, dtype=np.float32)
    acc_mag = np.linalg.norm(w[:, :3], axis=1, keepdims=True)
    gyro_mag = np.linalg.norm(w[:, 3:], axis=1, keepdims=True)
    return np.concatenate([w, acc_mag, gyro_mag], axis=1).T


def conv1d(x, w, b, pad, dil=1):
    """x: (C_in, L), w: (C_out, C_in, K) -> (C_out, L_out), like torch's Conv1d."""
    x = np.pad(x, ((0, 0), (pad, pad)))
    k = w.shape[2]
    n = x.shape[1] - dil * (k - 1)
    out = np.zeros((w.shape[0], n), dtype=np.float32)
    for j in range(k):
        out += w[:, :, j] @ x[:, j * dil:j * dil + n]
    return out + b[:, None]


def relu(x):
    return np.maximum(x, 0)


def avgpool2(x):
    n = x.shape[1] // 2
    return x[:, :2 * n].reshape(x.shape[0], n, 2).mean(axis=2)


class StepNet:
    def __init__(self, spec):
        self.spec = spec
        self.p = {k: np.asarray(v, dtype=np.float32) for k, v in spec["weights"].items()}
        self.mean = np.asarray(spec["in_mean"], dtype=np.float32)[:, None]
        self.std = np.asarray(spec["in_std"], dtype=np.float32)[:, None]
        self.step_length = spec["step_length_m"]

    def run(self, window):
        """Returns (steps in the window, {activity: probability})."""
        p = self.p
        x = (inputs(window) - self.mean) / self.std
        x = avgpool2(relu(conv1d(x, p["conv1.weight"], p["conv1.bias"], pad=3)))
        x = avgpool2(relu(conv1d(x, p["conv2.weight"], p["conv2.bias"], pad=3)))
        x = relu(conv1d(x, p["conv3.weight"], p["conv3.bias"], pad=4, dil=2))
        density = relu(conv1d(x, p["density.weight"], p["density.bias"], pad=0))
        steps = float(density.sum())
        logits = p["classify.weight"] @ x.mean(axis=1) + p["classify.bias"]
        e = np.exp(logits - logits.max())
        probs = e / e.sum()
        return steps, {lab: float(q) for lab, q in zip(LABELS, probs)}


def load(path):
    with open(path) as f:
        return StepNet(json.load(f))
