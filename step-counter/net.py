"""
The step-counting net in PyTorch (laptop only), plus export to the JSON that
unoq_app/python/stepnet.py runs with numpy on the board.

    window (2 s, 8 channels: ax ay az gx gy gz |a| |g|)
      -> Conv1d 8->16 (k=7) -> ReLU -> AvgPool /2      200 -> 100 time steps
      -> Conv1d 16->32 (k=7) -> ReLU -> AvgPool /2     100 -> 50
      -> Conv1d 32->32 (k=5, dilation 2) -> ReLU       50  (sees ~0.6 s each)
      -> density: Conv1d 32->1 (k=1) -> ReLU -> sum over time = steps in window
      -> classify: average over time -> Linear 32->3 = still / walk / run

The ReLU on the density keeps every bit of it >= 0, so the step count can
only go up, never down.
"""

import json

import numpy as np
import torch
import torch.nn as nn

from learn import S
import stepnet


class StepNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv1d(8, 16, 7, padding=3)
        self.conv2 = nn.Conv1d(16, 32, 7, padding=3)
        self.conv3 = nn.Conv1d(32, 32, 5, padding=4, dilation=2)
        self.density = nn.Conv1d(32, 1, 1)
        self.classify = nn.Linear(32, len(S.LABELS))
        self.relu = nn.ReLU()
        self.pool = nn.AvgPool1d(2)

    def forward(self, x):
        """x: (batch, 8, 200), already standardized. Returns (steps, logits)."""
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = self.relu(self.conv3(x))
        steps = self.relu(self.density(x)).sum(dim=(1, 2))
        logits = self.classify(x.mean(dim=2))
        return steps, logits


def to_inputs(X, mean, std):
    """(n, 200, 6) windows -> standardized (n, 8, 200) tensor."""
    x = np.stack([stepnet.inputs(w) for w in X])
    return torch.from_numpy((x - mean[None, :, None]) / std[None, :, None]).float()


def export(model, mean, std, extra):
    """The JSON spec stepnet.StepNet loads."""
    return {
        "type": "stepnet",
        "labels": S.LABELS,
        "in_mean": mean.tolist(),
        "in_std": std.tolist(),
        "weights": {k: v.detach().numpy().round(6).tolist()
                    for k, v in model.state_dict().items()},
        **extra,
    }


def check_export(model, spec, X, mean, std):
    """Largest difference in steps between PyTorch and the numpy version on
    a few windows (should be ~1e-5)."""
    board = stepnet.StepNet(json.loads(json.dumps(spec)))
    model.eval()
    with torch.no_grad():
        steps, _ = model(to_inputs(X, mean, std))
    return max(abs(board.run(w)[0] - s) for w, s in zip(X, steps.numpy()))
