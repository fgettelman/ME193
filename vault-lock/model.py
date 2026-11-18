"""
FingerNet: the CNN that counts fingers, built from scratch in PyTorch (laptop
only), plus Grad-CAM and the export to the JSON the board runs with numpy.

    input   3 x 64 x 64   (centre of the camera frame, each colour channel standardized)
    block 1 conv 3x3, 3 -> W     BN ReLU   maxpool 2   ->  W  x 32 x 32
    block 2 conv 3x3, W -> 2W    BN ReLU   maxpool 2   -> 2W  x 16 x 16
    block 3 conv 3x3, 2W -> 4W   BN ReLU   maxpool 2   -> 4W  x  8 x  8
    block 4 conv 3x3, 4W -> 8W   BN ReLU               -> 8W  x  8 x  8   <- Grad-CAM looks here
    global average pool                                -> 8W
    dropout, linear 8W -> 6                            -> scores for 0..5 fingers

WIDTH (W) = how many channels (feature detectors) the first layer has; each
later block doubles it. DEPTH = how many conv blocks. W=16, depth=4 is the
default: about 98k weights. sweep.py trains other widths and depths.

BN = batch normalization: during training each channel is rescaled to mean 0,
std 1 over the batch, then by a learned scale and shift. It makes training
about 3x faster here. After training it's just a fixed scale and shift per
channel, so export() folds it into the conv's weights and bias, and the board
never needs to know it was there.

    python model.py                  # print the layer table for the default net
    python model.py --width 8 --depth 3
"""

import argparse
import os
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "unoq_app", "python"))

import settings as S       # noqa: E402
import fingernet           # noqa: E402  (the numpy version, for checking the export)


class FingerNet(nn.Module):
    def __init__(self, width=16, depth=4, dropout=0.3, pool_blocks=3):
        super().__init__()
        self.width, self.depth = width, depth
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        self.pools = []                         # does block i end with a 2x2 max pool?
        c_in = 3
        for i in range(depth):
            c_out = width * 2 ** i
            self.convs.append(nn.Conv2d(c_in, c_out, 3, padding=1))
            self.norms.append(nn.BatchNorm2d(c_out))
            # Pool after every block but the last, at most pool_blocks times
            # (64 -> 32 -> 16 -> 8): the last block keeps a map for Grad-CAM.
            self.pools.append(i < min(depth - 1, pool_blocks))
            c_in = c_out
        self.drop = nn.Dropout(dropout)
        self.fc = nn.Linear(c_in, S.N_CLASSES)

    def forward(self, x, return_features=False):
        """x: (batch, 3, 64, 64) -> scores (batch, 6). With return_features,
        also the last conv block's output (for Grad-CAM)."""
        for conv, norm, pool in zip(self.convs, self.norms, self.pools):
            x = F.relu(norm(conv(x)))
            if pool:
                x = F.max_pool2d(x, 2)
        features = x
        scores = self.fc(self.drop(x.mean(dim=(2, 3))))     # global average pool
        return (scores, features) if return_features else scores


def standardize(x):
    """Torch version of vision.to_input for a batch (batch, 3, H, W) in 0..1."""
    mean = x.mean(dim=(2, 3), keepdim=True)
    std = x.std(dim=(2, 3), keepdim=True, unbiased=False) + 0.02
    return (x - mean) / std


def images_to_batch(images):
    """List of 64x64x3 uint8 BGR images -> (n, 3, 64, 64) float tensor in 0..1."""
    a = np.stack(images).astype(np.float32).transpose(0, 3, 1, 2) / 255.0
    return torch.from_numpy(a)


# --- Grad-CAM: where is the net looking? ----------------------------------------

def gradcam(model, x, target=None):
    """Grad-CAM (Selvaraju et al., 2017) for a batch x (n, 3, 64, 64), already
    standardized.

    The last conv block gives C feature maps A_c (8x8). Backpropagation gives
    d score / d A_c: how much each spot of each map pushes the chosen class's
    score up. Average that over the map to get one weight per channel,
        alpha_c = mean over (i, j) of d score / d A_c[i, j],
    and the heat map is ReLU(sum_c alpha_c A_c): the places that make the net
    say "3 fingers". Upsampled to 64x64 and scaled to 0..1.

    Returns (heat (n, 64, 64) numpy, predicted classes)."""
    model.eval()
    x = x.clone().requires_grad_(False)
    scores, A = model(x, return_features=True)
    A.retain_grad()
    pred = scores.argmax(dim=1)
    target = pred if target is None else torch.as_tensor(target).expand_as(pred)
    scores.gather(1, target[:, None]).sum().backward()
    alpha = A.grad.mean(dim=(2, 3), keepdim=True)            # (n, C, 1, 1)
    cam = F.relu((alpha * A).sum(dim=1, keepdim=True))       # (n, 1, h, w)
    cam = F.interpolate(cam, size=x.shape[2:], mode="bilinear", align_corners=False)[:, 0]
    cam = cam / (cam.amax(dim=(1, 2), keepdim=True) + 1e-8)
    model.zero_grad(set_to_none=True)
    return cam.detach().numpy(), pred.numpy()


# --- Export to the board ----------------------------------------------------------

def fold_batchnorm(conv, norm):
    """BN after training: y = gamma * (z - mean) / sqrt(var + eps) + beta, with
    z = conv(x) = w * x + b. That's a scale s = gamma / sqrt(var + eps) per
    output channel, so it's the same as one conv with
        w' = s * w,   b' = s * (b - mean) + beta."""
    s = norm.weight / torch.sqrt(norm.running_var + norm.eps)
    w = conv.weight * s[:, None, None, None]
    b = s * (conv.bias - norm.running_mean) + norm.bias
    return w, b


def export(model, extra=None):
    """The JSON spec that unoq_app/python/fingernet.py runs."""
    def arr(t):
        return t.detach().cpu().numpy().round(6).tolist()

    layers = []
    for conv, norm, pool in zip(model.convs, model.norms, model.pools):
        w, b = fold_batchnorm(conv, norm)
        layers.append({"type": "conv", "w": arr(w), "b": arr(b), "pad": 1})
        layers.append({"type": "relu"})
        if pool:
            layers.append({"type": "maxpool", "k": 2})
    layers.append({"type": "gap"})                  # dropout does nothing after training
    layers.append({"type": "linear", "w": arr(model.fc.weight), "b": arr(model.fc.bias)})
    return {"type": "fingernet", "labels": S.LABELS, "img": S.IMG,
            "width": model.width, "depth": model.depth, "layers": layers, **(extra or {})}


def check_export(model, spec, images):
    """Largest difference in probabilities between PyTorch and the board's
    numpy version on some images (should be ~1e-6)."""
    board = fingernet.FingerNet(spec)
    model.eval()
    with torch.no_grad():
        p_torch = F.softmax(model(standardize(images_to_batch(images))), dim=1).numpy()
    p_np = np.stack([board.probs_small(im) for im in images])
    return float(np.abs(p_torch - p_np).max())


# --- What the net looks like ------------------------------------------------------

def describe(model):
    """Layer table: output shape, weights, and receptive field (how many input
    pixels across one output value can 'see')."""
    rows = []
    shape = (3, S.IMG, S.IMG)
    rf, jump = 1, 1        # receptive field, and input pixels between neighbouring outputs
    total = 0
    for i, (conv, norm, pool) in enumerate(zip(model.convs, model.norms, model.pools)):
        n = sum(p.numel() for p in conv.parameters()) + sum(p.numel() for p in norm.parameters())
        total += n
        shape = (conv.out_channels, shape[1], shape[2])
        rf += (3 - 1) * jump
        rows.append((f"conv{i + 1} 3x3 {conv.in_channels}->{conv.out_channels} BN ReLU", shape, n, rf))
        if pool:
            shape = (shape[0], shape[1] // 2, shape[2] // 2)
            rf += (2 - 1) * jump
            jump *= 2
            rows.append(("maxpool 2x2", shape, 0, rf))
    rows.append(("global average pool", (shape[0],), 0, S.IMG))
    n = sum(p.numel() for p in model.fc.parameters())
    total += n
    rows.append((f"linear {model.fc.in_features}->{S.N_CLASSES}", (S.N_CLASSES,), n, S.IMG))

    lines = [f"FingerNet width={model.width} depth={model.depth}",
             f"{'layer':32s} {'output':>14s} {'weights':>9s} {'sees (px)':>10s}"]
    for name, shp, n, r in rows:
        lines.append(f"{name:32s} {'x'.join(map(str, shp)):>14s} {n:9,d} {min(r, S.IMG):>7d}x{min(r, S.IMG)}")
    lines.append(f"{'total':32s} {'':>14s} {total:9,d}")
    return "\n".join(lines)


def n_params(model):
    return sum(p.numel() for p in model.parameters())


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Print the FingerNet layer table")
    ap.add_argument("--width", type=int, default=16)
    ap.add_argument("--depth", type=int, default=4)
    a = ap.parse_args()
    print(describe(FingerNet(a.width, a.depth)))
