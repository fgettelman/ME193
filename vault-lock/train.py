"""
Train FingerNet on the pictures from collect.py and export it for the board.

    python train.py                       # width 16, depth 4, 40 epochs
    python train.py --width 8 --depth 3 --epochs 60

Saves:
    models/fingernet.pt                      PyTorch weights (for gradcam.py, laptop_lock.py)
    unoq_app/python/models/fingernet.json    what the board runs (numpy), checked against PyTorch
    results/training.png                     loss and accuracy per epoch
    results/confusion.png                    which counts get mixed up

The pieces, in the order they happen each step:
  1. augment()       random rotations, zoom, shifts, mirror (= the other hand),
                     brightness and colour: ~50 pictures per recording become
                     endless slightly different ones
  2. forward pass    model(x) -> 6 scores per picture
  3. loss            cross-entropy: -log(probability given to the right count)
  4. backward pass   loss.backward(): backpropagation, the chain rule from the
                     loss back through every layer, gives d loss / d weight
  5. update          SGDMomentum.step(): gradient descent, written out by hand
"""

import argparse
import glob
import json
import math
import os
import random
from collections import defaultdict

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from model import (FingerNet, standardize, images_to_batch, export, check_export,
                   describe, n_params, HERE, S)

DATA = os.path.join(HERE, "data")
MODELS = os.path.join(HERE, "models")
RESULTS = os.path.join(HERE, "results")
BOARD_MODEL = os.path.join(HERE, "unoq_app", "python", "models", "fingernet.json")


# --- Data -----------------------------------------------------------------------

def load_data(data_dir=DATA, val_frac=0.2, seed=0):
    """Returns (train, val), each a list of (image, label). Whole recordings go
    to one side or the other: pictures from the same recording are nearly
    identical, so splitting them would make the test far too easy."""
    rng = random.Random(seed)
    train, val = [], []
    for k, lab in enumerate(S.LABELS):
        sessions = defaultdict(list)
        for path in sorted(glob.glob(os.path.join(data_dir, lab, "*.png"))):
            sessions[os.path.basename(path).rsplit("_", 1)[0]].append(path)
        names = sorted(sessions)
        rng.shuffle(names)
        n_val = max(1, round(val_frac * len(names))) if len(names) >= 2 else 0
        if len(names) == 1:
            print(f"  '{lab}' has only one recording: testing on part of it (too easy, record more)")
            paths = sessions[names[0]]
            cut = int(len(paths) * (1 - val_frac))
            train += [(p, k) for p in paths[:cut]]
            val += [(p, k) for p in paths[cut:]]
            continue
        for i, name in enumerate(names):
            (val if i < n_val else train).extend((p, k) for p in sessions[name])

    def read(items):
        return [(cv2.imread(p), k) for p, k in items]
    return read(train), read(val)


def to_tensors(items):
    x = images_to_batch([im for im, _ in items])
    y = torch.tensor([k for _, k in items], dtype=torch.long)
    return x, y


def augment(x, strength=1.0):
    """Random changes to a batch (n, 3, 64, 64) in 0..1, done on the whole
    batch at once with an affine grid (rotate + zoom + shift + mirror)."""
    n = x.shape[0]
    angle = (torch.rand(n) * 2 - 1) * math.radians(25) * strength
    scale = 1 + (torch.rand(n) * 2 - 1) * 0.15 * strength
    flip = torch.where(torch.rand(n) < 0.5, -1.0, 1.0)
    shift = (torch.rand(n, 2) * 2 - 1) * 0.12 * strength
    cos, sin = torch.cos(angle) / scale, torch.sin(angle) / scale
    theta = torch.stack([torch.stack([cos * flip, -sin, shift[:, 0]], 1),
                         torch.stack([sin * flip, cos, shift[:, 1]], 1)], 1)
    grid = F.affine_grid(theta, x.shape, align_corners=False)
    x = F.grid_sample(x, grid, padding_mode="reflection", align_corners=False)
    # Lighting: brightness, contrast, a slight colour cast, sensor noise.
    bright = 1 + (torch.rand(n, 1, 1, 1) * 2 - 1) * 0.3 * strength
    cast = 1 + (torch.rand(n, 3, 1, 1) * 2 - 1) * 0.08 * strength
    mean = x.mean(dim=(1, 2, 3), keepdim=True)
    contrast = 1 + (torch.rand(n, 1, 1, 1) * 2 - 1) * 0.3 * strength
    x = ((x - mean) * contrast + mean) * bright * cast
    x = x + torch.randn_like(x) * 0.02 * strength
    return x.clamp(0, 1)


# --- Gradient descent, by hand ---------------------------------------------------

class SGDMomentum:
    """Stochastic gradient descent with momentum and weight decay; the same
    maths as torch.optim.SGD, written out.

    After loss.backward(), every weight w has w.grad = d loss / d w (from
    backpropagation). Gradient descent moves each weight a small step downhill:
        g = d loss / d w + weight_decay * w      (weight decay pulls weights towards 0)
        v = momentum * v + g                     (v: a running sum of recent gradients,
                                                  so the steps keep going in directions
                                                  that agree and the zig-zags cancel)
        w = w - lr * v
    'Stochastic' because each gradient comes from a random batch of 64
    pictures, not the whole dataset."""

    def __init__(self, params, lr, momentum=0.9, weight_decay=5e-4):
        self.params = [p for p in params if p.requires_grad]
        self.lr, self.momentum, self.wd = lr, momentum, weight_decay
        self.v = [torch.zeros_like(p) for p in self.params]

    @torch.no_grad()
    def step(self):
        for p, v in zip(self.params, self.v):
            if p.grad is None:
                continue
            g = p.grad + self.wd * p if p.dim() > 1 else p.grad     # no decay on biases
            v.mul_(self.momentum).add_(g)
            p.sub_(self.lr * v)

    def zero_grad(self):
        for p in self.params:
            p.grad = None


# --- Training --------------------------------------------------------------------

def evaluate(model, x, y, batch=256):
    model.eval()
    probs = []
    with torch.no_grad():
        for i in range(0, len(x), batch):
            probs.append(F.softmax(model(standardize(x[i:i + batch])), dim=1))
    p = torch.cat(probs)
    loss = F.nll_loss(torch.log(p + 1e-9), y).item()
    return loss, (p.argmax(1) == y).float().mean().item(), p


def train(train_items, val_items, width=16, depth=4, epochs=40, batch=64, lr=0.05,
          seed=0, verbose=True):
    """Returns (best model, history dict, val probabilities)."""
    torch.manual_seed(seed)
    xt, yt = to_tensors(train_items)
    xv, yv = to_tensors(val_items)
    model = FingerNet(width, depth)
    opt = SGDMomentum(model.parameters(), lr)
    # Counts with fewer pictures count for more in the loss, so a pile of "0"
    # pictures can't teach the net to just say 0.
    freq = torch.bincount(yt, minlength=S.N_CLASSES).float().clamp(min=1)
    class_weight = freq.sum() / (S.N_CLASSES * freq)

    steps_per_epoch = math.ceil(len(xt) / batch)
    total = epochs * steps_per_epoch
    hist = defaultdict(list)
    best, best_state = -1.0, None
    step = 0
    for epoch in range(epochs):
        model.train()
        order = torch.randperm(len(xt))
        run_loss, run_right = 0.0, 0
        for i in range(0, len(xt), batch):
            # Learning rate: a short warm-up, then a cosine curve down to ~0.
            opt.lr = lr * min(1.0, (step + 1) / steps_per_epoch) * 0.5 * (1 + math.cos(math.pi * step / total))
            idx = order[i:i + batch]
            x = standardize(augment(xt[idx]))
            scores = model(x)                                         # forward pass
            loss = F.cross_entropy(scores, yt[idx], weight=class_weight)
            opt.zero_grad()
            loss.backward()                                           # backpropagation
            opt.step()                                                # gradient descent
            run_loss += loss.item() * len(idx)
            run_right += (scores.argmax(1) == yt[idx]).sum().item()
            step += 1
        vl, va, _ = evaluate(model, xv, yv) if len(xv) else (float("nan"), float("nan"), None)
        hist["train_loss"].append(run_loss / len(xt))
        hist["train_acc"].append(run_right / len(xt))
        hist["val_loss"].append(vl)
        hist["val_acc"].append(va)
        if verbose:
            print(f"epoch {epoch + 1:3d}/{epochs}  lr {opt.lr:.4f}  "
                  f"train loss {hist['train_loss'][-1]:.3f} acc {hist['train_acc'][-1]:.1%}   "
                  f"test loss {vl:.3f} acc {va:.1%}")
        score = va if len(xv) else -epoch
        if score >= best:            # keep the best epoch on the test recordings
            best = score
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
    model.load_state_dict(best_state)
    _, _, pv = evaluate(model, xv, yv) if len(xv) else (None, None, None)
    return model, dict(hist), pv


# --- Reports ---------------------------------------------------------------------

def confusion(y, pred):
    m = np.zeros((S.N_CLASSES, S.N_CLASSES), int)
    for a, b in zip(y, pred):
        m[a, b] += 1
    return m


def plots(hist, cm):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    os.makedirs(RESULTS, exist_ok=True)
    ep = np.arange(1, len(hist["train_loss"]) + 1)
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.8))
    ax[0].plot(ep, hist["train_loss"], label="train (augmented)")
    ax[0].plot(ep, hist["val_loss"], label="test recordings")
    ax[0].set(xlabel="epoch", ylabel="cross-entropy loss", title="Loss")
    ax[1].plot(ep, np.array(hist["train_acc"]) * 100, label="train (augmented)")
    ax[1].plot(ep, np.array(hist["val_acc"]) * 100, label="test recordings")
    ax[1].set(xlabel="epoch", ylabel="% right", title="Accuracy", ylim=(0, 101))
    for a in ax:
        a.grid(alpha=0.3)
        a.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(RESULTS, "training.png"), dpi=120)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(4.6, 4))
    ax.imshow(cm, cmap="Blues")
    for i in range(S.N_CLASSES):
        for j in range(S.N_CLASSES):
            ax.text(j, i, cm[i, j], ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black")
    ax.set(xticks=range(S.N_CLASSES), yticks=range(S.N_CLASSES), xticklabels=S.LABELS,
           yticklabels=S.LABELS, xlabel="net says", ylabel="really", title="Test recordings")
    fig.tight_layout()
    fig.savefig(os.path.join(RESULTS, "confusion.png"), dpi=120)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description="Train the finger counter")
    ap.add_argument("--width", type=int, default=16)
    ap.add_argument("--depth", type=int, default=4)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--lr", type=float, default=0.05)
    ap.add_argument("--data", default=DATA)
    args = ap.parse_args()

    train_items, val_items = load_data(args.data)
    if not train_items:
        raise SystemExit(f"No pictures in {args.data}/. Run collect.py first.")
    print(f"{len(train_items)} training pictures, {len(val_items)} test pictures (held-out recordings)")
    print("per count (train/test):", {lab: (sum(k == i for _, k in train_items), sum(k == i for _, k in val_items))
                                      for i, lab in enumerate(S.LABELS)})
    model_preview = FingerNet(args.width, args.depth)
    print(describe(model_preview), "\n")

    model, hist, pv = train(train_items, val_items, args.width, args.depth, args.epochs, lr=args.lr)

    yv = np.array([k for _, k in val_items])
    extra = {"n_train": len(train_items), "n_val": len(val_items), "params": n_params(model)}
    if pv is not None and len(yv):
        pv = pv.numpy()
        cm = confusion(yv, pv.argmax(1))
        acc = (pv.argmax(1) == yv).mean()
        sure = pv.max(1) >= S.CONF_MIN
        print(f"\nTest accuracy: {acc:.1%}")
        print("Confusion (rows = really, columns = net says):")
        print("     " + " ".join(f"{lab:>4s}" for lab in S.LABELS))
        for i, lab in enumerate(S.LABELS):
            print(f"  {lab}: " + " ".join(f"{v:4d}" for v in cm[i]))
        right_when_sure = (pv.argmax(1) == yv)[sure].mean() if sure.any() else float("nan")
        print(f"\nThe lock only believes answers with probability >= {S.CONF_MIN:.0%}: "
              f"{sure.mean():.0%} of test pictures are that sure, "
              f"and {right_when_sure:.1%} of those are right.")
        plots(hist, cm)
        extra.update({"val_acc": round(float(acc), 4), "confusion": cm.tolist()})
        print(f"Plots: {os.path.relpath(RESULTS)}/training.png, confusion.png")

    os.makedirs(MODELS, exist_ok=True)
    torch.save({"state": model.state_dict(), "width": args.width, "depth": args.depth},
               os.path.join(MODELS, "fingernet.pt"))
    spec = export(model, extra)
    diff = check_export(model, spec, [im for im, _ in (val_items or train_items)[:32]])
    print(f"numpy (board) vs PyTorch: largest difference in probability {diff:.1e}")
    if diff > 1e-3:
        raise SystemExit("The numpy version doesn't match PyTorch; not exporting.")
    os.makedirs(os.path.dirname(BOARD_MODEL), exist_ok=True)
    with open(BOARD_MODEL, "w") as f:
        json.dump(spec, f)
    print(f"Saved {os.path.relpath(BOARD_MODEL)} ({os.path.getsize(BOARD_MODEL) / 1e6:.1f} MB) "
          "and models/fingernet.pt")


if __name__ == "__main__":
    main()
