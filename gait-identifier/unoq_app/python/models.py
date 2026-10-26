"""
Run the trained models. Only needs numpy, so it runs on the UNO Q.
The models are trained on the laptop (train.py) and saved as JSON:

  peak_model.json   {"type": "peak", "axis": "ax", "sign": 1, "tree": {...}}
  full_model.json   {"type": "full", "kind": "tree", "tree": {...}}
                 or {"type": "full", "kind": "knn", "k": 5, "mean": [...],
                     "std": [...], "X": [[...]], "y": [...]}

A tree is {"nodes": [...]} with node 0 as the root. Each node is either
{"leaf": "walk"} or {"f": feature index, "t": threshold, "l": left, "r": right}:
go left when x[f] <= t.
"""

import json
import os
from collections import Counter

import numpy as np

import features


def tree_predict(tree, x):
    nodes = tree["nodes"]
    node = nodes[0]
    while "leaf" not in node:
        node = nodes[node["l"] if x[node["f"]] <= node["t"] else node["r"]]
    return node["leaf"]


def knn_predict(model, x):
    X = np.asarray(model["X"])
    z = (np.asarray(x) - np.asarray(model["mean"])) / np.asarray(model["std"])
    nearest = np.argsort(np.sum((X - z) ** 2, axis=1))[:model["k"]]
    # Counter keeps insertion order, so a tie goes to the closest neighbor.
    return Counter(model["y"][i] for i in nearest).most_common(1)[0][0]


class PeakModel:
    """Model 1: one number (peak acceleration along one direction) -> thresholds."""

    def __init__(self, spec):
        self.spec = spec
        self.axis, self.sign = spec["axis"], spec["sign"]

    def feature(self, window):
        return features.peak(window, self.axis, self.sign)

    def predict(self, window):
        return tree_predict(self.spec["tree"], [self.feature(window)])


class FullModel:
    """Model 2: all IMU outputs -> ~46 features -> decision tree or KNN."""

    def __init__(self, spec):
        self.spec = spec

    def predict(self, window):
        x = features.full_features(window)
        if self.spec["kind"] == "tree":
            return tree_predict(self.spec["tree"], x)
        return knn_predict(self.spec, x)


def load(path):
    with open(path) as f:
        spec = json.load(f)
    return PeakModel(spec) if spec["type"] == "peak" else FullModel(spec)


def load_all(folder):
    """{'peak': model, 'full': model} for whichever JSON files exist."""
    found = {}
    for name in ("peak", "full"):
        path = os.path.join(folder, f"{name}_model.json")
        if os.path.exists(path):
            found[name] = load(path)
    return found
