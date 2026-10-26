"""
Laptop-side learning code: load recordings, cut them into windows, train a
decision tree or KNN (plain numpy, no neural nets, no scikit-learn), and
cross-validate. train.py and evaluate.py both use this.
"""

import glob
import os
import sys
from collections import Counter

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "unoq_app", "python"))

import features                                   # noqa: E402
import models                                     # noqa: E402
import settings as S                              # noqa: E402


# --- Data -------------------------------------------------------------------

def load_sessions(folder, person=None):
    """Every data/<label>_<person>_<time>.csv as a dict with the samples
    (only that person's, if person is given)."""
    sessions = []
    for path in sorted(glob.glob(os.path.join(folder, "*.csv"))):
        name = os.path.splitext(os.path.basename(path))[0]
        parts = name.split("_")
        if len(parts) < 3 or parts[0] not in S.LABELS:
            print(f"skipping {path}: name should be <label>_<person>_<time>.csv")
            continue
        if person and parts[1] != person:
            continue
        data = np.loadtxt(path, delimiter=",", skiprows=1, ndmin=2)
        sessions.append({"name": name, "label": parts[0], "person": parts[1],
                         "t": data[:, 0], "imu": data[:, 1:7]})
    return sessions


def windows_of(session):
    """Cut one recording into 2 s windows every 0.5 s, after trimming the ends.
    Returns (windows, start sample indices)."""
    trim = int(S.TRIM_S * S.SAMPLE_HZ)
    imu = session["imu"][trim:len(session["imu"]) - trim]
    starts = list(range(0, len(imu) - S.WINDOW_N + 1, S.STEP_N))
    return [imu[i:i + S.WINDOW_N] for i in starts], starts


class Dataset:
    """All windows from all recordings, with labels and where they came from."""

    def __init__(self, sessions):
        self.sessions = sessions
        wins, y, sess, person, start = [], [], [], [], []
        for k, s in enumerate(sessions):
            w, st = windows_of(s)
            wins += w
            y += [s["label"]] * len(w)
            sess += [k] * len(w)
            person += [s["person"]] * len(w)
            start += st
        if not wins:
            raise SystemExit("No windows: record some data first (each recording needs > 8 s).")
        self.windows = np.array(wins)
        self.y = np.array(y)
        self.session = np.array(sess)
        self.person = np.array(person)
        self.start = np.array(start)
        self._full = None

    def peak(self, axis, sign):
        return np.array([[features.peak(w, axis, sign)] for w in self.windows])

    def full(self):
        if self._full is None:
            self._full = np.array([features.full_features(w) for w in self.windows])
        return self._full

    def summary(self):
        lines = []
        for lab in S.LABELS:
            m = self.y == lab
            n_sess = len(set(self.session[m]))
            lines.append(f"  {lab:6s} {m.sum():5d} windows from {n_sess} recording(s)")
        return "\n".join(lines)


# --- Cross-validation folds -------------------------------------------------

def folds(ds, by="session"):
    """List of (train_mask, test_mask, description).

    by="session": hold out one whole recording at a time (honest: the model
      never saw any part of it). Needs every activity recorded at least twice.
    by="person":  hold out one person at a time.
    by="block":   fallback when an activity has only one recording: split each
      recording into 5 time blocks and hold out one block per fold. Training
      windows that overlap a test window are dropped so nothing leaks.
    """
    out = []
    if by == "person":
        for p in sorted(set(ds.person)):
            out.append((ds.person != p, ds.person == p, f"person {p}"))
        return out
    if by == "session":
        for k, s in enumerate(ds.sessions):
            out.append((ds.session != k, ds.session == k, s["name"]))
        return out
    # by == "block"
    block = np.zeros(len(ds.y), dtype=int)
    for k in set(ds.session):
        idx = np.where(ds.session == k)[0]
        block[idx] = np.minimum(np.arange(len(idx)) * 5 // len(idx), 4)
    for b in range(5):
        test = block == b
        train = ~test
        for k in set(ds.session[test]):
            st = ds.start[test & (ds.session == k)]
            near = (ds.session == k) & (ds.start > st.min() - S.WINDOW_N) & (ds.start < st.max() + S.WINDOW_N)
            train &= ~near
        out.append((train, test, f"block {b + 1}/5"))
    return out


def pick_fold_type(ds):
    per_label = [len(set(ds.session[ds.y == lab])) for lab in S.LABELS if (ds.y == lab).any()]
    return "session" if min(per_label) >= 2 else "block"


def cross_val(fit, X, y, fold_list):
    """Predictions for every window from a model that never trained on it."""
    pred = np.empty(len(y), dtype=object)
    for train, test, _ in fold_list:
        if not train.any() or not test.any():
            continue
        model = fit(X[train], y[train])
        pred[test] = [model(x) for x in X[test]]
    return pred


# --- Decision tree (CART, Gini impurity) ------------------------------------

def fit_tree(X, y, max_depth=4, min_leaf=5):
    """Greedy decision tree. Returns {"nodes": [...]} in the format models.py reads."""
    X = np.asarray(X, dtype=float)
    classes = sorted(set(y))
    yi = np.array([classes.index(v) for v in y])
    K = len(classes)
    nodes = []

    def gini(counts, n):
        return 1.0 - np.sum((counts / n) ** 2, axis=-1)

    def build(idx, depth):
        me = len(nodes)
        nodes.append(None)
        counts = np.bincount(yi[idx], minlength=K)
        leaf = {"leaf": classes[int(np.argmax(counts))],
                "n": {classes[i]: int(c) for i, c in enumerate(counts) if c}}
        if depth >= max_depth or counts.max() == len(idx) or len(idx) < 2 * min_leaf:
            nodes[me] = leaf
            return me
        n = len(idx)
        best = (gini(counts, n), None, None)
        for f in range(X.shape[1]):
            order = np.argsort(X[idx, f], kind="stable")
            xs, ys = X[idx, f][order], yi[idx][order]
            left = np.cumsum(np.eye(K)[ys], axis=0)[:-1]          # counts left of each split
            nl = np.arange(1, n)[:, None]
            right = counts - left
            score = (nl[:, 0] * gini(left, nl) + (n - nl[:, 0]) * gini(right, n - nl)) / n
            ok = (xs[:-1] < xs[1:]) & (nl[:, 0] >= min_leaf) & (n - nl[:, 0] >= min_leaf)
            if not ok.any():
                continue
            score = np.where(ok, score, np.inf)
            i = int(np.argmin(score))
            if score[i] < best[0] - 1e-12:
                best = (score[i], f, (xs[i] + xs[i + 1]) / 2)
        if best[1] is None:
            nodes[me] = leaf
            return me
        _, f, t = best
        go_left = X[idx, f] <= t
        l = build(idx[go_left], depth + 1)
        r = build(idx[~go_left], depth + 1)
        nodes[me] = {"f": int(f), "t": float(t), "l": l, "r": r}
        return me

    build(np.arange(len(yi)), 0)
    return {"nodes": nodes}


def tree_fitter(max_depth, min_leaf=5):
    def fit(X, y):
        tree = fit_tree(X, y, max_depth, min_leaf)
        return lambda x: models.tree_predict(tree, x)
    return fit


def tree_text(tree, names):
    """The tree as indented if/else text, for reports."""
    lines = []

    def walk(i, indent):
        node = tree["nodes"][i]
        pad = "    " * indent
        if "leaf" in node:
            counts = ", ".join(f"{k} {v}" for k, v in node["n"].items())
            lines.append(f"{pad}-> {node['leaf']}   (training windows: {counts})")
            return
        lines.append(f"{pad}if {names[node['f']]} <= {node['t']:.3f}:")
        walk(node["l"], indent + 1)
        lines.append(f"{pad}else:")
        walk(node["r"], indent + 1)

    walk(0, 0)
    return "\n".join(lines)


def tree_thresholds(tree):
    return sorted(n["t"] for n in tree["nodes"] if "leaf" not in n)


# --- KNN --------------------------------------------------------------------

def fit_knn(X, y, k=5):
    X = np.asarray(X, dtype=float)
    mean, std = X.mean(axis=0), X.std(axis=0)
    std[std < 1e-9] = 1.0
    return {"k": int(k), "mean": mean.tolist(), "std": std.tolist(),
            "X": ((X - mean) / std).round(4).tolist(), "y": list(y)}


def knn_fitter(k):
    def fit(X, y):
        spec = fit_knn(X, y, k)
        return lambda x: models.knn_predict(spec, x)
    return fit


# --- Scoring ----------------------------------------------------------------

def confusion(y, pred):
    labels = [lab for lab in S.LABELS if (y == lab).any() or (pred == lab).any()]
    m = np.zeros((len(labels), len(labels)), dtype=int)
    for a, b in zip(y, pred):
        if b is not None:
            m[labels.index(a), labels.index(b)] += 1
    return labels, m


def accuracy(y, pred):
    ok = np.array([p is not None for p in pred])
    return float(np.mean(y[ok] == pred[ok])) if ok.any() else 0.0


def confusion_text(y, pred):
    labels, m = confusion(y, pred)
    lines = ["true \\ predicted  " + "".join(f"{lab:>7s}" for lab in labels) + "   correct"]
    for i, lab in enumerate(labels):
        row_total = m[i].sum()
        pct = f"{100 * m[i, i] / row_total:5.1f}%" if row_total else "   - "
        lines.append(f"  {lab:15s} " + "".join(f"{v:7d}" for v in m[i]) + f"    {pct}")
    return "\n".join(lines)


def majority(values):
    return Counter(values).most_common(1)[0][0]
