"""
Train both models from the recordings in data/ and save them as JSON.

    python train.py                  # uses data/
    python train.py --data data_fake # e.g. the synthetic test data
    python train.py --person fiona   # only fiona's recordings

Model 1 (peak):  tries each of the 6 directions (+ax, -ax, +ay, ...), keeps the
                 one whose peak value cross-validates best, and learns
                 thresholds on it with a small decision tree.
Model 2 (full):  ~46 features from all 6 IMU channels; compares a decision tree
                 with KNN and keeps the better one.

Writes models/peak_model.json and models/full_model.json, and copies them
into unoq_app/python/models/ so they go to the board with the app.
"""

import argparse
import json
import os
import shutil

import learn
from learn import S, features

PEAK_DEPTH = 3       # at most 8 ranges of the one number; usually only 3-4 used
FULL_DEPTH = 5
KNN_K = 5


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--data", default=os.path.join(learn.HERE, "data"))
    ap.add_argument("--person", help="train only on this person's recordings")
    ap.add_argument("--folds", choices=["auto", "session", "block", "person"], default="auto")
    args = ap.parse_args()

    ds = learn.Dataset(learn.load_sessions(args.data, args.person))
    fold_type = learn.pick_fold_type(ds) if args.folds == "auto" else args.folds
    fold_list = learn.folds(ds, fold_type)
    who = f" (person: {args.person})" if args.person else " (everyone)"
    print(f"Data from {args.data}{who}:\n{ds.summary()}")
    print(f"Cross-validation: hold out one {fold_type} at a time ({len(fold_list)} folds)\n")
    if fold_type == "block":
        print("  (some activity has only 1 recording; record each at least twice for "
              "leave-one-recording-out)\n")

    # --- Model 1: peak in one direction ---------------------------------------
    print("Model 1: peak acceleration in one direction")
    best = None
    for axis, sign in features.PEAK_DIRECTIONS:
        X = ds.peak(axis, sign)
        pred = learn.cross_val(learn.tree_fitter(PEAK_DEPTH, min_leaf=10), X, ds.y, fold_list)
        acc = learn.accuracy(ds.y, pred)
        print(f"  {features.direction_name(axis, sign):4s}  cross-validated accuracy {100 * acc:5.1f}%")
        if best is None or acc > best[0]:
            best = (acc, axis, sign)
    acc, axis, sign = best
    tree = learn.fit_tree(ds.peak(axis, sign), ds.y, PEAK_DEPTH, min_leaf=10)
    peak_spec = {"type": "peak", "axis": axis, "sign": sign, "tree": tree,
                 "cv_accuracy": round(acc, 4), "folds": fold_type}
    name = f"peak {features.direction_name(axis, sign)} (g)"
    print(f"  -> using {features.direction_name(axis, sign)} ({100 * acc:.1f}%). Rule:")
    print("     " + learn.tree_text(tree, [name]).replace("\n", "\n     ") + "\n")

    # --- Model 2: all IMU outputs ---------------------------------------------
    print("Model 2: all IMU outputs (" + str(len(features.FULL_NAMES)) + " features)")
    X = ds.full()
    candidates = {
        "tree": learn.tree_fitter(FULL_DEPTH),
        "knn": learn.knn_fitter(KNN_K),
    }
    scores = {}
    for kind, fit in candidates.items():
        pred = learn.cross_val(fit, X, ds.y, fold_list)
        scores[kind] = learn.accuracy(ds.y, pred)
        label = f"decision tree (depth {FULL_DEPTH})" if kind == "tree" else f"KNN (k={KNN_K})"
        print(f"  {label:24s} cross-validated accuracy {100 * scores[kind]:5.1f}%")
    kind = max(scores, key=scores.get)
    if kind == "tree":
        full_spec = {"type": "full", "kind": "tree", "tree": learn.fit_tree(X, ds.y, FULL_DEPTH)}
    else:
        full_spec = {"type": "full", **learn.fit_knn(X, ds.y, KNN_K), "kind": "knn"}
    full_spec.update({"features": features.FULL_NAMES, "cv_accuracy": round(scores[kind], 4),
                      "folds": fold_type})
    print(f"  -> using {kind}\n")

    # --- Save -----------------------------------------------------------------
    # Only models trained on real data/ go to the board.
    real = os.path.abspath(args.data) == os.path.join(learn.HERE, "data")
    out_dirs = [os.path.join(learn.HERE, "models")]
    if real:
        out_dirs.append(os.path.join(learn.HERE, "unoq_app", "python", "models"))
    for d in out_dirs:
        os.makedirs(d, exist_ok=True)
    for fname, spec in (("peak_model.json", peak_spec), ("full_model.json", full_spec)):
        spec["data"] = os.path.basename(os.path.abspath(args.data))
        spec["person"] = args.person or "everyone"
        path = os.path.join(out_dirs[0], fname)
        with open(path, "w") as f:
            json.dump(spec, f)
        for d in out_dirs[1:]:
            shutil.copy(path, os.path.join(d, fname))
    print("Saved models/peak_model.json and models/full_model.json "
          + ("(also copied to unoq_app/python/models/ for the board)." if real
             else f"(trained on {args.data}, so NOT copied to the board)."))
    print("Next: python evaluate.py   for confusion matrices and plots.")


if __name__ == "__main__":
    main()
