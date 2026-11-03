"""
Uncertainty analysis for the step counter.

    python uncertainty.py             # everything (the neural-net part takes ~5 min)
    python uncertainty.py --no-net    # skip retraining the neural net

Works out, from the recordings in data/ and the 5 m walks in calibration.json:

  1. Sensor noise: how noisy the gyro is when the leg is still, next to the
     spike and dip levels the step counter uses.
  2. Timing: how steady the 100 Hz sampling really is.
  3. Step count: the board's count against yours on the 5 m walks.
  4. Step length: spread over the 5 m walks (type A) and what limits it
     (type B: where your feet land against the start and finish lines).
  5. Distance: the two calibrations combined, checked on the 50 m walks.
  6. Sensitivity: how much the count moves if the spike or dip level changes.
  7. The neural net (PyTorch): leave-one-recording-out error, and how much
     the answer moves with the random seed it was trained with.

Prints a summary and saves everything to results/uncertainty.json.
"""

import argparse
import glob
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "unoq_app", "python"))

import learn                                      # noqa: E402
import spikes                                     # noqa: E402
from learn import S                               # noqa: E402

CALIB_FILE = os.path.join(HERE, "calibration.json")
OUT = os.path.join(HERE, "results", "uncertainty.json")
T95 = {1: 12.71, 2: 4.30, 3: 3.18, 4: 2.78, 5: 2.57, 6: 2.45, 7: 2.36, 8: 2.31, 9: 2.26, 10: 2.23}
FOOT_PLACEMENT = 0.5         # your feet can land up to half a step past either line


def load(path):
    d = np.loadtxt(path, delimiter=",", skiprows=1, ndmin=2)
    return d[:, 0], d[:, 1:7]


def count(t, imu, **levels):
    """Run the board's step counter over a recording. Returns the counter."""
    saved = {k: getattr(spikes, k) for k in levels}
    for k, v in levels.items():
        setattr(spikes, k, v)
    try:
        c = spikes.SpikeCounter()
        for ti, row in zip(t, imu):
            c.add(ti, row[3:])
        return c
    finally:
        for k, v in saved.items():
            setattr(spikes, k, v)


def mean_ci(x):
    """Mean, standard deviation, standard error and 95% half-width (t)."""
    x = np.asarray(x, dtype=float)
    n = len(x)
    sd = float(x.std(ddof=1)) if n > 1 else 0.0
    se = sd / np.sqrt(n) if n else 0.0
    return {"n": n, "mean": float(x.mean()), "sd": sd, "se": se,
            "ci95": T95.get(n - 1, 1.96) * se if n > 1 else None}


def recordings():
    out = {}
    for path in sorted(glob.glob(os.path.join(learn.DATA_DIR, "*.csv"))):
        name = os.path.splitext(os.path.basename(path))[0]
        if name.split("_")[0] in S.LABELS:
            out[name] = load(path)
    return out


# --- 1. Sensor noise --------------------------------------------------------

def sensor_noise(recs):
    t, imu = recs["still_fiona_20261026-162456"]       # standing still, quiet
    gyro_sd = imu[:, 3:].std(axis=0)
    c = spikes.SpikeCounter()
    swing = np.array([c.add(ti, row[3:]) for ti, row in zip(t, imu)])[200:]
    sd = float(swing.std())
    return {
        "recording": "still_fiona_20261026-162456",
        "gyro_sd_dps": gyro_sd.round(3).tolist(),
        "accel_sd_g": imu[:, :3].std(axis=0).round(4).tolist(),
        "gyro_resolution_dps": round(1 / S.GYRO_COUNTS_PER_DPS, 4),
        "swing_sd_dps": round(sd, 3),
        "swing_max_dps": round(float(np.abs(swing).max()), 2),
        "spike_level_in_sd": round(spikes.SPIKE_DPS / sd, 0),
        "dip_level_in_sd": round(spikes.DIP_DPS / sd, 0),
    }


# --- 2. Timing --------------------------------------------------------------

def timing(recs):
    dts = np.concatenate([np.diff(t) for t, _ in recs.values()])
    gap = dts > 3 / S.SAMPLE_HZ
    ok = dts[~gap]
    return {"samples": int(len(dts) + len(recs)), "dt_mean_ms": round(1000 * ok.mean(), 3),
            "dt_sd_ms": round(1000 * ok.std(), 3), "gaps": int(gap.sum()),
            "gap_s": round(float(dts[gap].sum()), 2), "rate_hz": round(1 / ok.mean(), 2)}


# --- 3-5. Step count, step length, distance ---------------------------------

def calibration_walks(person):
    with open(CALIB_FILE) as f:
        return [w for w in json.load(f)[person] if "detected" in w]


def step_count(walks):
    rows = [{"kind": w["kind"], "you": w["steps"], "board": w["detected"],
             "error": w["detected"] - w["steps"]} for w in walks]
    err = np.array([r["error"] for r in rows], dtype=float)
    rel = err / np.array([r["you"] for r in rows])
    rng = np.random.default_rng(0)
    boot = [rng.choice(rel, len(rel)).mean() for _ in range(10000)]
    return {
        "walks": rows,
        "exact": int((err == 0).sum()),
        "bias_steps": mean_ci(err),
        "abs_error_steps": float(np.abs(err).mean()),
        "median_error_steps": float(np.median(err)),
        "rel_error_mean": float(rel.mean()),
        "rel_error_ci95": [float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))],
        "never_under": bool((err >= 0).all()),
    }


def step_length(walks):
    out = {}
    for kind in ("walk", "run"):
        mine = [w for w in walks if w["kind"] == kind]
        if not mine:
            continue
        L = [w["meters"] / w["steps"] for w in mine]
        pooled = sum(w["meters"] for w in mine) / sum(w["steps"] for w in mine)
        n_mean = np.mean([w["steps"] for w in mine])
        # Type B for one walk: each foot can land up to half a step past its
        # line (uniform), at the start and the finish.
        u_n = FOOT_PLACEMENT / np.sqrt(3) * np.sqrt(2)
        out[kind] = {
            "per_walk_m": [round(x, 3) for x in L],
            "pooled_m": round(pooled, 4),
            "typeA": mean_ci(L),
            "typeB_one_walk_rel": float(u_n / n_mean),
            "typeB_steps": float(u_n),
        }
    return out


def distance_check(recs, lengths):
    """The 50 m walks: the board's step count x your calibrated step length."""
    counts = learn.read_counts()
    out = []
    for name, c in counts.items():
        if not c.get("meters") or name not in recs:
            continue
        t, imu = recs[name]
        steps = count(t, imu).steps
        L = lengths["walk"]["pooled_m"]
        rel_L = lengths["walk"]["typeA"]["se"] / L
        d = steps * L
        out.append({"recording": name, "true_m": c["meters"], "steps": steps,
                    "distance_m": round(d, 2), "error_pct": round(100 * (d / c["meters"] - 1), 2),
                    "u_from_step_length_m": round(d * rel_L, 2), "you_counted": c["steps"]})
    return out


# --- 6. Sensitivity -----------------------------------------------------------

def sensitivity(recs):
    moving = {n: v for n, v in recs.items() if not n.startswith("still")}
    still = {n: v for n, v in recs.items() if n.startswith("still")}
    base = {n: count(*v).steps for n, v in moving.items()}
    rows = []
    for name, values in (("SPIKE_DPS", [60, 70, 80, 90, 100, 120]), ("DIP_DPS", [20, 25, 35, 45, 60])):
        for v in values:
            got = {n: count(*rec, **{name: v}).steps for n, rec in moving.items()}
            change = [abs(got[n] - base[n]) / base[n] for n in base if base[n]]
            false = sum(count(*rec, **{name: v}).steps for rec in still.values())
            rows.append({"level": name, "value": v, "mean_change_pct": round(100 * np.mean(change), 2),
                         "max_change_pct": round(100 * np.max(change), 2), "still_steps": false})
    return rows


# --- 7. The neural net --------------------------------------------------------

def neural_net(person, seeds=5, epochs=60):
    import train                                  # needs torch
    from net import export
    import stepnet

    sessions = learn.load_sessions(person=person)
    lengths, _ = learn.step_lengths(sessions, person)
    loo = []
    for k, s in enumerate(sessions):
        others = [o for i, o in enumerate(sessions) if i != k]
        model, mean, std = train.fit(others, epochs, seed=k, log=None)
        net = stepnet.StepNet(export(model, mean, std, {"step_length_m": lengths}))
        got, _ = train.replay(net, s)
        labelled, _ = train.target(s)
        loo.append({"name": s["name"], "label": s["label"], "net": round(got, 2),
                    "labels": round(float(labelled), 2)})
        print(f"  held out {s['name']}: net {got:.1f}, labels {labelled:.1f}", flush=True)
    moving = [r for r in loo if r["label"] != "still" and r["labels"] > 0]
    rel = [(r["net"] - r["labels"]) / r["labels"] for r in moving]

    # Training randomness: hold one long walk out and train with several seeds.
    held = max((s for s in sessions if s["label"] == "walk"), key=lambda s: len(s["imu"]))
    others = [s for s in sessions if s is not held]
    per_seed = []
    for seed in range(seeds):
        model, mean, std = train.fit(others, epochs, seed=100 + seed, log=None)
        net = stepnet.StepNet(export(model, mean, std, {"step_length_m": lengths}))
        per_seed.append(round(train.replay(net, held)[0], 2))
        print(f"  seed {seed}: {per_seed[-1]:.1f} steps on {held['name']}", flush=True)
    return {"leave_one_out": loo, "rel_error": mean_ci(rel),
            "abs_rel_error_mean": float(np.mean(np.abs(rel))),
            "seed_test": {"recording": held["name"], "steps": per_seed, **mean_ci(per_seed)}}


def main():
    ap = argparse.ArgumentParser(description="Uncertainty analysis of the step counter")
    ap.add_argument("--person", default="fiona")
    ap.add_argument("--no-net", action="store_true", help="skip the neural-net part")
    args = ap.parse_args()

    recs = recordings()
    walks = calibration_walks(args.person)
    out = {"sensor": sensor_noise(recs), "timing": timing(recs),
           "step_count": step_count(walks)}
    out["step_length"] = step_length(walks)
    out["distance_check"] = distance_check(recs, out["step_length"])
    out["sensitivity"] = sensitivity(recs)
    out["spike_counter_on_recordings"] = [
        {"name": n, "steps": count(*v).steps, "labels": float(learn.step_labels(v[1], n.split("_")[0])[-1])}
        for n, v in recs.items()]
    if not args.no_net:
        print("Neural net, leave-one-recording-out (this retrains it 17 times)...")
        out["neural_net"] = neural_net(args.person)

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w") as f:
        json.dump(out, f, indent=1)

    s, c, L = out["sensor"], out["step_count"], out["step_length"]
    print(f"\nSensor: still-leg swing noise {s['swing_sd_dps']:.2f} deg/s; spike level is "
          f"{s['spike_level_in_sd']:.0f} sd, dip level {s['dip_level_in_sd']:.0f} sd above it")
    print(f"Timing: {out['timing']['rate_hz']} Hz, jitter {out['timing']['dt_sd_ms']} ms, "
          f"{out['timing']['gaps']} gaps ({out['timing']['gap_s']} s missing)")
    print(f"Count:  {c['exact']}/{len(c['walks'])} walks exact; board - you = "
          f"{c['bias_steps']['mean']:+.2f} +/- {c['bias_steps']['ci95']:.2f} steps (95%)")
    for kind, v in L.items():
        a = v["typeA"]
        print(f"Length: {kind} {v['pooled_m']:.3f} m/step, sd {a['sd']:.3f}, "
              f"+/- {a['ci95']:.3f} m (95%, n={a['n']}); one 5 m walk alone: "
              f"+/- {100 * v['typeB_one_walk_rel']:.0f}% from foot placement")
    for d in out["distance_check"]:
        print(f"50 m:   {d['recording']}: {d['steps']:.0f} steps -> {d['distance_m']} m "
              f"({d['error_pct']:+.1f}%)")
    if "neural_net" in out:
        n = out["neural_net"]
        print(f"Net:    held-out error {100 * n['abs_rel_error_mean']:.1f}% on average; seeds "
              f"{n['seed_test']['mean']:.1f} +/- {n['seed_test']['sd']:.1f} steps")
    print(f"\nSaved {OUT}")


if __name__ == "__main__":
    main()
