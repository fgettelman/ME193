"""
Make SYNTHETIC recordings so the training/evaluation pipeline can be tested
before any real data exists. Writes to data_fake/ (never data/).

    python make_fake_data.py
    python train.py --data data_fake
    python evaluate.py --data data_fake

The signals come from a simple thigh model: the thigh swings at the stride or
pedal rate, gravity is split between the axes by the thigh angle, and walking
and running add a sharp jolt at each foot strike. Biking has no jolts; half
the bike sessions are air-biking on your back (thigh pointing up).
Accuracy on this data says nothing about real-world accuracy.
"""

import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "data_fake")
HZ = 100
SECONDS = 60
R_THIGH = 0.25          # m, sensor distance from the hip


def thigh(t, center, swing, freq, rng):
    phase = rng.uniform(0, 2 * np.pi)
    drift = 1 + 0.03 * np.sin(2 * np.pi * 0.05 * t + rng.uniform(0, 6))   # pace wanders
    ang = 2 * np.pi * freq * np.cumsum(drift) / HZ + phase
    return np.radians(center + swing * np.sin(ang)), ang


def impacts(ang, g_peak, rng, width_s=0.03):
    """A jolt each time the leg's own foot lands, a smaller one for the other foot."""
    out = np.zeros(len(ang))
    k = np.arange(-int(4 * width_s * HZ), int(4 * width_s * HZ) + 1)
    pulse = np.exp(-0.5 * (k / (width_s * HZ)) ** 2)
    for offset, scale in ((0.0, 1.0), (np.pi, 0.4)):
        hits = np.where(np.diff(np.floor((ang + offset) / (2 * np.pi))) > 0)[0]
        for h in hits:
            amp = scale * g_peak * rng.uniform(0.75, 1.2)
            lo, hi = max(0, h + k[0]), min(len(out), h + k[-1] + 1)
            out[lo:hi] += amp * pulse[lo - (h + k[0]):hi - (h + k[0])]
    return out


def session(label, rng, variant=0):
    t = np.arange(SECONDS * HZ) / HZ
    if label == "still":
        theta = np.radians(rng.uniform(0, 6) + 0.8 * np.sin(2 * np.pi * 0.25 * t))
        jolt = np.zeros_like(t)
        if variant:                                  # shifting weight now and then
            jolt = impacts(2 * np.pi * 0.15 * t, 0.15, rng, width_s=0.2)
    elif label == "walk":
        theta, ang = thigh(t, 10, rng.uniform(16, 24), rng.uniform(0.8, 1.0), rng)
        jolt = impacts(ang, rng.uniform(0.5, 0.9), rng)
    elif label == "run":
        theta, ang = thigh(t, 20, rng.uniform(32, 42), rng.uniform(1.3, 1.5), rng)
        jolt = impacts(ang, rng.uniform(2.0, 3.5), rng)
    else:  # bike; variant 1 = air-bike lying on your back
        center = rng.uniform(140, 160) if variant else rng.uniform(50, 60)
        theta, _ = thigh(t, center, rng.uniform(20, 28), rng.uniform(1.0, 1.4), rng)
        jolt = np.zeros_like(t)

    dtheta = np.gradient(theta) * HZ
    tangential = R_THIGH * np.gradient(dtheta) * HZ / 9.81      # in g
    ax = -np.cos(theta) - jolt * np.cos(theta)
    az = np.sin(theta) + jolt * np.sin(theta) + tangential
    ay = 0.02 * np.sin(2 * np.pi * rng.uniform(0.2, 0.6) * t)
    gy = np.degrees(dtheta)
    gx = 3 * np.sin(2 * np.pi * rng.uniform(0.5, 1.5) * t)
    gz = 4 * np.sin(2 * np.pi * rng.uniform(0.5, 1.5) * t)
    noise_g = 0.04 if label == "bike" and not variant else 0.01   # road buzz
    imu = np.column_stack([ax, ay, az, gx, gy, gz])
    imu[:, :3] += rng.normal(0, noise_g, (len(t), 3))
    imu[:, 3:] += rng.normal(0, 0.8, (len(t), 3))
    return t, imu


def main():
    rng = np.random.default_rng(193)
    os.makedirs(OUT, exist_ok=True)
    n = 0
    for person in ("p1", "p2"):
        for label in ("still", "walk", "run", "bike"):
            for k in range(3):
                t, imu = session(label, rng, variant=k % 2)
                path = os.path.join(OUT, f"{label}_{person}_fake{k}.csv")
                with open(path, "w") as f:
                    f.write("t,ax,ay,az,gx,gy,gz\n")
                    for ti, row in zip(t, imu):
                        f.write(f"{ti:.3f}," + ",".join(f"{v:.4f}" for v in row) + "\n")
                n += 1
    print(f"Wrote {n} synthetic recordings to {OUT}/")


if __name__ == "__main__":
    main()
