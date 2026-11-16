"""Turn your CAD export (an STL of the whole robot, from Onshape) into the parts the simulation uses.

    pip install trimesh            (once)
    python3 import_cad.py          # reads sillywalks.stl next to this file, writes silly_cad.npz

The STL must be the whole robot in millimetres, as Onshape exports it: the Double Motor, its two round output hubs,
a beam on each hub (and any pins). It is split into its separate pieces, and each is recognised by shape:
the biggest piece is the motor, the two flat discs on the motor's sides are the outputs, the next two biggest are the
beams, and anything small near an output turns with that output. It prints what it found: check it.

Directions are taken from the CAD: z is up (the studs), the outputs turn about the CAD x axis, and the motor's
long end points along CAD +y. silly_robot.py places the long end behind the outputs or in front of them
(MOTOR_LONG_END), whichever matches your real robot; calibrate.py helps you check.

silly_cad.npz holds the shapes in the simulation's own frame (metres, x forward, y left, z up, origin midway between
the outputs), so silly_robot.py can load them with numpy alone, on your laptop or on the cluster.
Run this again whenever you export a new STL.
"""
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
STL = HERE / "sillywalks.stl"
OUT = HERE / "silly_cad.npz"
DENSITY = 1050.0        # kg/m^3, LEGO's ABS plastic: used for the beams' mass (the motor you should weigh)


def main():
    try:
        import trimesh
    except ImportError:
        sys.exit("needs trimesh:  pip install trimesh")
    stl = Path(sys.argv[1]) if len(sys.argv) > 1 else STL
    mesh = trimesh.load(stl)
    pieces = sorted(mesh.split(only_watertight=False), key=lambda p: -abs(p.volume))
    print(f"{stl.name}: {len(pieces)} pieces, {mesh.extents.round(1)} mm overall")

    motor = pieces[0]
    lo, hi = motor.bounds
    # The outputs: flat discs just outside the motor's two x sides.
    discs = [p for p in pieces[1:] if min(p.extents) < 0.4 * max(p.extents) and p.extents[0] == min(p.extents)
             and p.extents[1] > 10
             and abs(p.extents[1] - p.extents[2]) < 1.0 and (p.bounds[1][0] <= lo[0] + 1 or p.bounds[0][0] >= hi[0] - 1)]
    # The two outputs turn on one axle, so their centres line up across the motor: pick the pair that does.
    pairs = [(a, b) for i, a in enumerate(discs) for b in discs[i + 1:]
             if np.linalg.norm(a.bounds.mean(0)[1:] - b.bounds.mean(0)[1:]) < 1.0
             and (a.bounds.mean(0)[0] - motor.bounds.mean(0)[0]) * (b.bounds.mean(0)[0] - motor.bounds.mean(0)[0]) < 0]
    if len(pairs) != 1:
        sys.exit(f"expected 2 round motor outputs facing each other on the motor's sides, found {len(pairs)} pairs. "
                 "Is the STL the whole robot, with the outputs turning about the CAD x axis?")
    hubs = sorted(pairs[0], key=lambda p: p.bounds.mean(0)[0])          # CAD -x side first
    centres = [h.bounds.mean(0) for h in hubs]
    axle_mid = (centres[0] + centres[1]) / 2
    beams = [p for p in pieces[1:] if not any(p is h for h in hubs)][:2]
    beams = sorted(beams, key=lambda p: p.bounds.mean(0)[0])
    small = [p for p in pieces[1:] if not any(p is q for q in hubs + beams)]

    # CAD -> simulation frame (MOTOR_LONG_END = "back"): long end of the motor (CAD +y) toward -x, z up.
    # x_sim = -y_cad, y_sim = +x_cad, z_sim = z_cad
    rot = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]], dtype=float)

    def to_sim(v):
        return (np.asarray(v) - axle_mid) @ rot.T / 1000.0

    out = {"motor_vert": to_sim(motor.vertices), "motor_face": motor.faces}
    print(f"  motor: {motor.extents.round(1)} mm, outputs {abs(centres[1][0] - centres[0][0]):.1f} mm apart, "
          f"{hi[1] - centres[0][1]:.1f} mm of motor on one side of them and {centres[0][1] - lo[1]:.1f} on the other")
    # The CAD -x side is the simulation's right side (y_sim = x_cad).
    for side, beam, hub, centre in (("right", beams[0], hubs[0], centres[0]), ("left", beams[1], hubs[1], centres[1])):
        extra = [p for p in small if abs(p.bounds.mean(0)[0] - centre[0]) < 15]
        pivot = to_sim(centre)
        vert = to_sim(beam.vertices) - pivot
        # Turn the beam about the axle so its farthest point hangs straight down: that is beam angle 0.
        far = vert[np.argmax(np.hypot(vert[:, 0], vert[:, 2]))]
        angle = np.arctan2(far[0], -far[2])                              # 0 = straight down, + = toward the front
        c, s = np.cos(angle), np.sin(angle)
        turn = np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])               # rotates `far` onto -z
        def local(v, turn=turn, pivot=pivot):
            return (to_sim(v) - pivot) @ turn.T
        out[f"{side}_pivot"] = pivot
        out[f"{side}_beam_vert"], out[f"{side}_beam_face"] = local(beam.vertices), beam.faces
        out[f"{side}_hub_vert"], out[f"{side}_hub_face"] = local(hub.vertices), hub.faces
        for i, p in enumerate(extra):
            out[f"{side}_pin{i}_vert"], out[f"{side}_pin{i}_face"] = local(p.vertices), p.faces
        volume = abs(beam.volume) + abs(hub.volume) + sum(abs(p.volume) for p in extra)      # mm^3
        out[f"{side}_mass"] = np.array(volume * 1e-9 * DENSITY)
        reach = np.max(np.hypot(out[f"{side}_beam_vert"][:, 0], out[f"{side}_beam_vert"][:, 2]))
        out[f"{side}_reach"] = np.array(reach)
        print(f"  {side} beam: {beam.extents.round(1)} mm, reaches {reach * 1000:.1f} mm from the axle, "
              f"{len(extra)} pin(s)")
    np.savez_compressed(OUT, **out)
    print(f"wrote {OUT.name}")


if __name__ == "__main__":
    main()
