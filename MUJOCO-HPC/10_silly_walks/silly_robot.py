"""The silly walker: a LEGO Double Motor with a Technic beam on each output instead of a wheel.

This file builds the MuJoCo model in Python (no XML to edit). walk.py and mjlab_task/walk_task.py both use it.
Run it on its own to write silly_robot.xml, which you can open in MuJoCo's viewer (python -m mujoco.viewer):

    python3 silly_robot.py

Directions: x is forward, y is to the robot's left, z is up. The two motor outputs turn about the y axis.
A beam angle of 0 means the beam's long end points straight down; a positive speed swings that end backward,
which pushes the robot forward (like a wheel turning forward).

The shapes come from your CAD (silly_cad.npz, made by import_cad.py from sillywalks.stl); without it, simple boxes
sized by the numbers below are used. The motor's real top speed comes from real_robot.json once you have run
calibrate.py on the real robot. Weigh your motor and change MOTOR_MASS below.
"""
import json
import math
from pathlib import Path

import mujoco
import numpy as np

# ---- your CAD (made by import_cad.py from sillywalks.stl) -------------------------------------------------
# When silly_cad.npz is here, the motor and beams have the exact shapes of your Onshape robot, and the
# box-and-capsule sizes below (MOTOR_SIZE ... BEAM_MASS_PER_HOLE) are not used.
CAD_FILE = "silly_cad.npz"
MOTOR_LONG_END = "back"              # the motor sticks out 64 mm on one side of the outputs and 16 mm on the other:
                                     # "back" if that long end trails behind as it walks (ours does), "front" if it
                                     # leads. calibrate.py asks you, and its answer (in real_robot.json) wins.
BEAM_MASS = 0.002                    # kg per beam with its pins, weighed: whole robot 94 g - motor 90 g = 4 g for two

# ---- the Double Motor (ESTIMATES: measure yours) ----------------------------------------------------------
MOTOR_SIZE = (0.056, 0.040, 0.032)   # m: length (x, front to back), width (y, output to output), height (z)
MOTOR_MASS = 0.090                   # kg, weighed: the Double Motor with its battery (and its two output hubs)
AXLE_HEIGHT = 0.016                  # m: how far the outputs are above the bottom of the motor
AXLE_X = 0.0                         # m: where along the motor the outputs are (0 = middle, + = toward the front)
MAX_SPEED = 15.0                     # rad/s at speed 100% (about 145 rpm)
STALL_TORQUE = 0.15                  # N*m the motor can push with at most
MOTOR_KV = 0.05                      # N*m per rad/s: how hard the motor pushes toward the speed it was told
MOTOR_FRICTION = 0.2                 # the motor's plastic sliding on the floor (tile)
GEARBOX_INERTIA = 0.0005             # kg*m^2: how hard the gearbox makes the output to spin up. Without it the
                                     # 2 g beams are so light that the simulation blows up.

# ---- the beams (one on each output) ----------------------------------------------------------------------
BEAM_HOLES = 9                       # length in holes: a 9-hole beam is 72 mm long
AXLE_HOLE = 5                        # which hole goes on the motor output, counting from one end (5 = middle of 9)
BEAM_THICKNESS = 0.0072              # m (a Technic beam is 7.2 mm thick and 7.2 mm tall)
BEAM_GAP = 0.002                     # m between the side of the motor and the beam
BEAM_MASS_PER_HOLE = 0.00025         # kg (a 9-hole beam is about 2.3 g)
BEAM_FRICTION = 0.25                 # the beam ends on the floor: tile is slippery for LEGO plastic

HOLE = 0.008                         # m, one LEGO hole
LEFT, RIGHT = "left_beam", "right_beam"   # the joint (and motor) names

# ---- shared by the simulation and the real robot ---------------------------------------------------------
CONTROL_HZ = 20                      # the policy decides new beam speeds this many times a second. The real
                                     # Double Motor sends its sensor readings over Bluetooth every 25 ms at best,
                                     # so 20 times a second (every 50 ms) is what the real robot can keep up with.

# What calibrate.py measured on the real robot (it sits next to this file, or next to the script on the cluster).
REAL_ROBOT_FILE = "real_robot.json"
for _folder in (Path(__file__).resolve().parent, Path.cwd()):
    if (_folder / REAL_ROBOT_FILE).exists():
        REAL_ROBOT = json.loads((_folder / REAL_ROBOT_FILE).read_text())
        MAX_SPEED = REAL_ROBOT["max_speed_rad_s"]     # the real motor's speed at 100%, measured
        MOTOR_LONG_END = REAL_ROBOT.get("motor_long_end", MOTOR_LONG_END)
        break
else:
    REAL_ROBOT = None

CAD = None
for _folder in (Path(__file__).resolve().parent, Path.cwd()):
    if (_folder / CAD_FILE).exists():
        CAD = dict(np.load(_folder / CAD_FILE))
        break


def _cad_side(side: str) -> str:
    """Which side of the CAD becomes our `side`: turning the robot round (long end in front) swaps them."""
    return {"left": "right", "right": "left"}[side] if MOTOR_LONG_END == "front" else side


def _cad_part(key: str):
    """Vertices and faces of one CAD part, turned round if the motor's long end leads (MOTOR_LONG_END = "front")."""
    flip = np.array([-1.0, -1.0, 1.0]) if MOTOR_LONG_END == "front" else np.ones(3)
    return CAD[key + "_vert"] * flip, CAD[key + "_face"]


def beam_long_end() -> float:
    """How far below the outputs the robot's lowest point can be (the beam's far end), in metres."""
    if CAD is not None:
        return float(max(CAD["left_reach"], CAD["right_reach"], -CAD["motor_vert"][:, 2].min()))
    return (max(AXLE_HOLE - 1, BEAM_HOLES - AXLE_HOLE)) * HOLE + BEAM_THICKNESS / 2


def _add_mesh(spec, body, name, vert, face, **geom):
    spec.add_mesh(name=name, uservert=vert.flatten().tolist(), userface=face.flatten().tolist())
    return body.add_geom(name=name, type=mujoco.mjtGeom.mjGEOM_MESH, meshname=name, **geom)


def make_spec(motors: bool = True) -> mujoco.MjSpec:
    """The robot as an MjSpec. motors=False leaves out the motors (mjlab adds its own)."""
    spec = mujoco.MjSpec()
    spec.modelname = "silly_walker"
    spec.compiler.degree = False

    # The body is placed at the motor outputs (the axle), so the beams turn about its origin.
    motor = spec.worldbody.add_body(name="motor", pos=[0, 0, beam_long_end() + 0.005])
    motor.add_freejoint(name="root")
    # priority=1: in a contact with the floor, the robot's own friction is used. (Otherwise MuJoCo uses the larger of
    # the two, and the floor's default of 1.0 would hide every friction value below 1.)
    case = dict(mass=MOTOR_MASS, friction=[MOTOR_FRICTION, 0.005, 0.0001], priority=1, rgba=[0.95, 0.95, 0.95, 1])
    if CAD is not None:
        vert, face = _cad_part("motor")
        _add_mesh(spec, motor, "motor_case", vert, face, **case)
        front, half_width, top = vert[:, 0].max(), vert[:, 1].max(), vert[:, 2].max()
    else:
        length, width, height = MOTOR_SIZE
        motor.add_geom(name="motor_case", type=mujoco.mjtGeom.mjGEOM_BOX, size=[length / 2, width / 2, height / 2],
                       pos=[-AXLE_X, 0, height / 2 - AXLE_HEIGHT], **case)
        front, half_width, top = length / 2 - AXLE_X, width / 2, height - AXLE_HEIGHT
    # Where the motion sensor is (at the axle, lined up with the robot). The training task puts an accelerometer here.
    motor.add_site(name="imu")
    # Two "eyes" on the front, so you can tell which way it faces. They have no mass and never touch anything.
    for side in (1, -1):
        motor.add_geom(
            type=mujoco.mjtGeom.mjGEOM_SPHERE, size=[0.005, 0, 0], pos=[front, side * half_width / 2, 0.7 * top],
            contype=0, conaffinity=0, mass=0, rgba=[0.1, 0.1, 0.1, 1],
        )

    for name, side, colour in ((LEFT, "left", [0.1, 0.45, 0.85, 1]), (RIGHT, "right", [0.95, 0.55, 0.1, 1])):
        sign = 1 if side == "left" else -1
        grip = dict(friction=[BEAM_FRICTION, 0.005, 0.0001], priority=1, rgba=colour)
        looks = dict(contype=0, conaffinity=0, mass=0, rgba=[0.3, 0.3, 0.3, 1])   # seen, but no weight or contact
        if CAD is not None:
            # The beam, its output hub and any pins, from your CAD. Only the beam itself touches the floor.
            src = _cad_side(side)
            beam = motor.add_body(name=name + "_link", pos=[0, sign * abs(float(CAD[f"{src}_pivot"][1])), 0])
            _add_mesh(spec, beam, name + "_geom", *_cad_part(f"{src}_beam"), mass=BEAM_MASS, **grip)
            _add_mesh(spec, beam, name + "_hub", *_cad_part(f"{src}_hub"), **looks)
            pins = sorted(k[:-5] for k in CAD if k.startswith(f"{src}_pin") and k.endswith("_vert"))
            for i, pin in enumerate(pins):
                _add_mesh(spec, beam, f"{name}_pin{i}", *_cad_part(pin), **looks)
        else:
            # A straight beam: a capsule from hole 1 to the last hole, with the axle hole at the origin.
            beam = motor.add_body(name=name + "_link",
                                  pos=[0, sign * (MOTOR_SIZE[1] / 2 + BEAM_GAP + BEAM_THICKNESS / 2), 0])
            beam.add_geom(name=name + "_geom", type=mujoco.mjtGeom.mjGEOM_CAPSULE,
                          fromto=[0, 0, (AXLE_HOLE - 1) * HOLE, 0, 0, -(BEAM_HOLES - AXLE_HOLE) * HOLE],
                          size=[BEAM_THICKNESS / 2, 0, 0], mass=BEAM_MASS_PER_HOLE * BEAM_HOLES, **grip)
            beam.add_geom(type=mujoco.mjtGeom.mjGEOM_CYLINDER, size=[0.0025, BEAM_THICKNESS / 2 + 0.001, 0],
                          quat=[math.cos(math.pi / 4), math.sin(math.pi / 4), 0, 0], **looks)
        beam.add_joint(name=name, type=mujoco.mjtJoint.mjJNT_HINGE, axis=[0, 1, 0], damping=0.0005,
                       armature=GEARBOX_INERTIA)
        if motors:
            # A speed-controlled motor: it pushes toward the speed it is told, up to STALL_TORQUE.
            act = spec.add_actuator(name=name, target=name, trntype=mujoco.mjtTrn.mjTRN_JOINT)
            act.set_to_velocity(kv=MOTOR_KV)
            act.ctrlrange = [-MAX_SPEED, MAX_SPEED]
            act.forcelimited = True
            act.forcerange = [-STALL_TORQUE, STALL_TORQUE]
    return spec


def add_scene(spec: mujoco.MjSpec) -> mujoco.MjSpec:
    """Add a floor, a sky, lights and the physics settings (for walk.py; mjlab sets its own in walk_task.py)."""
    spec.option.timestep = 0.002
    spec.option.integrator = mujoco.mjtIntegrator.mjINT_IMPLICITFAST   # steadier with fast motors on light parts
    spec.visual.headlight.ambient = [0.4, 0.4, 0.4]
    spec.visual.headlight.diffuse = [0.6, 0.6, 0.6]
    spec.add_texture(name="sky", type=mujoco.mjtTexture.mjTEXTURE_SKYBOX, builtin=mujoco.mjtBuiltin.mjBUILTIN_GRADIENT,
                     rgb1=[0.9, 0.95, 1], rgb2=[0.6, 0.75, 0.9], width=256, height=256)
    spec.add_texture(name="grid", type=mujoco.mjtTexture.mjTEXTURE_2D, builtin=mujoco.mjtBuiltin.mjBUILTIN_CHECKER,
                     rgb1=[0.85, 0.85, 0.85], rgb2=[0.7, 0.7, 0.7], width=256, height=256)
    grid = spec.add_material(name="grid", texrepeat=[20, 20], reflectance=0.1)
    grid.textures[mujoco.mjtTextureRole.mjTEXROLE_RGB] = "grid"
    spec.worldbody.add_light(pos=[0, -1, 3], dir=[0, 0.3, -1], diffuse=[0.8, 0.8, 0.8])
    spec.worldbody.add_geom(name="floor", type=mujoco.mjtGeom.mjGEOM_PLANE, size=[5, 5, 0.1], material="grid")
    return spec


if __name__ == "__main__":
    spec = add_scene(make_spec())
    model = spec.compile()
    with open("silly_robot.xml", "w") as f:
        f.write(spec.to_xml())
    print(f"wrote silly_robot.xml: total mass {model.body_subtreemass[1] * 1000:.0f} g, "
          f"beam reaches {beam_long_end() * 1000:.0f} mm from the axle")
    print("shapes: " + (f"your CAD ({CAD_FILE}), motor's long end at the {MOTOR_LONG_END}" if CAD is not None
                        else "simple boxes (run import_cad.py to use your CAD)"))
    print(f"motor top speed {MAX_SPEED:.1f} rad/s: "
          + ("measured on your robot (real_robot.json)" if REAL_ROBOT else "an estimate (run calibrate.py to measure it)"))
