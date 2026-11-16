"""The silly walker: a LEGO Double Motor with a Technic beam on each output instead of a wheel.

This file builds the MuJoCo model in Python (no XML to edit). walk.py and mjlab_task/walk_task.py both use it.
Run it on its own to write silly_robot.xml, which you can open in MuJoCo's viewer (python -m mujoco.viewer):

    python3 silly_robot.py

Directions: x is forward, y is to the robot's left, z is up. The two motor outputs turn about the y axis.
A beam angle of 0 means the beam's long end points straight down; a positive speed swings that end backward,
which pushes the robot forward (like a wheel turning forward).

Measure your own parts and change the numbers below. 1 LEGO hole (stud) = 8 mm.
"""
import math

import mujoco

# ---- the Double Motor (ESTIMATES: measure yours) ----------------------------------------------------------
MOTOR_SIZE = (0.056, 0.040, 0.032)   # m: length (x, front to back), width (y, output to output), height (z)
MOTOR_MASS = 0.090                   # kg (weigh it, with its battery)
AXLE_HEIGHT = 0.016                  # m: how far the outputs are above the bottom of the motor
AXLE_X = 0.0                         # m: where along the motor the outputs are (0 = middle, + = toward the front)
MAX_SPEED = 15.0                     # rad/s at speed 100% (about 145 rpm)
STALL_TORQUE = 0.15                  # N*m the motor can push with at most
MOTOR_KV = 0.05                      # N*m per rad/s: how hard the motor pushes toward the speed it was told
MOTOR_FRICTION = 0.3                 # the motor's plastic sliding on the floor
GEARBOX_INERTIA = 0.0005             # kg*m^2: how hard the gearbox makes the output to spin up. Without it the
                                     # 2 g beams are so light that the simulation blows up.

# ---- the beams (one on each output) ----------------------------------------------------------------------
BEAM_HOLES = 9                       # length in holes: a 9-hole beam is 72 mm long
AXLE_HOLE = 5                        # which hole goes on the motor output, counting from one end (5 = middle of 9)
BEAM_THICKNESS = 0.0072              # m (a Technic beam is 7.2 mm thick and 7.2 mm tall)
BEAM_GAP = 0.002                     # m between the side of the motor and the beam
BEAM_MASS_PER_HOLE = 0.00025         # kg (a 9-hole beam is about 2.3 g)
BEAM_FRICTION = 0.8                  # the beam ends gripping the floor

HOLE = 0.008                         # m, one LEGO hole
LEFT, RIGHT = "left_beam", "right_beam"   # the joint (and motor) names


def beam_long_end() -> float:
    """Distance from the motor output to the far end of the longer side of the beam, in metres."""
    return (max(AXLE_HOLE - 1, BEAM_HOLES - AXLE_HOLE)) * HOLE + BEAM_THICKNESS / 2


def make_spec(motors: bool = True) -> mujoco.MjSpec:
    """The robot as an MjSpec. motors=False leaves out the motors (mjlab adds its own)."""
    spec = mujoco.MjSpec()
    spec.modelname = "silly_walker"
    spec.compiler.degree = False

    # The body is placed at the motor outputs (the axle), so the beams turn about its origin.
    motor = spec.worldbody.add_body(name="motor", pos=[0, 0, beam_long_end() + 0.005])
    motor.add_freejoint(name="root")
    length, width, height = MOTOR_SIZE
    motor.add_geom(
        name="motor_case", type=mujoco.mjtGeom.mjGEOM_BOX,
        size=[length / 2, width / 2, height / 2], pos=[-AXLE_X, 0, height / 2 - AXLE_HEIGHT],
        mass=MOTOR_MASS, friction=[MOTOR_FRICTION, 0.005, 0.0001], rgba=[0.95, 0.95, 0.95, 1],
    )
    # Two "eyes" on the front, so you can tell which way it faces. They have no mass and never touch anything.
    for side in (1, -1):
        motor.add_geom(
            type=mujoco.mjtGeom.mjGEOM_SPHERE, size=[0.006, 0, 0],
            pos=[length / 2 - AXLE_X, side * width / 4, height / 2 - AXLE_HEIGHT + height / 5],
            contype=0, conaffinity=0, mass=0, rgba=[0.1, 0.1, 0.1, 1],
        )

    # One beam on each side. The capsule runs from hole 1 to the last hole, with the axle hole at the origin.
    up = (AXLE_HOLE - 1) * HOLE
    down = (BEAM_HOLES - AXLE_HOLE) * HOLE
    y = width / 2 + BEAM_GAP + BEAM_THICKNESS / 2
    for name, side, colour in ((LEFT, 1, [0.1, 0.45, 0.85, 1]), (RIGHT, -1, [0.95, 0.55, 0.1, 1])):
        beam = motor.add_body(name=name + "_link", pos=[0, side * y, 0])
        beam.add_joint(name=name, type=mujoco.mjtJoint.mjJNT_HINGE, axis=[0, 1, 0], damping=0.0005,
                       armature=GEARBOX_INERTIA)
        beam.add_geom(
            name=name + "_geom", type=mujoco.mjtGeom.mjGEOM_CAPSULE,
            fromto=[0, 0, up, 0, 0, -down], size=[BEAM_THICKNESS / 2, 0, 0],
            mass=BEAM_MASS_PER_HOLE * BEAM_HOLES, friction=[BEAM_FRICTION, 0.005, 0.0001], rgba=colour,
        )
        # the axle, so you can see where the beam is held
        beam.add_geom(
            type=mujoco.mjtGeom.mjGEOM_CYLINDER, size=[0.0025, BEAM_THICKNESS / 2 + 0.001, 0],
            quat=[math.cos(math.pi / 4), math.sin(math.pi / 4), 0, 0],
            contype=0, conaffinity=0, mass=0, rgba=[0.3, 0.3, 0.3, 1],
        )
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
