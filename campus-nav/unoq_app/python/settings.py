"""
Settings shared by the UNO Q app and the laptop scripts (the laptop scripts
import this file straight from unoq_app/python/, so there is only one copy).
"""

# --- Sampling (must match sketch.ino) ---------------------------------------
SAMPLE_HZ = 100
ACCEL_COUNTS_PER_G = 4096.0      # MPU-6050 at +/-8 g
GYRO_COUNTS_PER_DPS = 32.8       # MPU-6050 at +/-1000 deg/s

# --- Web page (the phone talks to the board directly) ------------------------
WEB_PORT = 8050                  # also listed in app.yaml

# --- The project's rule -------------------------------------------------------
MAX_WALKS = 10                   # full walks given to the navigator, DAgger walks included

DEFAULT_DEST = "Nolop"

# --- Heading (gyro yaw about "up") --------------------------------------------
GRAVITY_GAIN = 0.02              # how fast the "up" estimate trusts the accelerometer (per sample)
STILL_GYRO_DPS = 10.0            # gyro below this, accel ~1 g: standing still -> learn gyro bias (noise peaks ~8)
STILL_ACCEL_G = 0.12              # this sensor reads ~0.95 g at rest
BIAS_GAIN = 0.002                # per sample while still, after the first 3 s (~5 s time constant)

# --- Turns ------------------------------------------------------------------
TURN_DEG = 50.0                  # heading must change this much from the current straight...
TURN_SETTLE_DPS = 20.0           # ...and settle (turn rate below this, 1.5 s average) to count
DRIFT_ABSORB_DEG = 20.0          # small, slow heading changes are bends/drift, absorbed slowly

# --- Stairs -----------------------------------------------------------------
STAIR_CLASSES = ["flat", "up", "down"]
STAIR_STEP_M = 0.30              # a step on stairs moves you this far horizontally
FLIGHT_STRIDES = 2               # strides in a row to start / end a flight

# --- Navigator (behavior cloning) ---------------------------------------------
ACTIONS = ["straight", "left", "right", "stairs_up", "stairs_down", "arrived"]
LOOKAHEAD_M = 5.0                # an instruction shows up this far before the turn / stairs
TRACK_HZ = 5                     # rows per second in a walk's track (the policy's states)
