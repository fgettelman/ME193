"""
Settings shared by the UNO Q app and the laptop scripts. The laptop scripts
import this file straight from unoq_app/python/, so there is only one copy and
the laptop and the board can't drift apart.
"""

# --- What the net answers ---------------------------------------------------
# Class k = k fingers. Class 0 = "no number": empty scene, a fist, or anything
# that isn't 1-5 fingers. The lock only ever accepts 1-5.
LABELS = ["0", "1", "2", "3", "4", "5"]
N_CLASSES = len(LABELS)

# --- Camera and images --------------------------------------------------------
# 640x480 on both the laptop and the board: many webcams crop the sensor for
# 16:9 modes, so a different resolution would change what the camera sees.
CAM_W, CAM_H = 640, 480
IMG = 64                  # the net sees the centre square of the frame at 64x64
SWAP_RB = False           # set True if the board's camera gives RGB instead of BGR

# --- Lock behaviour (lock.py) -------------------------------------------------
CODE_LENGTH = 3           # three numbers, 1-5, no two in a row the same
SMOOTH = 0.5              # smoothing of the probabilities between frames (0 = none)
CONF_MIN = 0.80           # smoothed probability needed to believe a count
HOLD_S = 0.8              # a count has to be held this long to be entered
SCAN_TIMEOUT_S = 20.0     # give up (stay locked) if nothing is entered for this long
UNLOCK_S = 30.0           # relock automatically after this long (or press the button)
DENIED_S = 2.0            # how long the "wrong code" display stays up
MAX_TRIES = 3             # wrong codes in a row before a lockout
LOCKOUT_S = 30.0          # first lockout; doubles after each further wrong code

# --- Servo ------------------------------------------------------------------
SERVO_LOCKED_DEG = 0
SERVO_OPEN_DEG = 90
