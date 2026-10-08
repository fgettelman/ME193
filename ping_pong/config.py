"""
All the knobs for virtual ping-pong in one place. Edit this file, not the
game code, to match your hardware and tune the feel of the game.
"""

import legoeducation as le

# --- Double Motor (the paddle) ---------------------------------------------
# Same connection-card pattern as 9-17.py / apriltag_tracker.py.
CARD_COLOR = le.LEGO_COLOR_BLUE   # Change to your card's color
CARD_SERIAL = '3685'              # Change to your card's 4-digit serial

# Motor A spins an off-center weight -> rumble. Motor B swings a short arm
# that taps the back of your hand -> "thunk" on contact (PD position move).
RUMBLE_MOTOR = le.MOTOR_LEFT
THUNK_MOTOR = le.MOTOR_RIGHT

# How often the hub streams IMU/motor notifications (ms). The library default
# is 100 ms (10 Hz), which is too slow to catch a quick swing.
NOTIFICATION_DELAY_MS = 20

# --- MQTT scoreboard --------------------------------------------------------
MQTT_BROKER = "test.mosquitto.org"
MQTT_PORT = 1883
MQTT_TOPIC = "ME193/Rogers/Fiona"
MQTT_HEARTBEAT_S = 5.0            # re-publish the current score this often

# --- Camera / pose ----------------------------------------------------------
CAMERA_WIDTH = 1280
CAMERA_HEIGHT = 720
PROCESS_WIDTH = 640               # frames are shrunk to this for pose + tags
# Which wrist holds the paddle: "right" or "left" (the player's own side).
# If the green wrist dot shows up on the wrong hand, flip this.
PADDLE_HAND = "right"
MIN_WRIST_VISIBILITY = 0.5
POSE_MODEL_URL = ("https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
                  "pose_landmarker_lite/float16/latest/pose_landmarker_lite.task")
POSE_MODEL_FILE = "pose_landmarker_lite.task"

# --- AprilTags --------------------------------------------------------------
TAG_FAMILY = "tag36h11"
TAG_START = 0
# tag id -> (level name, ball speed in depth-units/s, hub light color)
LEVELS = {
    1: ("Easy",   0.35, le.LEGO_COLOR_GREEN),
    2: ("Medium", 0.55, le.LEGO_COLOR_YELLOW),
    3: ("Hard",   0.80, le.LEGO_COLOR_RED),
}
DEFAULT_LEVEL = 1
TAG_STABLE_FRAMES = 10            # frames in a row before a tag "counts"

# --- Swing detection (IMU) --------------------------------------------------
CAL_STILL_S = 2.0                 # hold still this long to measure noise
CAL_SWINGS = 3                    # then swing this many times
GYRO_EMA_ALPHA = 0.4              # low-pass on gyro magnitude
SWING_REFRACTORY_S = 0.30         # one swing can't count twice
# Threshold = noise + this fraction of the way up to your typical swing peak.
SWING_THRESHOLD_FRACTION = 0.4

# --- Game feel --------------------------------------------------------------
HIT_RADIUS = 0.16                 # how close (normalized screen units) the wrist must be
SWING_WINDOW_S = 0.18             # swing must land within +/- this of ball arrival
SPEEDUP_PER_HIT = 1.05
MAX_SPEED = 1.4
K_OFFSET = 2.0                    # hitting off-center angles the return
K_TILT = 0.015                    # paddle roll (deg) angles the return
MAX_MISSES = 3
COUNTDOWN_S = 3.0
GAME_OVER_S = 4.0
MILESTONE_EVERY = 5               # celebrate every N hits in a row

# --- Haptics ----------------------------------------------------------------
HIT_RUMBLE_SPEED = 100
HIT_RUMBLE_MS = 120
MISS_RUMBLE_SPEED = 100
MISS_PULSES = 3
PROXIMITY_MAX_SPEED = 45          # rumble while the ball approaches, scaled 0..this
PROXIMITY_MIN_SPEED = 15          # below this the motor won't actually turn
THUNK_DEG = 40
THUNK_KP = 2.0                    # PD gains from the 9-17 tracker idea
THUNK_KD = 0.05
THUNK_PHASE_S = 0.12

# --- Whistle pause (optional, reuses the whistle_class.py detector) ---------
USE_WHISTLE = True
WHISTLE_SAMPLE_RATE = 44100
WHISTLE_CHUNK = 2048
WHISTLE_BAND = (1500.0, 3000.0)   # "right" + "speed_up" bands from whistle_class.py
WHISTLE_MIN_RMS = 2500.0 / 32768  # whistle_class used int16; sounddevice gives float32
WHISTLE_MIN_TONALITY = 15.0
WHISTLE_HOLD_CHUNKS = 4           # ~0.2 s of steady whistle to toggle pause
