"""
Settings shared by the UNO Q app and the laptop scripts (the laptop scripts
import this file straight from unoq_app/python/, so there is only one copy).
"""

# Activities, in the order used for LED matrix letters (S, W, R, B).
LABELS = ["still", "walk", "run", "bike"]

# --- Sampling (must match sketch.ino) ---------------------------------------
SAMPLE_HZ = 100
ACCEL_COUNTS_PER_G = 4096.0      # MPU-6050 at +/-8 g
GYRO_COUNTS_PER_DPS = 32.8       # MPU-6050 at +/-1000 deg/s
COLUMNS = ["ax", "ay", "az", "gx", "gy", "gz"]   # g, g, g, deg/s, deg/s, deg/s

# --- Windows ----------------------------------------------------------------
WINDOW_S = 2.0                   # each prediction looks at the last 2 s
STEP_S = 0.5                     # a new prediction every 0.5 s
TRIM_S = 3.0                     # drop this much from each end of a recording
VOTE_WINDOWS = 3                 # live output = most common of the last 3 answers

WINDOW_N = int(WINDOW_S * SAMPLE_HZ)
STEP_N = int(STEP_S * SAMPLE_HZ)

# --- MQTT ---------------------------------------------------------------------
# test.mosquitto.org (used by ping-pong and sumo) stopped answering on port
# 1883 from Tufts in Oct 2026, so this uses EMQX's free public broker instead.
MQTT_BROKER = "broker.emqx.io"
MQTT_PORT = 1883
MQTT_BASE = "ME193/Rogers/Fiona/gait"
TOPIC_CMD = MQTT_BASE + "/cmd"         # laptop -> board: "record walk", "stop", "classify", ...
TOPIC_RAW = MQTT_BASE + "/raw"         # board -> laptop while recording: CSV lines
TOPIC_RESULT = MQTT_BASE + "/result"   # board -> anyone: JSON with both models' answers
TOPIC_STATUS = MQTT_BASE + "/status"   # board -> anyone: mode, sample rate, errors
