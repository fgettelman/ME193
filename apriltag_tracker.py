"""
Camera-guided AprilTag centering for a LEGO Double Motor car.

Setup:
  - An iPhone rides on the car as its camera, streaming to the Mac over
    Continuity Camera (same feed iphone_video.py shows).
  - An AprilTag (family 'tag36h11' by default) sits still in the world,
    in front of the car, facing it.
  - The Double Motor drives the car's two wheels. This script watches the
    tag's horizontal position in the camera feed and steers the car
    (differential drive) to keep the tag centered in frame -- i.e. to aim
    the car at the tag.

Before the first run:
  1. python enable_continuity_camera.py   (once, per interpreter)
  2. python list_cameras.py               (confirm the phone is listed)
  3. python iphone_video.py               (confirm the feed looks right)
The phone-side checklist lives at the top of iphone_video.py.

This script will not silently fall back to the Mac's built-in webcam --
that camera isn't on the car, so its view would steer the car nowhere.
If the phone isn't found it stops and tells you.

Run:
    python apriltag_tracker.py
Press 'q' in the video window to stop.
"""

import time

import cv2
import legoeducation as le
from pupil_apriltags import Detector

from camera import open_camera

# ---------------------------------------------------------------------------
# Configuration - edit these for your hardware / setup
# ---------------------------------------------------------------------------

# Double Motor connection info (from its Connection Card)
CARD_COLOR = le.LEGO_COLOR_BLUE  # Change to your card's color
CARD_SERIAL = '3685'              # Change to your card's 4-digit serial number

# Which camera to use. CAMERA_NAME is matched against the camera names macOS
# reports (run list_cameras.py to see them), so the phone is found no matter
# what index it lands on -- and the index does move around depending on whether
# the phone was connected when the script started.
# Set CAMERA_INDEX to a number to override the name match entirely.
CAMERA_NAME = 'iPhone'
CAMERA_INDEX = None

# Require CAMERA_NAME to actually be there. The car steers by what the on-board
# camera sees, so quietly falling back to the Mac's built-in webcam would drive
# the car off a view bolted to the desk. Set False only if you're deliberately
# testing with the built-in camera (also set CAMERA_NAME = 'FaceTime').
REQUIRE_NAMED_CAMERA = True

# Optional capture resolution. None keeps the camera's default -- the iPhone
# hands over 1920x1080, which is more pixels than tag detection needs; dropping
# to 1280 x 720 speeds the loop up if detection is lagging. Changing this changes
# how fast frames arrive, so re-check the tuning below if you touch it.
CAMERA_WIDTH = None
CAMERA_HEIGHT = None

TAG_FAMILY = 'tag36h11'           # AprilTag family of the tag the car aims at

# Control tuning (full PID on the normalized horizontal error)
KP = 50.0             # Proportional gain: reacts to the current offset
KI = 0.0                # Integral gain: OFF. MIN_SPEED (below) already guarantees the car keeps
                         # closing the gap, so KI has no steady-state error left to fix — it only
                         # adds extra push right at the setpoint, which feeds oscillation.
KD = 10.0                # Derivative gain: damps overshoot/oscillation from fast-changing error
INTEGRAL_LIMIT = 1.0    # Anti-windup clamp on the accumulated integral term
BASE_SPEED = 0        # Constant forward speed (0-100) added to both wheels while tracking.
                       # Leave at 0 to only rotate in place to center the tag.
MAX_SPEED = 100        # Clamp applied to each wheel's final speed (-100 to 100)
# Must be wide enough that the car can actually stop inside it. Continuous driving can't
# reliably land inside anything tighter than about SLOWDOWN_ZONE's low end, but the pulsed
# creep (FINE_ZONE, below) takes much smaller steps near the target, so this can be tight.
DEADBAND = 0.015
LOST_TAG_TIMEOUT = 0.5  # Seconds without a detection before stopping the motors

# Adaptive gain scheduling: whenever the error crosses zero (the dot swung past
# the line and overshot), back off KP and add extra damping via KD; when it holds
# steady with no overshoot, slowly restore the base gains.
ADAPT_DECAY = 0.85       # KP multiplier applied to the running scale on each overshoot
ADAPT_RECOVERY = 1.01    # Per-frame growth of the KP scale back toward 1.0 when stable
KP_SCALE_MIN = 0.3       # Floor on how much KP can be damped down
KD_BOOST = 1.15          # KD multiplier applied to the running scale on each overshoot
KD_SCALE_MAX = 2.5       # Ceiling on how much extra damping can be added
OVERSHOOT_MIN_ERROR = 0.015  # Ignore sign flips smaller than this (camera jitter, not a real overshoot).
                              # Kept below DEADBAND's neighborhood so genuine bounces near the
                              # target still get damped instead of staying at full responsiveness.

# Real hardware, not just math: a motor commanded below its minimum effective
# speed just sits there (not enough torque to overcome friction), which looks
# like the car "stopped" long before it's actually centered. MIN_SPEED guarantees
# every non-centered command is strong enough to actually move the car.
MIN_SPEED = 25

# Start decelerating once the normalized error falls inside this zone: the max
# allowed speed ramps linearly from MAX_SPEED (at the zone's outer edge) down to
# MIN_SPEED (right at FINE_ZONE), so the car eases in instead of cruising at full
# tilt right up to the line. Must be noticeably bigger than FINE_ZONE.
SLOWDOWN_ZONE = 0.15

# Inside this zone, stop driving continuously and switch to short pulses instead
# (see PULSE_ON_TIME / PULSE_OFF_TIME): continuous MIN_SPEED is still too strong a
# push to land inside a tight DEADBAND, so nudge briefly, coast, and re-measure
# instead of driving straight through the target. Must be > DEADBAND, < SLOWDOWN_ZONE.
FINE_ZONE = 0.08
PULSE_ON_TIME = 0.06   # seconds each nudge runs at MIN_SPEED
PULSE_OFF_TIME = 0.15  # seconds to coast/settle and get a fresh camera reading before the next nudge

# Limit how fast the commanded speed can change per second. Jumping straight to
# full speed when the tag starts far away builds up momentum the camera loop
# can't brake in time, which is the main cause of overshoot on long approaches.
SLEW_RATE = 250.0  # percent-speed per second

# Debug overlay: box drawn around the detected tag, and the on-screen readout of
# the tag's center in pixels. OpenCV colors are BGR, so this is red. Bump
# COORD_TEXT_SCALE if you're reading the window from further away.
BOX_COLOR = (0, 0, 255)
COORD_TEXT_SCALE = 1.2
COORD_TEXT_THICKNESS = 3

# If the car turns the wrong way to re-center the tag, flip this to True.
# False is correct for the camera-on-car setup above (tag drifts right in
# frame -> car turns right to face it). A rig with the camera off the car
# watching a tag mounted *on* the car needs the opposite sign.
REVERSE_STEERING = False


def clamp(value, lo, hi):
    return max(lo, min(hi, value))


def main():
    # --- Connect to the Double Motor -------------------------------------
    doublemotor = le.DoubleMotor()
    print("Connecting to the double motor...")
    doublemotor.connect(card_color=CARD_COLOR, card_serial=CARD_SERIAL)

    if not doublemotor.connected:
        print("Error connecting to Double Motor. Make sure it is turned on!")
        return

    print("Connected successfully!")

    # --- Set up the camera and AprilTag detector --------------------------
    try:
        cap = open_camera(
            prefer=CAMERA_NAME,
            index=CAMERA_INDEX,
            width=CAMERA_WIDTH,
            height=CAMERA_HEIGHT,
            required=REQUIRE_NAMED_CAMERA,
        )
    except RuntimeError as exc:
        print(f"Error: {exc}")
        doublemotor.disconnect()
        return

    detector = Detector(families=TAG_FAMILY)

    last_seen_time = None
    integral = 0.0
    prev_error = 0.0
    prev_time = None
    kp_scale = 1.0
    kd_scale = 1.0
    prev_steering = 0.0
    pulse_active_until = 0.0
    next_pulse_ready_time = 0.0
    pulse_direction = 0.0

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                print("Error: failed to read frame from camera.")
                break

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            frame_height, frame_width = gray.shape
            frame_center_x = frame_width / 2.0

            detections = detector.detect(gray)

            if detections:
                # Track the largest tag (closest / most prominent one) if several are seen.
                tag = max(detections, key=lambda d: cv2.contourArea(d.corners.astype('float32')))
                tag_center_x = tag.center[0]
                last_seen_time = time.time()

                # Normalized horizontal error: -1 (tag at left edge) .. +1 (tag at right edge)
                error = (tag_center_x - frame_center_x) / frame_center_x
                if REVERSE_STEERING:
                    error = -error

                now = time.time()
                dt = (now - prev_time) if prev_time is not None else 0.0
                prev_time = now

                centered = abs(error) < DEADBAND
                if centered:
                    # Sitting on the line: don't let the integral keep accumulating.
                    integral = 0.0
                    derivative = 0.0
                    steering = 0.0
                elif abs(error) < FINE_ZONE:
                    # Fine pulsed creep: even MIN_SPEED driven continuously is too strong
                    # a push to land inside DEADBAND, so nudge briefly, coast, and
                    # re-measure instead of driving straight through the target.
                    derivative = 0.0
                    if now < pulse_active_until:
                        steering = pulse_direction * MIN_SPEED
                    elif now < next_pulse_ready_time:
                        steering = 0.0  # coasting between nudges, let the camera settle
                    else:
                        pulse_direction = 1.0 if error > 0 else -1.0
                        pulse_active_until = now + PULSE_ON_TIME
                        next_pulse_ready_time = pulse_active_until + PULSE_OFF_TIME
                        steering = pulse_direction * MIN_SPEED
                else:
                    # Adapt: a real overshoot (sign flip past a noise threshold, not
                    # just camera jitter near zero) damps down; otherwise slowly
                    # recover toward full responsiveness.
                    overshot = (
                        prev_error != 0.0
                        and abs(prev_error) > OVERSHOOT_MIN_ERROR
                        and (error > 0) != (prev_error > 0)
                    )
                    if overshot:
                        kp_scale = max(KP_SCALE_MIN, kp_scale * ADAPT_DECAY)
                        kd_scale = min(KD_SCALE_MAX, kd_scale * KD_BOOST)
                    else:
                        kp_scale = min(1.0, kp_scale * ADAPT_RECOVERY)
                        kd_scale = max(1.0, kd_scale * (2.0 - ADAPT_RECOVERY))

                    integral = clamp(integral + error * dt, -INTEGRAL_LIMIT, INTEGRAL_LIMIT)
                    derivative = (error - prev_error) / dt if dt > 0 else 0.0
                    steering = clamp(
                        (KP * kp_scale) * error + KI * integral + (KD * kd_scale) * derivative,
                        -MAX_SPEED, MAX_SPEED,
                    )

                    # Slew-rate limit: cap how much the command can jump in one frame so a
                    # far-away start ramps up smoothly instead of slamming to full speed
                    # and overshooting from momentum the camera loop can't react to in time.
                    max_delta = SLEW_RATE * dt if dt > 0 else SLEW_RATE / 30.0
                    steering = clamp(steering, prev_steering - max_delta, prev_steering + max_delta)
                    steering = clamp(steering, -MAX_SPEED, MAX_SPEED)

                    # Landing zone: taper the max allowed speed down toward MIN_SPEED
                    # as the error approaches FINE_ZONE, so it slows on approach instead
                    # of holding full speed right up until pulsed creep takes over.
                    zone_span = max(SLOWDOWN_ZONE - FINE_ZONE, 1e-6)
                    taper = clamp((abs(error) - FINE_ZONE) / zone_span, 0.0, 1.0)
                    max_allowed = MIN_SPEED + (MAX_SPEED - MIN_SPEED) * taper
                    steering = clamp(steering, -max_allowed, max_allowed)

                    # Reset pulse timing so a fresh pulse starts cleanly if/when we
                    # cross back into FINE_ZONE, rather than resuming mid-cycle.
                    pulse_active_until = 0.0
                    next_pulse_ready_time = 0.0
                prev_error = error
                prev_steering = steering

                # Differential drive: turn toward the side the tag drifted to.
                left_speed = clamp(BASE_SPEED + steering, -MAX_SPEED, MAX_SPEED)
                right_speed = clamp(BASE_SPEED - steering, -MAX_SPEED, MAX_SPEED)

                if not centered and abs(error) >= FINE_ZONE:
                    # Guarantee real motion: a command weaker than the motor's minimum
                    # effective speed won't overcome friction, so the car just sits there.
                    # (Skipped inside FINE_ZONE — the pulse's coast phase is deliberately 0.)
                    if 0 < left_speed < MIN_SPEED:
                        left_speed = MIN_SPEED
                    elif -MIN_SPEED < left_speed < 0:
                        left_speed = -MIN_SPEED
                    if 0 < right_speed < MIN_SPEED:
                        right_speed = MIN_SPEED
                    elif -MIN_SPEED < right_speed < 0:
                        right_speed = -MIN_SPEED

                doublemotor.motor_run(
                    direction=le.MOTOR_MOVE_DIRECTION_CLOCKWISE,
                    motor=le.MOTOR_LEFT,
                    speed=int(left_speed),
                    blocking=False,
                )
                doublemotor.motor_run(
                    direction=le.MOTOR_MOVE_DIRECTION_CLOCKWISE,
                    motor=le.MOTOR_RIGHT,
                    speed=int(right_speed),
                    blocking=False,
                )

                # --- Debug overlay ---
                # Red box around the tag; red dot = tag center, which turns green
                # once it lands on the blue center line.
                dot_color = (0, 255, 0) if centered else (0, 0, 255)
                for pt_a, pt_b in zip(tag.corners, tag.corners[[1, 2, 3, 0]]):
                    cv2.line(frame, tuple(pt_a.astype(int)), tuple(pt_b.astype(int)), BOX_COLOR, 2)
                cv2.circle(frame, (int(tag_center_x), int(tag.center[1])), 5, dot_color, -1)

                # Tag center in pixels, big enough to read from across the room.
                # Drawn just above the tag, nudged back inside the frame when the
                # tag is near an edge so the text never runs off screen.
                coords = f"({tag_center_x:.0f}, {tag.center[1]:.0f})"
                (text_w, text_h), _ = cv2.getTextSize(
                    coords, cv2.FONT_HERSHEY_SIMPLEX, COORD_TEXT_SCALE, COORD_TEXT_THICKNESS)
                text_x = clamp(int(tag_center_x - text_w / 2), 5, max(5, frame_width - text_w - 5))
                text_y = clamp(int(tag.center[1]) - 20, text_h + 5, frame_height - 5)
                cv2.putText(frame, coords, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX,
                            COORD_TEXT_SCALE, BOX_COLOR, COORD_TEXT_THICKNESS)

                if centered:
                    status = "CENTERED"
                elif abs(error) < FINE_ZONE:
                    status = f"error={error:+.3f} PULSE"
                else:
                    status = f"error={error:+.2f} kp_x{kp_scale:.2f} kd_x{kd_scale:.2f}"
                cv2.putText(frame, f"{status} L={left_speed:.0f} R={right_speed:.0f}",
                            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, dot_color, 2)
            else:
                centered = False
                # No tag visible: stop if it's been missing too long, and reset the
                # integral/derivative state so there's no stale kick when it reappears.
                if last_seen_time is None or (time.time() - last_seen_time) > LOST_TAG_TIMEOUT:
                    doublemotor.motor_stop(motor=le.MOTOR_BOTH, blocking=False)
                    integral = 0.0
                    prev_error = 0.0
                    prev_time = None
                    kp_scale = 1.0
                    kd_scale = 1.0
                    prev_steering = 0.0
                    pulse_active_until = 0.0
                    next_pulse_ready_time = 0.0
                cv2.putText(frame, "Tag not found", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

            # Blue center line; turns green while the tag's red dot is centered on it.
            line_color = (0, 255, 0) if centered else (255, 0, 0)
            cv2.line(frame, (int(frame_center_x), 0), (int(frame_center_x), frame_height), line_color, 1)
            cv2.imshow("AprilTag Tracker", frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    finally:
        print("Stopping motors and disconnecting.")
        doublemotor.motor_stop(motor=le.MOTOR_BOTH)
        doublemotor.disconnect()
        cap.release()
        cv2.destroyAllWindows()
        print("Done!")


if __name__ == "__main__":
    main()
