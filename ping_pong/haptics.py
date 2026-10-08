"""
Haptic feedback -- the "something new" for this project.

- Rumble: motor A spins an off-center weight, so the paddle buzzes in your
  hand. It buzzes softly and gets stronger as the ball gets closer, so you
  can *feel* when to swing.
- Thunk: on a hit, motor B swings a short arm into your hand and back using
  a PD position loop (same Kp / Kd-on-measurement idea as 9-17.py).
- Hub light shows the level color; hub beeps for hit / miss / milestones.

Every BLE command goes through one worker thread, so a buzz pattern never
stalls the camera loop and commands never interleave.
"""

import queue
import threading
import time

import legoeducation as le

import config


class Haptics:
    def __init__(self, dm):
        self.dm = dm
        self.jobs = queue.Queue()
        self._proximity = 0.0
        self._sent_rumble = None
        self._running = True
        self._thread = threading.Thread(target=self._worker, daemon=True)
        self._thread.start()

    # --- public API (called from the game loop, never blocks) ---------------

    def hit(self):        self.jobs.put(self._hit)
    def miss(self):       self.jobs.put(self._miss)
    def milestone(self):  self.jobs.put(self._milestone)
    def serve(self):      self.jobs.put(self._serve)
    def confirm(self):    self.jobs.put(self._confirm)

    def level_light(self, color):
        self.jobs.put(lambda: self.dm.light_color(color, blocking=False))

    def proximity(self, level):
        """0..1, how close the incoming ball is. Applied between jobs."""
        self._proximity = level

    def close(self):
        self._running = False
        self._thread.join(timeout=2)
        self._rumble(0)
        self.dm.motor_stop(motor=config.THUNK_MOTOR)

    # --- worker ------------------------------------------------------------

    def _worker(self):
        while self._running:
            try:
                job = self.jobs.get(timeout=0.05)
            except queue.Empty:
                self._apply_proximity()
                continue
            try:
                job()
            except Exception as e:   # a dropped BLE command shouldn't kill the game
                print(f"haptics: {e}")
            self._sent_rumble = None  # force proximity to re-apply after a pattern

    def _apply_proximity(self):
        p = self._proximity
        speed = 0 if p < 0.3 else int(config.PROXIMITY_MIN_SPEED +
                                      (config.PROXIMITY_MAX_SPEED - config.PROXIMITY_MIN_SPEED) * p)
        # Only send when it changes noticeably, to keep the BLE link free.
        if self._sent_rumble is None or abs(speed - self._sent_rumble) >= 5:
            self._rumble(speed)

    def _rumble(self, speed):
        if speed <= 0:
            self.dm.motor_stop(motor=config.RUMBLE_MOTOR, blocking=False)
        else:
            self.dm.motor_run(motor=config.RUMBLE_MOTOR, speed=speed, blocking=False)
        self._sent_rumble = speed

    # --- patterns ----------------------------------------------------------

    def _hit(self):
        self._rumble(config.HIT_RUMBLE_SPEED)
        self.dm.beep(frequency=880, blocking=False)
        self._thunk()
        time.sleep(max(0.0, config.HIT_RUMBLE_MS / 1000 - 2 * config.THUNK_PHASE_S))
        self._rumble(0)

    def _miss(self):
        self.dm.beep(pattern=le.SOUND_PATTERN_BEEP_TRIPLE, frequency=220, blocking=False)
        for _ in range(config.MISS_PULSES):
            self._rumble(config.MISS_RUMBLE_SPEED)
            time.sleep(0.25)
            self._rumble(0)
            time.sleep(0.15)

    def _milestone(self):
        self.dm.beep(pattern=le.SOUND_PATTERN_BEEP_UP_MIDDLE_DOWN, frequency=1200, blocking=False)
        self.dm.light_color(le.LEGO_COLOR_MAGENTA, pattern=le.LIGHT_PATTERN_SHORT_BLINK, blocking=False)

    def _serve(self):
        self.dm.beep(frequency=660, blocking=False)

    def _confirm(self):
        self.dm.beep(pattern=le.SOUND_PATTERN_BEEP_DOUBLE, frequency=990, blocking=False)

    def _thunk(self):
        """Arm out to THUNK_DEG and back to 0 with a PD loop on motor B."""
        for target in (config.THUNK_DEG, 0):
            self._pd_move(target, config.THUNK_PHASE_S)
        self.dm.motor_stop(motor=config.THUNK_MOTOR, blocking=False)

    def _pd_move(self, target, duration):
        motor = config.THUNK_MOTOR
        end = time.monotonic() + duration
        prev_pos, prev_t = None, time.monotonic()
        while time.monotonic() < end:
            now = time.monotonic()
            pos = float(self.dm.motor[motor].position)
            dt = max(1e-3, now - prev_t)
            # Derivative on measurement (not error), same reason as 9-17.py.
            vel = 0.0 if prev_pos is None else (pos - prev_pos) / dt
            speed = config.THUNK_KP * (target - pos) - config.THUNK_KD * vel
            speed = int(max(-100, min(100, speed)))
            self.dm.motor_run(motor=motor, speed=speed, blocking=False)
            prev_pos, prev_t = pos, now
            time.sleep(0.02)


class NullHaptics:
    """Stand-in for --sim mode (no Double Motor)."""

    def __getattr__(self, name):
        return lambda *args, **kwargs: None
