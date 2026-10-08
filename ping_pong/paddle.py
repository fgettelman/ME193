"""
The paddle: a Double Motor held in your hand. A background thread reads its
IMU and turns gyro spikes into swing events.

Swing threshold is calibrated per player, because raw gyro units (and how
hard people swing) vary: hold still to measure noise, then swing a few times
to measure your typical peak; the threshold sits between the two.
"""

import math
import statistics
import threading
import time
from collections import deque

import legoeducation as le

import config
from lelib import doubleMotor


class SwingDetector:
    """Pure logic: feed it (time, gyro magnitude) samples, it reports swings."""

    def __init__(self):
        self.threshold = math.inf     # no swings until calibrated
        self.swing_times = deque(maxlen=50)
        self._above = False
        self._last_swing = -math.inf

    def feed(self, t, mag):
        swung = False
        if mag >= self.threshold and not self._above and t - self._last_swing >= config.SWING_REFRACTORY_S:
            self.swing_times.append(t)
            self._last_swing = t
            swung = True
        self._above = mag >= self.threshold
        return swung

    def swung_between(self, t0, t1):
        return any(t0 <= t <= t1 for t in self.swing_times)


class Calibrator:
    """Still phase -> noise level. Swing phase -> typical peak. Then a threshold."""

    def __init__(self):
        self.still = []
        self.peaks = []
        self.noise = None
        self._cur_peak = 0.0
        self._last_peak_t = -math.inf

    def add_still(self, mag):
        self.still.append(mag)

    def finish_still(self):
        mean = statistics.fmean(self.still) if self.still else 0.0
        std = statistics.pstdev(self.still) if len(self.still) > 1 else 0.0
        self.noise = mean + 4 * std + 1e-6

    def add_swing_sample(self, t, mag):
        """Track peaks well above the noise; returns how many swings so far."""
        if mag > 3 * self.noise:
            self._cur_peak = max(self._cur_peak, mag)
        elif self._cur_peak and t - self._last_peak_t >= config.SWING_REFRACTORY_S:
            self.peaks.append(self._cur_peak)
            self._last_peak_t = t
            self._cur_peak = 0.0
        return len(self.peaks)

    def threshold(self):
        peak = statistics.median(self.peaks)
        return self.noise + config.SWING_THRESHOLD_FRACTION * (peak - self.noise)


class Paddle:
    def __init__(self):
        self.dm = doubleMotor()
        self.detector = SwingDetector()
        self.calibrator = None
        self.mode = "idle"          # "idle" | "still" | "swing" | "play"
        self.mag = 0.0              # smoothed gyro magnitude, for the debug bar
        self.roll0 = 0.0
        self.lock = threading.Lock()
        self._running = False
        self._thread = None

    def connect(self):
        print("Connecting to Double Motor (paddle)...")
        self.dm.connect(card_serial=config.CARD_SERIAL, card_color=config.CARD_COLOR)
        self.dm.device_notification_request(config.NOTIFICATION_DELAY_MS)
        self.dm.reset_heading()
        self.dm.motor_reset_relative_position(motor=config.THUNK_MOTOR, position=0)
        print("Paddle connected.")
        self._running = True
        self._thread = threading.Thread(target=self._imu_loop, daemon=True)
        self._thread.start()

    def _read_gyro_mag(self):
        imu = self.dm.imu_device
        g = (imu.gyroscopeX, imu.gyroscopeY, imu.gyroscopeZ)
        if any(isinstance(v, float) and math.isnan(v) for v in g):
            return None   # no notification yet
        return math.sqrt(sum(float(v) ** 2 for v in g))

    def _imu_loop(self):
        alpha = config.GYRO_EMA_ALPHA
        while self._running:
            raw = self._read_gyro_mag()
            if raw is not None:
                now = time.monotonic()
                with self.lock:
                    self.mag = alpha * raw + (1 - alpha) * self.mag
                    if self.mode == "still":
                        self.calibrator.add_still(self.mag)
                    elif self.mode == "swing":
                        self.calibrator.add_swing_sample(now, self.mag)
                    elif self.mode == "play":
                        self.detector.feed(now, self.mag)
            time.sleep(0.005)

    # --- calibration, driven step by step from the main loop ---------------

    def begin_still(self):
        with self.lock:
            self.calibrator = Calibrator()
            self.mode = "still"

    def begin_swing(self):
        with self.lock:
            self.calibrator.finish_still()
            self.mode = "swing"

    def swings_recorded(self):
        with self.lock:
            return len(self.calibrator.peaks)

    def finish_calibration(self):
        with self.lock:
            self.detector.threshold = self.calibrator.threshold()
            self.mode = "play"
            print(f"Swing calibration: noise={self.calibrator.noise:.0f} "
                  f"peaks={[round(p) for p in self.calibrator.peaks]} "
                  f"threshold={self.detector.threshold:.0f}")
        self.roll0 = self._raw_roll()

    # --- used by the game --------------------------------------------------

    def swung_between(self, t0, t1):
        with self.lock:
            return self.detector.swung_between(t0, t1)

    def _raw_roll(self):
        r = self.dm.imu_device.roll
        return 0.0 if isinstance(r, float) and math.isnan(r) else float(r)

    def tilt(self):
        """Paddle roll relative to how you held it at calibration (degrees)."""
        return self._raw_roll() - self.roll0

    def meter(self):
        """(smoothed gyro magnitude, threshold) for the on-screen swing bar."""
        with self.lock:
            return self.mag, self.detector.threshold

    def close(self):
        self._running = False
        if self._thread:
            self._thread.join(timeout=1)
        try:
            self.dm.motor_stop(motor=le.MOTOR_BOTH)
        finally:
            self.dm.disconnect()


class SimPaddle:
    """Keyboard stand-in for testing without hardware: SPACE = swing."""

    def __init__(self):
        self.detector = SwingDetector()
        self.detector.threshold = 1.0

    def connect(self):
        pass

    def key_swing(self):
        self.detector.feed(time.monotonic(), 2.0)
        self.detector.feed(time.monotonic(), 0.0)

    def swung_between(self, t0, t1):
        return self.detector.swung_between(t0, t1)

    def tilt(self):
        return 0.0

    def meter(self):
        return 0.0, 1.0

    def close(self):
        pass
