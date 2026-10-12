"""
The paddle: a Double Motor held in your hand. A background thread reads its
IMU and turns gyro spikes into swing events.

Two ways to decide what counts as a swing:
- ML (default when swing_model.json exists): a logistic-regression model
  trained on your own recorded swings / non-swings (train_swing.py) looks at
  the 0.4 s of gyro + accelerometer around every spin spike.
- Threshold (fallback): calibrated per player -- hold still to measure noise,
  then swing a few times to measure your typical peak; the threshold sits
  between the two.
"""

import json
import math
import os
import statistics
import threading
import time
from collections import deque

import legoeducation as le
import numpy as np

import config
from lelib import doubleMotor

HERE = os.path.dirname(os.path.abspath(__file__))


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

    def last_swing(self):
        return self.swing_times[-1] if self.swing_times else -math.inf


class MLSwingDetector:
    """Feed it (time, [gyro xyz, accel xyz]) at the model's sample rate. Each
    spin spike is judged once, when it sits in the middle of the model's
    window -- the same 0.4 s view of a swing the model was trained on."""

    def __init__(self, model, rate_hz, window_s, info=""):
        self.model = model
        self.rate_hz = rate_hz
        self.n = int(round(window_s * rate_hz))
        self.info = info
        self.buf = deque(maxlen=self.n)
        self.times = deque(maxlen=self.n)
        self.swing_times = deque(maxlen=50)
        self._last_swing = -math.inf
        self.p_last, self.p_time = 0.0, -math.inf

    @classmethod
    def from_file(cls, path):
        with open(path) as f:
            m = json.load(f)
        model = (np.array(m["mean"]), np.array(m["std"]), np.array(m["weights"]), m["bias"])
        info = f"{m.get('n_clips', '?')} clips, {m.get('loo_accuracy', 0) * 100:.0f}% test accuracy"
        return cls(model, m["rate_hz"], m["window_s"], info)

    def feed(self, t, sample):
        from train_swing import features, predict_proba
        self.buf.append(np.asarray(sample, float))
        self.times.append(t)
        if len(self.buf) < self.n:
            return False
        win = np.array(self.buf)
        spin = np.linalg.norm(win[:, :3], axis=1)
        c = self.n // 2
        if int(np.argmax(spin)) != c or spin[c] < config.ML_MIN_SPIN:
            return False
        self.p_last = float(predict_proba(self.model, features(win)))
        self.p_time = t
        t_peak = self.times[c]
        if self.p_last >= config.ML_SWING_P and t_peak - self._last_swing >= config.SWING_REFRACTORY_S:
            self.swing_times.append(t_peak)    # the swing happened at the spike, not now
            self._last_swing = t_peak
            return True
        return False

    def swung_between(self, t0, t1):
        return any(t0 <= t <= t1 for t in self.swing_times)

    def last_swing(self):
        return self.swing_times[-1] if self.swing_times else -math.inf


def load_ml_detector():
    path = os.path.join(HERE, config.ML_SWING_MODEL)
    if not (config.USE_ML_SWING and os.path.exists(path)):
        return None
    try:
        return MLSwingDetector.from_file(path)
    except Exception as e:
        print(f"Couldn't load {config.ML_SWING_MODEL} ({e}) -- using the threshold instead.")
        return None


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
        self.ml = load_ml_detector()
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
        if self.ml:
            print(f"Swing detection: ML model ({self.ml.info}) -- no calibration needed.")
            self.mode = "play"
            self.roll0 = self._raw_roll()
        self._running = True
        self._thread = threading.Thread(target=self._imu_loop, daemon=True)
        self._thread.start()

    def _read_imu(self):
        """[gyro x, y, z, accel x, y, z], or None before the first notification."""
        imu = self.dm.imu_device
        v = (imu.gyroscopeX, imu.gyroscopeY, imu.gyroscopeZ,
             imu.accelerometerX, imu.accelerometerY, imu.accelerometerZ)
        if any(isinstance(x, float) and math.isnan(x) for x in v):
            return None
        return [float(x) for x in v]

    def _imu_loop(self):
        alpha = config.GYRO_EMA_ALPHA
        last_ml = -math.inf
        while self._running:
            sample = self._read_imu()
            if sample is not None:
                raw = math.sqrt(sum(x * x for x in sample[:3]))
                now = time.monotonic()
                with self.lock:
                    if self.ml and self.mode == "play" and now - last_ml >= 0.95 / self.ml.rate_hz:
                        self.ml.feed(now, sample)      # at the rate the model was trained on
                        last_ml = now
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

    @property
    def needs_calibration(self):
        return self.ml is None

    @property
    def ml_info(self):
        return self.ml.info if self.ml else None

    def _swings(self):
        return self.ml or self.detector

    def swung_between(self, t0, t1):
        with self.lock:
            return self._swings().swung_between(t0, t1)

    def last_swing(self):
        with self.lock:
            return self._swings().last_swing()

    def _raw_roll(self):
        r = self.dm.imu_device.roll
        return 0.0 if isinstance(r, float) and math.isnan(r) else float(r)

    def tilt(self):
        """Paddle roll relative to how you held it at calibration (degrees)."""
        return self._raw_roll() - self.roll0

    def meter(self):
        """(value, threshold) for the on-screen swing bar. With the ML model the
        bar shows the last P(swing) (fading out), with the line at ML_SWING_P."""
        with self.lock:
            if self.ml:
                fade = max(0.0, 1.0 - (time.monotonic() - self.ml.p_time) / 0.8)
                return self.ml.p_last * fade * 2 * config.ML_SWING_P, config.ML_SWING_P
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

    needs_calibration = False
    ml_info = None

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

    def last_swing(self):
        return self.detector.last_swing()

    def tilt(self):
        return 0.0

    def meter(self):
        return 0.0, 1.0

    def close(self):
        pass
