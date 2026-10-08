"""
Whistle to pause / resume. Same loudness + tonality + frequency-band test as
whistle_class.py, but live from the mic (picked with miclib.pick_mic()).
"""

import threading

import numpy as np
import sounddevice as sd

import config
from miclib import pick_mic


class WhistleListener:
    def __init__(self):
        self._toggles = 0
        self._held = 0
        self._latched = False
        self._lock = threading.Lock()
        rate, chunk = config.WHISTLE_SAMPLE_RATE, config.WHISTLE_CHUNK
        self._window = np.hanning(chunk)
        freqs = np.fft.rfftfreq(chunk, 1.0 / rate)
        self._freqs = freqs
        self._band = (freqs >= 300.0) & (freqs <= 5000.0)
        self.stream = sd.InputStream(samplerate=rate, channels=1, blocksize=chunk,
                                     dtype="float32", device=pick_mic(), callback=self._callback)
        self.stream.start()

    def _is_whistle(self, seg):
        if np.sqrt(np.mean(seg ** 2)) < config.WHISTLE_MIN_RMS:
            return False
        spectrum = np.abs(np.fft.rfft(seg * self._window))
        in_band = spectrum[self._band]
        peak = int(np.argmax(in_band))
        if in_band[peak] < config.WHISTLE_MIN_TONALITY * np.mean(in_band):
            return False
        low, high = config.WHISTLE_BAND
        return low <= self._freqs[self._band][peak] < high

    def _callback(self, indata, frames, time_info, status):
        if self._is_whistle(indata[:, 0]):
            self._held += 1
            if self._held >= config.WHISTLE_HOLD_CHUNKS and not self._latched:
                self._latched = True     # one toggle per whistle
                with self._lock:
                    self._toggles += 1
        else:
            self._held = 0
            self._latched = False

    def pop_toggle(self):
        """True once per whistle heard since the last call."""
        with self._lock:
            if self._toggles:
                self._toggles -= 1
                return True
        return False

    def close(self):
        self.stream.stop()
        self.stream.close()
