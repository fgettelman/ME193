"""
Crowd sounds on the computer speakers: applause on a hit, booing on a miss.

The clips are synthesized with numpy the first time (no downloads) and cached
as WAV files in ./sounds/. Played with macOS `afplay` in the background, so the
game loop never waits on audio.
"""

import os
import subprocess
import wave

import numpy as np

SR = 44100
HERE = os.path.dirname(os.path.abspath(__file__))
SOUND_DIR = os.path.join(HERE, "sounds")


def _bandpass(x, lo, hi):
    spec = np.fft.rfft(x)
    f = np.fft.rfftfreq(len(x), 1 / SR)
    spec[(f < lo) | (f > hi)] = 0
    return np.fft.irfft(spec, len(x))


def _applause(seconds=2.2, people=30, rng=np.random.default_rng(1)):
    """Lots of people clapping at their own pace: short filtered noise bursts."""
    n = int(seconds * SR)
    out = np.zeros(n + SR)
    t = np.arange(int(0.04 * SR)) / SR
    claps = []
    for _ in range(10):   # a few clap "shapes" so it isn't one repeated click
        lo = rng.uniform(500, 1200)
        burst = rng.standard_normal(len(t)) * np.exp(-t / rng.uniform(0.004, 0.009))
        claps.append(_bandpass(burst, lo, lo * rng.uniform(3, 6)))
    for _ in range(people):
        rate = rng.uniform(3.5, 6.5)                     # claps per second
        start = rng.uniform(0.0, 0.25)
        loud = rng.uniform(0.3, 1.0)
        for c in np.arange(start, seconds, 1 / rate):
            i = int((c + rng.normal(0, 0.01)) * SR)
            k = claps[rng.integers(len(claps))]
            if 0 <= i < n:
                out[i:i + len(k)] += loud * rng.uniform(0.6, 1.0) * k
    out = out[:n]
    env = np.ones(n)
    fade_in, fade_out = int(0.08 * SR), int(0.8 * SR)
    env[:fade_in] = np.linspace(0, 1, fade_in)
    env[-fade_out:] = np.linspace(1, 0, fade_out) ** 1.5
    return out * env


def _boo(seconds=1.8, voices=14, rng=np.random.default_rng(2)):
    """A crowd going "booooo": buzzy low voices through an "oo" vowel filter,
    each sliding down in pitch."""
    n = int(seconds * SR)
    tt = np.arange(n) / SR
    out = np.zeros(n)
    f = np.fft.rfftfreq(n, 1 / SR)
    # "oo" vowel: formants near 300 Hz and 870 Hz.
    vowel = 1 / (1 + ((f - 300) / 90) ** 2) + 0.5 / (1 + ((f - 870) / 130) ** 2)
    for _ in range(voices):
        f0 = rng.uniform(95, 210)
        drop = rng.uniform(0.75, 0.9)                    # pitch falls by the end
        pitch = f0 * (1 + (drop - 1) * tt / seconds)
        pitch *= 1 + 0.015 * np.sin(2 * np.pi * rng.uniform(4, 6) * tt)   # wobble
        phase = np.cumsum(pitch) / SR
        saw = 2 * (phase % 1.0) - 1
        voice = np.fft.irfft(np.fft.rfft(saw) * vowel, n)
        start = rng.uniform(0.0, 0.25)
        env = np.clip((tt - start) / 0.15, 0, 1) * np.clip((seconds - tt) / 0.5, 0, 1)
        out += rng.uniform(0.5, 1.0) * voice * env
    return out


def _write(path, x):
    x = x / (np.max(np.abs(x)) + 1e-9) * 0.85
    with wave.open(path, "wb") as f:
        f.setnchannels(1)
        f.setsampwidth(2)
        f.setframerate(SR)
        f.writeframes((x * 32767).astype(np.int16).tobytes())


CLIPS = {"applause": _applause, "boo": _boo}


class Sounds:
    def __init__(self):
        os.makedirs(SOUND_DIR, exist_ok=True)
        self.paths = {}
        for name, make in CLIPS.items():
            path = os.path.join(SOUND_DIR, f"{name}.wav")
            if not os.path.exists(path):
                _write(path, make())
            self.paths[name] = path
        self._proc = None

    def play(self, name):
        """Start a clip, cutting off whatever was still playing."""
        if self._proc and self._proc.poll() is None:
            self._proc.terminate()
        try:
            self._proc = subprocess.Popen(["afplay", self.paths[name]],
                                          stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        except OSError as e:
            print(f"sound: {e}")

    def close(self):
        if self._proc and self._proc.poll() is None:
            self._proc.terminate()
