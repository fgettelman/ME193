"""
Classify whistle tone from an audio spectrogram and label the plot with
the detected command for each whistle.

Mirrors the pitch-detection and frequency-band scheme from the
whistle_bot project (whistlebot/pitch.py, whistlebot/commands.py): a
chunk only counts as a whistle when it is loud and tonal enough, and its
dominant frequency is looked up in the same BANDS used there, so a tone
that would trigger LEFT in whistle_bot gets labeled "left" here too.

Usage:
    python whistle_class.py                  # record 5s from the mic
    python whistle_class.py --duration 8      # record 8s from the mic
    python whistle_class.py --file take1.wav  # classify a saved WAV instead

Install first (macOS):
    brew install portaudio
    pip install pyaudio numpy matplotlib
"""

import argparse
import wave

import matplotlib.pyplot as plt
import numpy as np

SAMPLE_RATE = 44100
CHUNK = 2048       # samples per spectrogram column (~46 ms, ~21 Hz resolution)
HOP = CHUNK // 2   # 50% overlap between columns

MIN_HZ = 300.0
MAX_HZ = 5000.0
MIN_RMS = 2500.0       # loudness (int16 RMS) a chunk needs to count as a whistle
MIN_TONALITY = 15.0    # spectral peak / in-band average a whistle must clear

# (low, high) Hz -> label. Low inclusive, high exclusive. Gaps between
# ranges are deliberate -- a pitch drifting between two bands is left
# unlabeled instead of guessing, same as whistle_bot/commands.py.
BANDS = {
    "stop":     (600.0, 1000.0),
    "left":     (1000.0, 1500.0),
    "right":    (1500.0, 2200.0),
    "speed_up": (2200.0, 3000.0),
    "goal":     (3000.0, 4500.0),
}
LABEL_COLORS = {
    "stop": "tab:red", "left": "tab:orange", "right": "tab:green",
    "speed_up": "tab:cyan", "goal": "tab:purple",
}


def record_audio(duration, sample_rate=SAMPLE_RATE):
    """Record `duration` seconds from the default microphone."""
    import pyaudio
    pa = pyaudio.PyAudio()
    stream = pa.open(format=pyaudio.paInt16, channels=1, rate=sample_rate,
                      input=True, frames_per_buffer=CHUNK)
    print(f"Recording {duration:.1f}s -- whistle now.")
    n_chunks = int(sample_rate / CHUNK * duration)
    frames = [stream.read(CHUNK, exception_on_overflow=False) for _ in range(n_chunks)]
    stream.stop_stream()
    stream.close()
    pa.terminate()
    return np.frombuffer(b"".join(frames), dtype=np.int16)


def load_wav(path):
    """Load a mono (or first-channel-of-stereo) 16-bit WAV file."""
    with wave.open(path, "rb") as wf:
        sample_rate = wf.getframerate()
        raw = wf.readframes(wf.getnframes())
        samples = np.frombuffer(raw, dtype=np.int16)
        if wf.getnchannels() > 1:
            samples = samples.reshape(-1, wf.getnchannels())[:, 0]
    return samples, sample_rate


def analyze(samples, sample_rate=SAMPLE_RATE, chunk=CHUNK, hop=HOP):
    """Slide a window over `samples`, computing one spectrogram column and
    one classification per step.

    Returns:
        times  -- (n_frames,) seconds, one per column
        freqs  -- (n_bins,) Hz, the FFT bin frequencies
        mags   -- (n_bins, n_frames) magnitude spectrum per column
        tones  -- (n_frames,) dominant Hz where a whistle was detected, else nan
        labels -- list of length n_frames: command name, or None
    """
    window = np.hanning(chunk)
    n_frames = 1 + max(0, (len(samples) - chunk) // hop)
    freqs = np.fft.rfftfreq(chunk, 1.0 / sample_rate)
    band = (freqs >= MIN_HZ) & (freqs <= MAX_HZ)

    mags = np.empty((freqs.size, n_frames))
    times = np.empty(n_frames)
    tones = np.full(n_frames, np.nan)
    labels = [None] * n_frames

    for i in range(n_frames):
        start = i * hop
        seg = samples[start:start + chunk].astype(np.float64)
        times[i] = (start + chunk / 2) / sample_rate

        spectrum = np.abs(np.fft.rfft(seg * window))
        mags[:, i] = spectrum

        rms = float(np.sqrt(np.mean(seg ** 2))) if seg.size else 0.0
        if rms < MIN_RMS or not band.any():
            continue
        in_band = spectrum[band]
        peak = int(np.argmax(in_band))
        if in_band[peak] < MIN_TONALITY * np.mean(in_band):
            continue
        if in_band[peak] < 0.5 * spectrum.max():
            continue

        freq = float(freqs[band][peak])
        tones[i] = freq
        for label, (low, high) in BANDS.items():
            if low <= freq < high:
                labels[i] = label
                break

    return times, freqs, mags, tones, labels


def plot_spectrogram(times, freqs, mags, tones, labels, out_path=None):
    fig, ax = plt.subplots(figsize=(11, 6))

    db = 10.0 * np.log10(mags + 1e-6)
    plot_max_hz = min(MAX_HZ + 1000.0, freqs.max())
    mesh = ax.pcolormesh(times, freqs, db, shading="auto", cmap="magma")
    ax.set_ylim(0, plot_max_hz)
    fig.colorbar(mesh, ax=ax, label="Magnitude (dB)")

    # Reference lines for each command's band, labeled on the right margin.
    ax.set_xlim(times[0], times[-1] * 1.12)
    for label, (low, high) in BANDS.items():
        color = LABEL_COLORS[label]
        ax.axhspan(low, high, color=color, alpha=0.08)
        ax.text(times[-1] * 1.02, (low + high) / 2, label,
                color=color, va="center", fontsize=9, fontweight="bold", clip_on=False)

    # Dominant-frequency trace, colored by classification; unclassified
    # (but still tonal) points are plotted in white so gaps are visible.
    for label in [None, *BANDS]:
        mask = np.array([lab == label for lab in labels]) & ~np.isnan(tones)
        if not mask.any():
            continue
        color = "white" if label is None else LABEL_COLORS[label]
        ax.scatter(times[mask], tones[mask], s=14, color=color,
                   edgecolors="black", linewidths=0.4, zorder=3,
                   label=label or "unclassified")

    # Text annotation each time the classified label changes, so a long
    # steady whistle gets one clean label instead of one per frame.
    prev_label = None
    for t, freq, label in zip(times, tones, labels):
        if label is not None and label != prev_label:
            ax.annotate(label, xy=(t, freq), xytext=(t, freq + 250),
                       color=LABEL_COLORS[label], fontsize=10, fontweight="bold",
                       ha="center",
                       arrowprops=dict(arrowstyle="->", color=LABEL_COLORS[label]))
        prev_label = label

    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Frequency (Hz)")
    ax.set_title("Whistle tone spectrogram -- classified commands")
    ax.legend(loc="upper right", framealpha=0.9)
    fig.tight_layout()

    if out_path:
        fig.savefig(out_path, dpi=150)
        print(f"Saved spectrogram to {out_path}")
    plt.show()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--file", help="WAV file to classify instead of recording live")
    parser.add_argument("--duration", type=float, default=5.0, help="seconds to record (default 5)")
    parser.add_argument("--out", default="whistle_spectrogram.png", help="PNG path to save the plot to")
    args = parser.parse_args()

    if args.file:
        samples, sample_rate = load_wav(args.file)
    else:
        samples, sample_rate = record_audio(args.duration), SAMPLE_RATE

    times, freqs, mags, tones, labels = analyze(samples, sample_rate)

    detected = [lab for lab in labels if lab is not None]
    if detected:
        print("Detected (in order):", " -> ".join(dict.fromkeys(detected)))
    else:
        print("No whistle commands detected.")

    plot_spectrogram(times, freqs, mags, tones, labels, out_path=args.out)


if __name__ == "__main__":
    main()
