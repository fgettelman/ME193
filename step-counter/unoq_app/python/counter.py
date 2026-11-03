"""
Running totals: feed it the newest 2 s window every 0.5 s and it keeps the
step count, distance and time spent in each activity. Used live on the UNO Q
(main.py) and on the laptop (evaluate.py), so both count the same way.

Windows overlap: each 0.5 s of new data is seen by 4 windows. So each time,
the counter adds a quarter of the window's steps (0.5 s / 2 s), which is the
step rate times 0.5 s. Nothing is added while the net says "still", so
fidgeting doesn't slowly creep the count up.
"""

from collections import Counter as Tally, deque

from settings import LABELS, STEP_S, VOTE_WINDOWS, WINDOW_S


class StepCounter:
    def __init__(self, net):
        self.net = net
        self.reset()

    def reset(self):
        self.steps = 0.0
        self.distance_m = 0.0
        self.seconds = {lab: 0.0 for lab in LABELS}
        self.votes = deque(maxlen=VOTE_WINDOWS)

    def update(self, window):
        """Count one new 0.5 s. Returns what to show."""
        window_steps, probs = self.net.run(window)
        raw = max(probs, key=probs.get)
        self.votes.append(raw)
        activity = Tally(self.votes).most_common(1)[0][0]

        added = 0.0 if raw == "still" else window_steps * STEP_S / WINDOW_S
        self.steps += added
        self.distance_m += added * self.net.step_length.get(raw, 0.0)
        self.seconds[activity] += STEP_S
        cadence = 0.0 if raw == "still" else window_steps / WINDOW_S * 60
        return {
            "activity": activity,
            "raw": raw,
            "probs": {k: round(v, 3) for k, v in probs.items()},
            "steps": round(self.steps, 1),
            "distance_m": round(self.distance_m, 2),
            "cadence_spm": round(cadence, 1),
            "window_steps": round(window_steps, 2),
            "seconds": dict(self.seconds),
        }
