"""
The lock's logic, with no hardware in it: the laptop demo (laptop_lock.py)
and the board (main.py) both run this same file, so what you test on the
laptop is what runs on the safe.

Each frame, call  lock.step(now, button_presses, probs)  and it returns a
Display: what the LEDs/matrix should show and where the servo should be.

    LOCKED ──button──> SCANNING ──3 numbers──> correct? ──yes──> UNLOCKED
       ^                  │ timeout / button          │             │ button or
       │                  v                           no            │ UNLOCK_S
       └──────────────────┴──────── DENIED <──────────┘             │
       └──────────────── LOCKOUT (after MAX_TRIES wrong) <──────────┘ (to LOCKED)

Entering a number: the (smoothed) net has to say the same count, 1-5, with at
least CONF_MIN probability, continuously for HOLD_S seconds. The next number
must be different from the one just entered, so holding your hand still
doesn't enter the same number twice (codes like 3-2-3 are fine, 3-3-2 can't be
entered).
"""

import hashlib
import hmac
import json
import os
import secrets
from dataclasses import dataclass

import numpy as np

import settings as S

# Modes, also the numbers sent to the sketch (sketch.ino has the same list).
LOCKED, SCANNING, UNLOCKED, DENIED, LOCKOUT = range(5)
MODE_NAMES = ["LOCKED", "SCANNING", "UNLOCKED", "DENIED", "LOCKOUT"]


# --- The secret code ----------------------------------------------------------
# Stored as a salted SHA-256 hash in code.json, so the code itself is never in
# a file. (With only 5*4*3 = 60 possible codes this stops someone reading the
# file at a glance, not a determined attacker; the lockout is what stops
# guessing at the safe.)

HERE = os.path.dirname(os.path.abspath(__file__))
CODE_PATH = os.path.join(HERE, "code.json")
DEFAULT_CODE = (1, 3, 5)


def hash_code(code, salt):
    return hashlib.sha256((salt + "-".join(map(str, code))).encode()).hexdigest()


def make_code_file(code, path=CODE_PATH):
    code = tuple(int(c) for c in code)
    if len(code) != S.CODE_LENGTH or not all(1 <= c <= 5 for c in code):
        raise ValueError(f"the code must be {S.CODE_LENGTH} numbers from 1 to 5")
    if any(a == b for a, b in zip(code, code[1:])):
        # Holding your hand still must not enter a number twice, so the lock
        # needs each number to differ from the one before it.
        raise ValueError("two numbers in a row can't be the same (3-2-3 is fine, 3-3-2 isn't)")
    salt = secrets.token_hex(8)
    with open(path, "w") as f:
        json.dump({"salt": salt, "sha256": hash_code(code, salt)}, f)


def load_checker(path=CODE_PATH):
    """Returns (check(code) -> bool, is_default)."""
    try:
        with open(path) as f:
            stored = json.load(f)
        salt, want = stored["salt"], stored["sha256"]
        default = False
    except (OSError, ValueError, KeyError):
        salt = "default"
        want = hash_code(DEFAULT_CODE, salt)
        default = True
    return (lambda code: hmac.compare_digest(hash_code(code, salt), want)), default


# --- The state machine --------------------------------------------------------

@dataclass
class Display:
    mode: int
    live: int = 0           # count the net sees right now (0 = none / unsure)
    entered: int = 0        # how many numbers have been entered
    hold: float = 0.0       # 0..1, how far through holding the current number
    seconds: int = 0        # countdown: lockout, time left to unlock/scan
    servo_deg: int = S.SERVO_LOCKED_DEG
    digits: tuple = ()      # the numbers entered so far (for the laptop screen)
    fails: int = 0          # wrong codes in a row

    def code_for_sketch(self):
        """(mode, arg) as sent over the Bridge: arg packs the details."""
        if self.mode == SCANNING:
            return self.mode, self.live + 10 * self.entered + 100 * int(self.hold * 9)
        if self.mode == LOCKOUT:
            return self.mode, min(self.seconds, 999)
        return self.mode, 0


class Lock:
    def __init__(self, check_code):
        self.check_code = check_code
        self.mode = LOCKED
        self.until = 0.0
        self.fails = 0
        self.lockouts = 0
        self.smoothed = None
        self.events = []              # human-readable log, the caller prints and clears it
        self._reset_entry(0.0)

    def _reset_entry(self, now):
        self.entered = []
        self.candidate = 0
        self.candidate_since = now
        self.last_progress = now

    def _log(self, text):
        self.events.append(text)

    def smooth(self, probs):
        """Exponential moving average of the probabilities: one odd frame
        can't enter a number, a steady hand can."""
        p = np.asarray(probs, dtype=float)
        if self.smoothed is None or S.SMOOTH <= 0:
            self.smoothed = p
        else:
            self.smoothed = S.SMOOTH * self.smoothed + (1 - S.SMOOTH) * p
        return self.smoothed

    def seen(self):
        """The count the net is sure of (1-5), or 0."""
        if self.smoothed is None:
            return 0
        k = int(np.argmax(self.smoothed))
        return k if self.smoothed[k] >= S.CONF_MIN else 0

    def step(self, now, presses, probs=None):
        if probs is not None:
            self.smooth(probs)
        else:
            self.smoothed = None
        seen = self.seen()

        if self.mode == LOCKED:
            if presses:
                self.mode = SCANNING
                self._reset_entry(now)
                self._log("button: scanning, show your first number")

        elif self.mode == SCANNING:
            if presses:
                self.mode = LOCKED
                self._log("button: cancelled")
            elif now - self.last_progress > S.SCAN_TIMEOUT_S:
                self.mode = LOCKED
                self._log("timed out, still locked")
            else:
                self._scan(now, seen)

        elif self.mode == UNLOCKED:
            if presses or now >= self.until:
                self.mode = LOCKED
                self._log("locked again" + (" (button)" if presses else " (timer)"))

        elif self.mode == DENIED:
            if now >= self.until:
                if self.fails >= S.MAX_TRIES:
                    self.mode = LOCKOUT
                    wait = S.LOCKOUT_S * 2 ** self.lockouts
                    self.lockouts += 1
                    self.fails = 0
                    self.until = now + wait
                    self._log(f"{S.MAX_TRIES} wrong codes: locked out for {wait:.0f} s")
                else:
                    self.mode = LOCKED

        elif self.mode == LOCKOUT:
            if now >= self.until:     # button presses are ignored until then
                self.mode = LOCKED
                self._log("lockout over")

        return self.display(now, seen)

    def _scan(self, now, seen):
        last = self.entered[-1] if self.entered else 0
        if seen == 0 or seen == last:
            self.candidate = 0
            return
        if seen != self.candidate:
            self.candidate, self.candidate_since = seen, now
            return
        if now - self.candidate_since < S.HOLD_S:
            return
        self.entered.append(seen)
        self.candidate = 0
        self.last_progress = now
        self._log(f"number {len(self.entered)} entered")
        if len(self.entered) == S.CODE_LENGTH:
            if self.check_code(tuple(self.entered)):
                self.mode = UNLOCKED
                self.until = now + S.UNLOCK_S
                self.fails = 0
                self.lockouts = 0
                self._log("correct code: UNLOCKED")
            else:
                self.mode = DENIED
                self.until = now + S.DENIED_S
                self.fails += 1
                self._log(f"wrong code ({self.fails}/{S.MAX_TRIES})")

    def display(self, now, seen=0):
        # The numbers are only shown while they're being entered: once the
        # lock has decided (open, wrong, locked) they're never displayed or
        # sent anywhere, so the code can't be read off a screen afterwards.
        digits = tuple(self.entered) if self.mode == SCANNING else ()
        d = Display(self.mode, fails=self.fails, digits=digits)
        if self.mode == SCANNING:
            d.live = seen
            d.entered = len(self.entered)
            d.seconds = max(0, int(np.ceil(S.SCAN_TIMEOUT_S - (now - self.last_progress))))
            if self.candidate:
                d.hold = min(1.0, (now - self.candidate_since) / S.HOLD_S)
        elif self.mode == UNLOCKED:
            d.servo_deg = S.SERVO_OPEN_DEG
            d.seconds = max(0, int(np.ceil(self.until - now)))
        elif self.mode == LOCKOUT:
            d.seconds = max(0, int(np.ceil(self.until - now)))
        return d
