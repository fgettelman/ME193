"""
Ball physics and game rules. No hardware, camera, or network in here, so it
can be tested on its own (see test_game.py).

Coordinates: x, y are normalized screen position (0..1, mirrored selfie view)
at the player's plane; z is depth (1 = far wall, 0 = player).
"""

import math
import random

import config

LOBBY = "LOBBY"
COUNTDOWN = "COUNTDOWN"
RALLY = "RALLY"
PAUSED = "PAUSED"
GAME_OVER = "GAME_OVER"


class Ball:
    def __init__(self):
        self.x = self.y = 0.5
        self.z = 1.0
        self.vx = self.vy = self.vz = 0.0
        self.target = (0.5, 0.5)   # where it will cross the player's plane
        self.launch_y = 0.5        # height it left the last paddle at (for drawing the arc)

    @property
    def incoming(self):
        return self.vz < 0


class Game:
    def __init__(self, rng=None):
        self.rng = rng or random.Random()
        self.ball = Ball()
        self.state = LOBBY
        self.level = config.DEFAULT_LEVEL
        self.speed = self.base_speed
        self.streak = 0
        self.misses = 0
        self.best = 0
        self.state_until = 0.0
        self.arrival_time = None   # set while waiting to judge a ball at the plane
        self.closest_wrist = math.inf
        self.swing_open = None     # time the swing window opened for this ball
        self.miss_reason = ""
        self.miss_time = -math.inf
        self.paused_from = None

    # --- level / state ---------------------------------------------------

    @property
    def base_speed(self):
        return config.LEVELS[self.level][1]

    @property
    def level_name(self):
        return config.LEVELS[self.level][0]

    def set_level(self, level):
        self.level = level
        self.speed = self.base_speed

    def start(self, now):
        self.streak = 0
        self.misses = 0
        self.speed = self.base_speed
        self._countdown(now)

    def toggle_pause(self):
        if self.state == PAUSED:
            self.state = self.paused_from
        elif self.state in (RALLY, COUNTDOWN):
            self.paused_from = self.state
            self.state = PAUSED

    def _countdown(self, now):
        self.state = COUNTDOWN
        self.state_until = now + config.COUNTDOWN_S
        self.arrival_time = None
        self.swing_open = None

    def _serve(self):
        b = self.ball
        b.x, b.y, b.z = 0.5, 0.5, 1.0
        self._aim_incoming()
        self.state = RALLY

    def _aim_incoming(self):
        """Send the ball from the far wall to a random spot on the player's plane."""
        b = self.ball
        tx = self.rng.uniform(0.2, 0.8)
        ty = self.rng.uniform(0.35, 0.75)
        flight = b.z / self.speed
        b.vx = (tx - b.x) / flight
        b.vy = (ty - b.y) / flight
        b.vz = -self.speed
        b.target = (tx, ty)
        b.launch_y = b.y

    def proximity(self):
        """0 (far) .. 1 (at your paddle) while the ball is incoming, else 0."""
        if self.state != RALLY or not self.ball.incoming:
            return 0.0
        return max(0.0, min(1.0, 1.0 - self.ball.z))

    # --- main update -----------------------------------------------------

    def update(self, now, dt, wrist, swung_between, tilt_deg=0.0):
        """Advance the game.

        wrist          -- (x, y) of the paddle blade center, or None if not seen
        swung_between  -- f(t0, t1) -> bool, was there a swing in that window
        tilt_deg       -- paddle roll relative to neutral (aims the return)

        Returns a list of event names for sound/haptics/scoreboard:
        "serve", "hit", "milestone", "miss", "game_over", "wall".
        """
        events = []

        if self.state == COUNTDOWN and now >= self.state_until:
            self._serve()
            events.append("serve")
        elif self.state == GAME_OVER and now >= self.state_until:
            self.state = LOBBY
        if self.state != RALLY:
            return events

        b = self.ball

        # Once the window is open, remember the closest the paddle got to the
        # lilac square -- a swing carries it away, so "now" isn't good enough.
        if self.swing_open is not None and wrist is not None:
            self.closest_wrist = min(self.closest_wrist, box_dist(wrist, b.target))

        # Waiting at the plane for the swing window to play out.
        if self.arrival_time is not None:
            in_place = self.closest_wrist <= config.HIT_BOX
            if in_place and swung_between(self.swing_open, now):
                self._hit(wrist or b.target, tilt_deg)
                events.append("hit")
                if self.streak % config.MILESTONE_EVERY == 0:
                    events.append("milestone")
            elif now >= self.arrival_time + config.SWING_LATE_S:
                if not in_place:
                    self.miss_reason = "Paddle wasn't in the lilac square"
                elif swung_between(self.swing_open - 1.5, self.swing_open):
                    self.miss_reason = "Too early - swing after the bounce"
                else:
                    self.miss_reason = "No swing detected - swing harder"
                events.extend(self._miss(now))
            return events

        b.x += b.vx * dt
        b.y += b.vy * dt
        b.z += b.vz * dt

        # Side walls.
        if b.x < 0.05 or b.x > 0.95:
            b.x = min(0.95, max(0.05, b.x))
            b.vx = -b.vx
        if b.y < 0.15 or b.y > 0.9:
            b.y = min(0.9, max(0.15, b.y))
            b.vy = -b.vy

        if b.incoming and self.swing_open is None and b.z <= config.SWING_OPEN_Z:
            self.swing_open = now
            self.closest_wrist = box_dist(wrist, b.target) if wrist else math.inf
        if b.incoming and b.z <= 0.0:
            b.z = 0.0
            self.arrival_time = now
        elif not b.incoming and b.z >= 1.0:
            b.z = 1.0
            self._aim_incoming()
            events.append("wall")
        return events

    def _hit(self, wrist, tilt_deg):
        b = self.ball
        self.arrival_time = None
        self.swing_open = None
        b.launch_y = b.y
        self.streak += 1
        self.best = max(self.best, self.streak)
        self.speed = min(config.MAX_SPEED, self.speed * config.SPEEDUP_PER_HIT)
        b.vz = self.speed
        b.vx = config.K_OFFSET * (b.x - wrist[0]) + config.K_TILT * tilt_deg
        b.vy = 0.5 * (0.5 - b.y)

    def _miss(self, now):
        self.arrival_time = None
        self.swing_open = None
        self.miss_time = now
        self.streak = 0
        self.misses += 1
        self.speed = self.base_speed
        if self.misses >= config.MAX_MISSES:
            self.state = GAME_OVER
            self.state_until = now + config.GAME_OVER_S
            return ["miss", "game_over"]
        self._countdown(now)
        return ["miss"]


def box_dist(a, b):
    """How far a is from b, as a fraction of screen height, measured square-wise:
    <= config.HIT_BOX means a is inside the lilac square centered on b."""
    return max(abs(a[0] - b[0]) * config.ASPECT, abs(a[1] - b[1]))
