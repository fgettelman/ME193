"""
Tests for the parts that don't need hardware.  Run:  python -m unittest test_game
"""

import random
import unittest

import config
import game as g
from paddle import Calibrator, SwingDetector
from vision import TagStabilizer

DT = 1 / 30


def run_to_plane(game, now):
    """Advance until the incoming ball reaches the player's plane."""
    while game.arrival_time is None:
        now += DT
        game.update(now, DT, None, lambda a, b: False)
    return now


def started_game():
    game = g.Game(rng=random.Random(0))
    game.start(0.0)
    events = game.update(config.COUNTDOWN_S, DT, None, lambda a, b: False)
    assert events == ["serve"] and game.state == g.RALLY
    return game, config.COUNTDOWN_S


class GameTests(unittest.TestCase):
    def test_hit_needs_position_and_swing(self):
        game, now = started_game()
        now = run_to_plane(game, now)
        wrist = (game.ball.x, game.ball.y)
        events = game.update(now + DT, DT, wrist, lambda a, b: True)
        self.assertIn("hit", events)
        self.assertEqual(game.streak, 1)
        self.assertGreater(game.ball.vz, 0)

    def test_in_position_without_swing_is_a_miss(self):
        game, now = started_game()
        now = run_to_plane(game, now)
        wrist = (game.ball.x, game.ball.y)
        events = game.update(now + 1.0, DT, wrist, lambda a, b: False)
        self.assertEqual(events, ["miss"])
        self.assertEqual(game.misses, 1)

    def test_swing_out_of_position_is_a_miss(self):
        game, now = started_game()
        now = run_to_plane(game, now)
        far = (game.ball.x + 0.5, game.ball.y)
        events = game.update(now + 1.0, DT, far, lambda a, b: True)
        self.assertEqual(events, ["miss"])

    def test_miss_resets_streak_but_keeps_best(self):
        game, now = started_game()
        game.streak = game.best = 4
        now = run_to_plane(game, now)
        game.update(now + 1.0, DT, None, lambda a, b: False)
        self.assertEqual(game.streak, 0)
        self.assertEqual(game.best, 4)
        self.assertEqual(game.state, g.COUNTDOWN)

    def test_game_over_after_max_misses(self):
        game, now = started_game()
        for _ in range(config.MAX_MISSES):
            if game.state == g.COUNTDOWN:
                now = game.state_until
                game.update(now, DT, None, lambda a, b: False)
            now = run_to_plane(game, now)
            now += 1.0
            events = game.update(now, DT, None, lambda a, b: False)
        self.assertIn("game_over", events)
        self.assertEqual(game.state, g.GAME_OVER)
        game.update(now + config.GAME_OVER_S, DT, None, lambda a, b: False)
        self.assertEqual(game.state, g.LOBBY)

    def test_ball_returns_from_far_wall(self):
        game, now = started_game()
        now = run_to_plane(game, now)
        game.update(now + DT, DT, (game.ball.x, game.ball.y), lambda a, b: True)
        events = []
        while "wall" not in events:
            now += DT
            events = game.update(now, DT, None, lambda a, b: False)
        self.assertTrue(game.ball.incoming)

    def test_level_sets_speed(self):
        game = g.Game()
        game.set_level(3)
        self.assertEqual(game.speed, config.LEVELS[3][1])


class SwingTests(unittest.TestCase):
    def test_one_swing_counts_once(self):
        d = SwingDetector()
        d.threshold = 100
        hits = [d.feed(t * 0.01, m) for t, m in enumerate([0, 150, 200, 150, 0, 160])]
        self.assertEqual(sum(hits), 1)   # 2nd spike is inside the refractory window
        self.assertTrue(d.swung_between(0.0, 0.02))

    def test_calibration_threshold_between_noise_and_peak(self):
        c = Calibrator()
        for m in [10, 12, 11, 9, 10]:
            c.add_still(m)
        c.finish_still()
        t = 0.0
        for peak in [500, 600, 550]:
            for m in [20, peak, 20]:
                c.add_swing_sample(t, m)
                t += 0.2
        self.assertEqual(len(c.peaks), 3)
        self.assertTrue(c.noise < c.threshold() < 550)


class TagTests(unittest.TestCase):
    def test_tag_fires_once_after_stable(self):
        s = TagStabilizer(frames=3)
        self.assertEqual(s.update([1]), [])
        self.assertEqual(s.update([1]), [])
        self.assertEqual(s.update([1]), [1])
        self.assertEqual(s.update([1]), [])
        s.update([])
        self.assertEqual(s.update([1]), [])   # count restarts after it leaves


if __name__ == "__main__":
    unittest.main()
