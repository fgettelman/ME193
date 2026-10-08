"""
Virtual ping-pong with the LEGO Double Motor as the paddle.

  Pose (MediaPipe)  -> is your paddle hand where the ball is?
  IMU (Double Motor)-> are you actually swinging?
  AprilTags         -> tag 1/2/3 picks the level (ball speed), tag 0 starts
  MQTT              -> current run of continuous hits, as a float, live to
                       ME193/Rogers/Fiona on test.mosquitto.org
  Haptics (new!)    -> rumble that builds as the ball comes in, a PD "thunk"
                       on every hit, buzz pattern on a miss
  Whistle           -> pause / resume

Run (from the repo root, with the ping-pong environment):
    source my_env_pingpong_py312/bin/activate
    cd ping_pong
    python ping_pong.py              # full game
    python ping_pong.py --sim        # no hardware: mouse = paddle, SPACE = swing
    python ping_pong.py --no-whistle # skip the microphone

Keys: q quit | p pause | c recalibrate swing | 1/2/3 level | s start
      (keys are a backup for the AprilTags)
"""

import argparse
import time

import cv2
import numpy as np

import config
import game as g
from scoreboard import Scoreboard

WINDOW = "Virtual Ping-Pong"
DEPTH_K = 2.0          # far wall is drawn at 1 / (1 + DEPTH_K) the size
BALL_RADIUS = 40
HAPTIC_EVENTS = ("serve", "hit", "milestone", "miss")

WHITE, GREEN, RED, YELLOW, CYAN, GRAY = ((255, 255, 255), (0, 220, 0), (0, 0, 255),
                                         (0, 220, 255), (255, 220, 0), (140, 140, 140))


# --- drawing ----------------------------------------------------------------

def project(x, y, z, w, h):
    s = 1.0 / (1.0 + DEPTH_K * z)
    return int(w / 2 + (x - 0.5) * w * s), int(h / 2 + (y - 0.5) * h * s), s


def text(img, msg, org, scale=0.8, color=WHITE, thick=2):
    cv2.putText(img, msg, org, cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), thick + 3, cv2.LINE_AA)
    cv2.putText(img, msg, org, cv2.FONT_HERSHEY_SIMPLEX, scale, color, thick, cv2.LINE_AA)


def centered(img, msg, y, scale=1.2, color=WHITE, thick=3):
    (tw, _), _ = cv2.getTextSize(msg, cv2.FONT_HERSHEY_SIMPLEX, scale, thick)
    text(img, msg, ((img.shape[1] - tw) // 2, y), scale, color, thick)


def draw_court(img):
    h, w = img.shape[:2]
    near = [(0.05, 0.15), (0.95, 0.15), (0.95, 0.9), (0.05, 0.9)]
    far = [project(x, y, 1.0, w, h)[:2] for x, y in near]
    near_px = [project(x, y, 0.0, w, h)[:2] for x, y in near]
    cv2.polylines(img, [np.array(far)], True, CYAN, 2)
    for a, b in zip(near_px, far):
        cv2.line(img, a, b, CYAN, 1)


def draw_ball(img, game):
    h, w = img.shape[:2]
    b = game.ball
    if b.incoming:   # where it will land -- the hit zone
        tx, ty = int(b.target[0] * w), int(b.target[1] * h)
        cv2.ellipse(img, (tx, ty), (int(config.HIT_RADIUS * w), int(config.HIT_RADIUS * h)),
                    0, 0, 360, YELLOW, 2)
    x, y, s = project(b.x, b.y, b.z, w, h)
    cv2.circle(img, (x, y), max(4, int(BALL_RADIUS * s)), (0, 140, 255), -1)
    cv2.circle(img, (x, y), max(4, int(BALL_RADIUS * s)), WHITE, 2)


def draw_wrist(img, wrist, game):
    if wrist is None:
        return
    h, w = img.shape[:2]
    b = game.ball
    in_zone = b.incoming and np.hypot(wrist[0] - b.target[0], wrist[1] - b.target[1]) <= config.HIT_RADIUS
    cv2.circle(img, (int(wrist[0] * w), int(wrist[1] * h)), 18, GREEN if in_zone else WHITE, 4)


def draw_tags(img, tags, stab):
    for tag_id, corners in tags:
        pts = corners.astype(int)
        cv2.polylines(img, [pts], True, RED, 3)
        name = "START" if tag_id == config.TAG_START else config.LEVELS.get(tag_id, ("?",))[0]
        text(img, f"{name} {int(stab.progress(tag_id) * 100)}%", tuple(pts[0]), 0.7, RED)


def draw_hud(img, game, paddle, board):
    h, w = img.shape[:2]
    text(img, f"Streak: {game.streak}", (20, 40), 1.1, GREEN, 3)
    text(img, f"Best: {game.best}", (20, 80))
    text(img, f"Level: {game.level_name}", (20, 115))
    text(img, "Misses: " + "X" * game.misses + "-" * (config.MAX_MISSES - game.misses), (20, 150))
    mqtt = "MQTT: live" if board.connected else "MQTT: connecting..."
    text(img, mqtt, (w - 260, 40), 0.7, GREEN if board.connected else YELLOW)

    # Swing meter: smoothed gyro magnitude vs. threshold.
    mag, thresh = paddle.meter()
    if np.isfinite(thresh) and thresh > 0:
        x0, y0, bw, bh = w - 60, 80, 30, 200
        frac = min(1.0, mag / (2 * thresh))
        cv2.rectangle(img, (x0, y0), (x0 + bw, y0 + bh), GRAY, 2)
        cv2.rectangle(img, (x0, y0 + int(bh * (1 - frac))), (x0 + bw, y0 + bh),
                      GREEN if mag >= thresh else YELLOW, -1)
        cv2.line(img, (x0 - 5, y0 + bh // 2), (x0 + bw + 5, y0 + bh // 2), RED, 2)
        text(img, "swing", (x0 - 20, y0 + bh + 25), 0.6)

    if game.state == g.LOBBY:
        centered(img, "Show tag 1 / 2 / 3 to pick level", h // 2 - 30, 1.0)
        centered(img, "Show tag 0 to START", h // 2 + 20, 1.0, YELLOW)
    elif game.state == g.COUNTDOWN:
        left = max(0.0, game.state_until - time.monotonic())
        centered(img, str(int(left) + 1), h // 2, 4.0, YELLOW, 8)
    elif game.state == g.PAUSED:
        centered(img, "PAUSED", h // 2, 2.0, YELLOW, 5)
        centered(img, "whistle or press p to resume", h // 2 + 50, 0.9)
    elif game.state == g.GAME_OVER:
        centered(img, "GAME OVER", h // 2, 2.0, RED, 5)
        centered(img, f"Best streak: {game.best}", h // 2 + 55, 1.0)


# --- main -------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sim", action="store_true", help="no hardware: mouse = paddle, SPACE = swing")
    ap.add_argument("--no-whistle", action="store_true", help="don't open the microphone")
    args = ap.parse_args()

    board = Scoreboard()
    game = g.Game()

    # Paddle + haptics
    if args.sim:
        from haptics import NullHaptics
        from paddle import SimPaddle
        paddle, haptics = SimPaddle(), NullHaptics()
    else:
        from haptics import Haptics
        from paddle import Paddle
        paddle = Paddle()
        paddle.connect()
        haptics = Haptics(paddle.dm)
        haptics.level_light(config.LEVELS[game.level][2])

    # Camera + vision (sim mode draws on a blank canvas instead)
    cap = vision = None
    if not args.sim:
        from camlib import pick_camera
        from vision import TagStabilizer, Vision
        cap, start_ms = pick_camera(config.CAMERA_WIDTH, config.CAMERA_HEIGHT)
        vision = Vision(start_ms)
        stab = TagStabilizer()

    whistle = None
    if config.USE_WHISTLE and not args.no_whistle:
        try:
            from whistle import WhistleListener
            whistle = WhistleListener()
        except Exception as e:
            print(f"Whistle disabled: {e}")

    mouse = [None]
    cv2.namedWindow(WINDOW)
    cv2.setMouseCallback(WINDOW, lambda ev, x, y, *_: mouse.__setitem__(0, (x, y)))

    # Calibration: "STILL" -> "SWING" -> None (done). Sim mode skips it.
    cal = None if args.sim else "STILL"
    cal_until = 0.0
    if cal:
        paddle.begin_still()
        cal_until = time.monotonic() + config.CAL_STILL_S

    prev = time.monotonic()
    try:
        while True:
            now = time.monotonic()
            dt, prev = now - prev, now

            if cap is not None:
                ok, frame = cap.read()
                if not ok:
                    print("Camera read failed.")
                    break
                view = cv2.flip(frame, 1)
            else:
                frame = None
                view = np.full((config.CAMERA_HEIGHT, config.CAMERA_WIDTH, 3), 30, np.uint8)
            h, w = view.shape[:2]

            # --- sensors ---
            if vision:
                wrist = vision.wrist(frame)
            elif mouse[0]:
                wrist = (mouse[0][0] / w, mouse[0][1] / h)
            else:
                wrist = None

            tags = []
            if vision and game.state in (g.LOBBY, g.GAME_OVER) and cal is None:
                tags = vision.detect_tags(frame)
                for tag_id in stab.update([t for t, _ in tags]):
                    if tag_id in config.LEVELS:
                        game.set_level(tag_id)
                        haptics.level_light(config.LEVELS[tag_id][2])
                        haptics.confirm()
                    elif tag_id == config.TAG_START and game.state == g.LOBBY:
                        game.start(now)

            if whistle and whistle.pop_toggle():
                game.toggle_pause()

            # --- calibration ---
            if cal == "STILL" and now >= cal_until:
                paddle.begin_swing()
                cal = "SWING"
            elif cal == "SWING" and paddle.swings_recorded() >= config.CAL_SWINGS:
                paddle.finish_calibration()
                haptics.confirm()
                cal = None

            # --- game ---
            if cal is None:
                events = game.update(now, dt, wrist, paddle.swung_between, paddle.tilt())
                for ev in events:
                    if ev in HAPTIC_EVENTS:
                        getattr(haptics, ev)()
                haptics.proximity(game.proximity())
                board.set_score(game.streak, now)
            board.tick(now)

            # --- draw ---
            draw_court(view)
            if game.state in (g.RALLY, g.PAUSED):
                draw_ball(view, game)
            draw_wrist(view, wrist, game)
            if vision:
                draw_tags(view, tags, stab)
            draw_hud(view, game, paddle, board)
            if cal == "STILL":
                centered(view, "Hold the paddle still...", h // 2, 1.3, YELLOW)
            elif cal == "SWING":
                centered(view, f"Now swing {config.CAL_SWINGS} times  "
                               f"({paddle.swings_recorded()}/{config.CAL_SWINGS})", h // 2, 1.3, YELLOW)
            if args.sim:
                text(view, "SIM: mouse = paddle, SPACE = swing, 1/2/3 level, s start", (20, h - 20), 0.7, GRAY)
            cv2.imshow(WINDOW, view)

            # --- keys ---
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            elif key == ord("p"):
                game.toggle_pause()
            elif key == ord(" ") and args.sim:
                paddle.key_swing()
            elif key in (ord("1"), ord("2"), ord("3")) and game.state in (g.LOBBY, g.GAME_OVER):
                game.set_level(int(chr(key)))
                haptics.level_light(config.LEVELS[game.level][2])
            elif key == ord("s") and game.state == g.LOBBY and cal is None:
                game.start(now)
            elif key == ord("c") and not args.sim:
                paddle.begin_still()
                cal, cal_until = "STILL", now + config.CAL_STILL_S
    finally:
        print(f"Final best streak this session: {game.best}")
        board.set_score(game.streak, time.monotonic())
        board.close()
        if whistle:
            whistle.close()
        if not args.sim:
            haptics.close()
            paddle.close()
        if cap is not None:
            cap.release()
        if vision:
            vision.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
