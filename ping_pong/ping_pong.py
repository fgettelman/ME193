"""
Virtual ping-pong with the LEGO Double Motor as the paddle.

  Pose (MediaPipe)  -> is your paddle hand where the ball is?
  IMU (Double Motor)-> are you actually swinging?
  AprilTags         -> tag 1/2/3 picks the level (ball speed) and starts
  MQTT              -> current run of continuous hits, as a float, live to
                       ME193/Rogers/Fiona on test.mosquitto.org
  Haptics (new!)    -> rumble + a PD "thunk" only when you hit the ball
  Whistle           -> pause / resume
  Sounds            -> applause from the computer on a hit, booing on a miss

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
from sounds import Sounds

WINDOW = "Virtual Ping-Pong"
DEPTH_K = 1.4          # far end of the table is drawn at 1 / (1 + DEPTH_K) the size
BALL_RADIUS = 45
HAPTIC_EVENTS = ("serve", "hit", "milestone")   # motors only move on "hit"

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


# Table geometry, in the same (x, y at the player's plane, z depth) coords as
# the game: z = 0 is your end, z = 1 the opponent's. y grows downward.
TABLE_Y = 0.92          # height of the table top
TABLE_X = (0.1, 0.9)    # table width
NET_Z, NET_H = 0.5, 0.07
OPP_Y = 0.5             # height the opponent hits the ball at
ARC = 0.25              # how high the ball loops over the net
BOUNCE_IN, BOUNCE_OUT = 0.25, 0.75   # where it bounces on your / their half
LILAC = (250, 170, 215)
TABLE_BLUE = (110, 55, 15)
WOOD, RUBBER, RUBBER_EDGE = (70, 130, 190), (45, 45, 210), (20, 20, 90)


def draw_table(img):
    h, w = img.shape[:2]
    x0, x1 = TABLE_X
    corners = [project(x, TABLE_Y, z, w, h)[:2] for x, z in ((x0, 0), (x1, 0), (x1, 1), (x0, 1))]
    top = img.copy()
    cv2.fillPoly(top, [np.array(corners)], TABLE_BLUE, cv2.LINE_AA)
    cv2.addWeighted(top, 0.85, img, 0.15, 0, img)
    cv2.polylines(img, [np.array(corners)], True, WHITE, 3, cv2.LINE_AA)
    cv2.line(img, project(0.5, TABLE_Y, 0, w, h)[:2], project(0.5, TABLE_Y, 1, w, h)[:2], WHITE, 1, cv2.LINE_AA)
    # Front edge of the table, so it reads as a slab rather than a sheet.
    a, b = corners[0], corners[1]
    cv2.rectangle(img, (a[0], a[1]), (b[0], min(h - 1, b[1] + 18)), (70, 35, 10), -1)


def draw_net(img):
    h, w = img.shape[:2]
    x0, x1 = TABLE_X
    l_bot, r_bot = project(x0 - 0.02, TABLE_Y, NET_Z, w, h)[:2], project(x1 + 0.02, TABLE_Y, NET_Z, w, h)[:2]
    l_top, r_top = project(x0 - 0.02, TABLE_Y - NET_H, NET_Z, w, h)[:2], project(x1 + 0.02, TABLE_Y - NET_H, NET_Z, w, h)[:2]
    net = img.copy()
    cv2.fillPoly(net, [np.array([l_bot, r_bot, r_top, l_top])], (40, 40, 40))
    cv2.addWeighted(net, 0.55, img, 0.45, 0, img)
    for k in range(1, 30):   # mesh
        t = k / 30
        p = (int(l_bot[0] + (r_bot[0] - l_bot[0]) * t), l_bot[1])
        q = (int(l_top[0] + (r_top[0] - l_top[0]) * t), l_top[1])
        cv2.line(img, p, q, (90, 90, 90), 1)
    cv2.line(img, l_top, r_top, WHITE, 3, cv2.LINE_AA)
    cv2.line(img, l_bot, l_top, (200, 200, 200), 3)
    cv2.line(img, r_bot, r_top, (200, 200, 200), 3)


def ball_height(game):
    """Where the ball is drawn vertically: an arc over the net, a bounce on the
    far half of the table, then up to the paddle (and the mirror image going out)."""
    b = game.ball
    z = b.z
    if game.state == g.COUNTDOWN:
        return OPP_Y
    if b.incoming or game.arrival_time is not None:
        if z >= BOUNCE_IN:
            u = (1 - z) / (1 - BOUNCE_IN)
            return OPP_Y + (TABLE_Y - OPP_Y) * u - ARC * 4 * u * (1 - u)
        u = (BOUNCE_IN - z) / BOUNCE_IN
        return TABLE_Y + (b.target[1] - TABLE_Y) * np.sin(np.pi / 2 * u)
    if z <= BOUNCE_OUT:
        u = z / BOUNCE_OUT
        return b.launch_y + (TABLE_Y - b.launch_y) * u - ARC * 4 * u * (1 - u)
    u = (z - BOUNCE_OUT) / (1 - BOUNCE_OUT)
    return TABLE_Y + (OPP_Y - TABLE_Y) * np.sin(np.pi / 2 * u)


def draw_pingpong(img, x, y, r):
    """Shaded orange ping-pong ball: lit from the top-left, with a shine spot."""
    r = max(5, r)
    cv2.circle(img, (x, y), r, (0, 95, 200), -1, cv2.LINE_AA)            # shadowed edge
    for k in range(1, 6):                                                 # light falls off
        rr = int(r * (1 - k * 0.15))
        off = int(r * k * 0.06)
        shade = (40 + k * 12, 120 + k * 14, 235 + min(20, k * 4))
        cv2.circle(img, (x - off, y - off), rr, shade, -1, cv2.LINE_AA)
    cv2.ellipse(img, (x - r // 3, y - r // 3), (max(2, r // 4), max(1, r // 6)),
                -35, 0, 360, WHITE, -1, cv2.LINE_AA)                      # shine
    cv2.circle(img, (x, y), r, (0, 60, 140), 2, cv2.LINE_AA)              # outline


def draw_ball(img, game):
    h, w = img.shape[:2]
    b = game.ball
    bx, bz = (0.5, 1.0) if game.state == g.COUNTDOWN else (b.x, b.z)
    x, y, s = project(bx, ball_height(game), bz, w, h)
    r = int(BALL_RADIUS * s)
    if TABLE_X[0] <= bx <= TABLE_X[1]:   # shadow on the table top
        fx, fy, _ = project(bx, TABLE_Y, bz, w, h)
        shadow = img.copy()
        cv2.ellipse(shadow, (fx, fy), (max(4, int(r * 1.1)), max(2, int(r * 0.35))), 0, 0, 360, (0, 0, 0), -1)
        cv2.addWeighted(shadow, 0.5, img, 0.5, 0, img)
    draw_pingpong(img, x, y, r)


def draw_hit_box(img, game):
    """Lilac square on your side of the table where the ball will arrive."""
    b = game.ball
    if game.state != g.RALLY or not (b.incoming or game.arrival_time is not None):
        return
    h, w = img.shape[:2]
    cx, cy, half = int(b.target[0] * w), int(b.target[1] * h), int(config.HIT_BOX * h)
    fill = img.copy()
    cv2.rectangle(fill, (cx - half, cy - half), (cx + half, cy + half), LILAC, -1)
    cv2.addWeighted(fill, 0.25, img, 0.75, 0, img)
    cv2.rectangle(img, (cx - half, cy - half), (cx + half, cy + half), LILAC, 4, cv2.LINE_AA)
    text(img, "HIT HERE", (cx - 70, cy - half - 12), 0.8, LILAC)


def draw_paddle_shape(img, center, grip, r, outline=None, alpha=1.0):
    """Wooden handle from the grip to a red rubber blade."""
    c, gp = np.array(center, float), np.array(grip, float)
    d = c - gp
    n = np.linalg.norm(d)
    d = d / n if n > 1e-6 else np.array([0.0, -1.0])
    neck = c - d * r * 0.9
    layer = img.copy()
    cv2.line(layer, tuple(gp.astype(int)), tuple(neck.astype(int)), WOOD, max(4, int(r * 0.4)), cv2.LINE_AA)
    cv2.circle(layer, tuple(c.astype(int)), int(r), RUBBER, -1, cv2.LINE_AA)
    cv2.circle(layer, tuple(c.astype(int)), int(r), outline or RUBBER_EDGE, 6 if outline else 3, cv2.LINE_AA)
    if alpha >= 1.0:
        img[:] = layer
    else:
        cv2.addWeighted(layer, alpha, img, 1 - alpha, 0, img)


def draw_opponent(img, game):
    """The other player's paddle at the far end, following the ball."""
    h, w = img.shape[:2]
    b = game.ball
    ox = 0.5 if game.state in (g.LOBBY, g.COUNTDOWN) else min(TABLE_X[1], max(TABLE_X[0], b.x))
    cx, cy, s = project(ox, OPP_Y, 1.0, w, h)
    r = 0.09 * h * s
    draw_paddle_shape(img, (cx, cy), (cx + r * 0.9, cy + r * 1.8), r)


def draw_player_paddle(img, pad, game):
    if pad is None:
        return
    h, w = img.shape[:2]
    center, grip, radius = pad
    b = game.ball
    in_zone = game.state == g.RALLY and g.box_dist(center, b.target) <= config.HIT_BOX
    draw_paddle_shape(img, (center[0] * w, center[1] * h), (grip[0] * w, grip[1] * h),
                      max(15, radius * h), GREEN if in_zone else None, alpha=0.8)


def draw_scene(img, game, pad):
    """Table, net, ball, both paddles -- with the ball hidden behind the net when it's on the far side."""
    draw_table(img)
    draw_opponent(img, game)
    show_ball = game.state in (g.RALLY, g.PAUSED, g.COUNTDOWN)
    far_side = game.state == g.COUNTDOWN or game.ball.z > NET_Z
    if show_ball and far_side:
        draw_ball(img, game)
    draw_net(img)
    draw_hit_box(img, game)
    draw_player_paddle(img, pad, game)
    if show_ball and not far_side:
        draw_ball(img, game)


def draw_tags(img, tags, stab):
    """Thin green square while a tag is being read, thick + bold once it registers."""
    for tag_id, corners in tags:
        pts = corners.astype(int)
        x, y = pts[:, 0].min(), pts[:, 1].min() - 12
        name = config.LEVELS.get(tag_id, ("?",))[0]
        if stab.registered(tag_id):
            cv2.polylines(img, [pts], True, GREEN, 10)
            text(img, f"{name.upper()}!", (x, y), 1.4, GREEN, 5)
        else:
            cv2.polylines(img, [pts], True, GREEN, 2)
            text(img, f"{name} {int(stab.progress(tag_id) * 100)}%", (x, y), 0.7, GREEN)


def check_icon(img, center, r, state):
    """Step marker: green check when done, yellow ring when current, gray ring when later."""
    if state == "done":
        cv2.circle(img, center, r, GREEN, -1, cv2.LINE_AA)
        cx, cy = center
        cv2.polylines(img, [np.array([(cx - r // 2, cy), (cx - r // 8, cy + r // 2), (cx + r // 2, cy - r // 2)])],
                      False, WHITE, 4, cv2.LINE_AA)
    else:
        cv2.circle(img, center, r, YELLOW if state == "now" else GRAY, 3, cv2.LINE_AA)


def draw_steps(img, game, cal, cal_left, swings, level_locked, ml_info=None):
    """Setup checklist shown until the game starts: what to do, in order."""
    if cal is None and game.state != g.LOBBY:
        return
    h, w = img.shape[:2]
    pw, ph = 820, 400
    x0, y0 = (w - pw) // 2, (h - ph) // 2 + 20
    panel = img.copy()
    cv2.rectangle(panel, (x0, y0), (x0 + pw, y0 + ph), (25, 25, 25), -1)
    cv2.addWeighted(panel, 0.7, img, 0.3, 0, img)
    cv2.rectangle(img, (x0, y0), (x0 + pw, y0 + ph), WHITE, 2)
    text(img, "SETUP - follow the steps", (x0 + 25, y0 + 50), 1.1, WHITE, 3)

    current = {"STILL": 0, "SWING": 1}.get(cal, 2)
    lines = [
        ("Hold the paddle still", None),
        (f"Swing the paddle {config.CAL_SWINGS} times", None),
        ("Hold up a level card to start", "1 = Easy    2 = Medium    3 = Hard"),
        ("3, 2, 1... hit the ball!", "Paddle on the lilac square, swing when the ball arrives"),
    ]
    if ml_info:   # the trained model replaces calibration
        lines[0] = ("Paddle connected", None)
        lines[1] = (f"Swing model loaded ({ml_info})", None)
    if cal == "STILL":
        lines[0] = (f"Hold the paddle still  ({cal_left:.0f}s)", None)
    if level_locked:
        lines[2] = (f"{game.level_name} chosen - starts after calibration", None)

    for i, (msg, sub_msg) in enumerate(lines):
        y = y0 + 110 + i * 80
        done = i < current or (i == 2 and level_locked)
        state = "done" if done else "now" if i == current else "later"
        check_icon(img, (x0 + 45, y - 10), 18, state)
        color = GREEN if done else YELLOW if state == "now" else GRAY
        text(img, f"{i + 1}. {msg}", (x0 + 80, y), 0.95, color, 3 if state == "now" else 2)
        if sub_msg and state != "done":
            text(img, sub_msg, (x0 + 105, y + 32), 0.7, WHITE if state == "now" else GRAY)
        if i == 1 and cal == "SWING":   # one dot per recorded swing
            for k in range(config.CAL_SWINGS):
                c = (x0 + 560 + k * 45, y - 10)
                cv2.circle(img, c, 14, GREEN if k < swings else GRAY, -1 if k < swings else 2, cv2.LINE_AA)


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
    if time.monotonic() - paddle.last_swing() < 0.5:
        text(img, "SWING!", (w - 190, 340), 1.3, GREEN, 4)

    # Why the last ball was missed, for a couple of seconds after.
    if game.miss_reason and time.monotonic() - game.miss_time < 2.5:
        centered(img, f"Miss: {game.miss_reason}", 75, 1.0, RED, 3)

    if game.state == g.RALLY:
        centered(img, "Paddle on the lilac square - swing when the ball gets there", 30, 0.8)
    elif game.state == g.COUNTDOWN:
        left = max(0.0, game.state_until - time.monotonic())
        # Above the far wall, so the ball waiting there stays visible.
        centered(img, f"Get ready - {game.level_name}", h // 2 - 240, 1.2, WHITE)
        centered(img, str(int(left) + 1), h // 2 - 100, 5.0, YELLOW, 12)
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
    sounds = Sounds()   # applause on a hit, booing on a miss
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
    cal = "STILL" if paddle.needs_calibration and not args.sim else None
    cal_until = 0.0
    start_after_cal = False   # a level tag shown during calibration starts the game once it's done
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
            # pad = (blade center, grip, blade radius); the blade center is what hits.
            if vision:
                pad = vision.paddle(frame)
            elif mouse[0]:
                cx, cy = mouse[0][0] / w, mouse[0][1] / h
                pad = ((cx, cy), (cx + 0.03, cy + 0.15), 0.07)
            else:
                pad = None

            # Tags are read even during calibration (and drawn through the
            # countdown) so you always see the green square when one is seen.
            tags = []
            if vision and game.state in (g.LOBBY, g.GAME_OVER, g.COUNTDOWN):
                tags = vision.detect_tags(frame)
                for tag_id in stab.update([t for t, _ in tags]):
                    if tag_id in config.LEVELS and game.state in (g.LOBBY, g.GAME_OVER):
                        game.set_level(tag_id)
                        haptics.level_light(config.LEVELS[tag_id][2])
                        haptics.confirm()
                        if game.state == g.LOBBY:
                            if cal is None:
                                game.start(now)
                            else:
                                start_after_cal = True

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
                if start_after_cal and game.state == g.LOBBY:
                    game.start(now)
                start_after_cal = False

            # --- game ---
            if cal is None:
                events = game.update(now, dt, pad and pad[0], paddle.swung_between, paddle.tilt())
                for ev in events:
                    if ev in HAPTIC_EVENTS:
                        getattr(haptics, ev)()
                    if ev == "hit":
                        sounds.play("applause")
                    elif ev == "miss":
                        sounds.play("boo")
                board.set_score(game.streak, now)
            board.tick(now)

            # --- draw ---
            draw_scene(view, game, pad)
            draw_steps(view, game, cal, max(0.0, cal_until - now),
                       paddle.swings_recorded() if cal == "SWING" else 0, bool(cal and start_after_cal),
                       paddle.ml_info)
            if vision:
                draw_tags(view, tags, stab)   # on top of the steps panel so the green square shows
            draw_hud(view, game, paddle, board)
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
            elif key == ord("c") and paddle.needs_calibration and not args.sim:
                paddle.begin_still()
                cal, cal_until = "STILL", now + config.CAL_STILL_S
    finally:
        print(f"Final best streak this session: {game.best}")
        board.set_score(game.streak, time.monotonic())
        board.close()
        sounds.close()
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
