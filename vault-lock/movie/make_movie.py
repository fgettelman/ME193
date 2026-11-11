"""
Make the stop-motion movie that explains the finger lock's neural network,
with no filming: every frame is drawn as paper cut-outs on a paper
background, 12 frames a second, with the small "hand-moved" wobble of real
stop motion. It uses the project's real data: hand pictures from data/, the
trained network's actual feature maps and outputs, and the Grad-CAM images.

    python movie/make_movie.py            # the whole movie (a few minutes)
    python movie/make_movie.py --stills   # one test picture per scene, to check the layout

Scene lengths follow the voice-over in MOVIE_SCRIPT.md (each scene is read
by the Mac's text-to-speech to time it). Outputs, in movie/:
    vault_movie.mp4         silent: record your voice-over on top of this
    vault_movie_guide.mp4   with a computer voice reading the script and
                            captions, to practise the timing
    vault_movie.srt         the script as subtitles with times
"""

import argparse
import math
import os
import random
import re
import subprocess
import sys
import wave

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw, ImageFilter, ImageFont

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "unoq_app", "python"))

from model import FingerNet, standardize, images_to_batch     # noqa: E402
from gradcam import load_model                                # noqa: E402

W, H, FPS = 1280, 720, 12
OUT = HERE
AUDIO_DIR = os.path.join(HERE, "audio")
VOICE, RATE = "Samantha", 165          # Mac text-to-speech voice and words per minute
LEAD_S, TAIL_S = 0.5, 0.9              # quiet time before / after each scene's narration

# Colours (RGB)
INK = (45, 40, 38)
PAPER = (234, 224, 202)
YELLOW = (255, 226, 110)
ORANGE = (245, 150, 60)
RED = (215, 60, 50)
GREEN = (70, 160, 80)
BLUE = (80, 130, 200)
SKIN = (233, 186, 152)
SKIN_LINE = (170, 115, 85)
WHITE = (252, 250, 244)
GREY = (150, 145, 138)

FONT_FILES = {"chalk": ("/System/Library/Fonts/Supplemental/ChalkboardSE.ttc", 1),
              "bold": ("/System/Library/Fonts/Supplemental/ChalkboardSE.ttc", 2),
              "marker": ("/System/Library/Fonts/MarkerFelt.ttc", 1)}
_fonts = {}


SYMBOLS = set("×→✓✗←↓")                 # not in the handwriting fonts
SYMBOL_FONT = "/System/Library/Fonts/Supplemental/Arial Unicode.ttf"


def font(size, kind="chalk"):
    key = (kind, size)
    if key not in _fonts:
        path, idx = FONT_FILES[kind]
        try:
            _fonts[key] = ImageFont.truetype(path, size, index=idx)
        except OSError:
            _fonts[key] = ImageFont.load_default()
    return _fonts[key]


def sym_font(size):
    key = ("sym", size)
    if key not in _fonts:
        _fonts[key] = ImageFont.truetype(SYMBOL_FONT, size)
    return _fonts[key]


def draw_text(d, x, y, text, size, kind, fill, anchor="mm", spacing=6):
    """Text centred (anchor "mm") or from the left ("lm"), with symbols the
    handwriting font lacks drawn in a fallback font."""
    f = font(size, kind)
    asc, desc = f.getmetrics()
    lh = asc + desc + spacing
    lines = text.split("\n")
    top = y - (lh * len(lines) - spacing) / 2
    for k, line in enumerate(lines):
        runs, cur, cur_sym = [], "", None
        for ch in line:
            is_sym = ch in SYMBOLS
            if cur and is_sym != cur_sym:
                runs.append((cur, cur_sym))
                cur = ""
            cur, cur_sym = cur + ch, is_sym
        if cur:
            runs.append((cur, cur_sym))
        runs = [(t, sym_font(size) if s else f) for t, s in runs]
        width = sum(ff.getlength(t) for t, ff in runs)
        x0 = x - width / 2 if anchor[0] == "m" else x
        cy = top + k * lh + (asc + desc) / 2
        for t, ff in runs:
            d.text((x0, cy), t, font=ff, fill=fill, anchor="lm")
            x0 += ff.getlength(t)


# --- Paper background ---------------------------------------------------------------

def make_paper(seed):
    rng = np.random.default_rng(seed)
    low = cv2.GaussianBlur(rng.normal(0, 1, (H // 8, W // 8)).astype(np.float32), (0, 0), 3)
    low = cv2.resize(low, (W, H)) * 9
    fine = rng.normal(0, 3.2, (H, W)).astype(np.float32)
    img = np.array(PAPER, np.float32)[None, None, :] + (low + fine)[:, :, None]
    yy, xx = np.mgrid[0:H, 0:W]
    vign = 1 - 0.10 * (((xx - W / 2) / (W / 2)) ** 2 + ((yy - H / 2) / (H / 2)) ** 2)
    return np.clip(img * vign[:, :, None], 0, 255).astype(np.uint8)


PAPERS = [make_paper(s) for s in range(3)]


class Frame:
    """One photo of the stop-motion set."""

    def __init__(self, i):
        self.i = i
        r = random.Random(i // 2 * 31 + 7)
        base = PAPERS[(i // 2) % 3].astype(np.float32) * (1 + r.uniform(-0.012, 0.012))
        self.img = Image.fromarray(np.clip(base, 0, 255).astype(np.uint8))
        self.d = ImageDraw.Draw(self.img, "RGBA")

    def jit(self, eid, amp=1.4, rot=0.7):
        """Small random wobble for one piece of paper, new every 2 frames."""
        r = random.Random((self.i // 2) * 7919 + eid * 104729 + 13)
        return r.uniform(-amp, amp), r.uniform(-amp, amp), r.uniform(-rot, rot)


# --- Drawing helpers -------------------------------------------------------------------

def ease(p):
    p = min(1.0, max(0.0, p))
    return p * p * (3 - 2 * p)


def ph(p, a, b):
    """Progress 0..1 through the part of the scene from fraction a to b."""
    return min(1.0, max(0.0, (p - a) / (b - a))) if b > a else float(p >= a)


def lerp(a, b, t):
    return a + (b - a) * t


def sprite(fr, im, cx, cy, angle=0.0, eid=0, shadow=True, amp=1.4, rot=0.7, scale=1.0):
    """Paste a paper cut-out (RGBA image) centred at (cx, cy) with a shadow."""
    if scale <= 0.01:
        return
    dx, dy, dr = fr.jit(eid, amp, rot)
    if scale != 1.0:
        im = im.resize((max(1, int(im.width * scale)), max(1, int(im.height * scale))), Image.LANCZOS)
    a = angle + dr
    r = im.rotate(a, resample=Image.BICUBIC, expand=True) if abs(a) > 0.05 else im
    x, y = int(cx + dx - r.width / 2), int(cy + dy - r.height / 2)
    if shadow:
        alpha = r.getchannel("A").point(lambda v: int(v * 0.33))
        sh = Image.new("RGBA", r.size, (35, 25, 15, 0))
        sh.putalpha(alpha)
        sh = sh.filter(ImageFilter.GaussianBlur(4))
        fr.img.paste(sh, (x + 5, y + 7), sh)
    fr.img.paste(r, (x, y), r)


_cards = {}


def card(w, h, fill=WHITE, text=None, size=28, color=INK, kind="chalk", radius=12, border=None, bw=3):
    key = (w, h, fill, text, size, color, kind, radius, border, bw)
    if key not in _cards:
        im = Image.new("RGBA", (w, h), (0, 0, 0, 0))
        d = ImageDraw.Draw(im)
        d.rounded_rectangle([0, 0, w - 1, h - 1], radius, fill=fill + (255,) if len(fill) == 3 else fill,
                            outline=border, width=bw if border else 0)
        if text:
            draw_text(d, w / 2, h / 2, text, size, kind, color)
        _cards[key] = im
    return _cards[key]


def write(fr, text, x, y, size=30, color=INK, kind="chalk", eid=0, anchor="mm", amp=1.0):
    dx, dy, _ = fr.jit(eid + 5000, amp, 0)
    draw_text(fr.d, x + dx, y + dy, text, size, kind, color, anchor)


def arrow(fr, p0, p1, color=INK, width=6, eid=0, head=18):
    dx, dy, _ = fr.jit(eid + 9000, 1.5, 0)
    x0, y0 = p0[0] + dx, p0[1] + dy
    x1, y1 = p1[0] + dx, p1[1] + dy
    ang = math.atan2(y1 - y0, x1 - x0)
    xe, ye = x1 - head * 0.6 * math.cos(ang), y1 - head * 0.6 * math.sin(ang)
    fr.d.line([(x0, y0), (xe, ye)], fill=color, width=width)
    left = (x1 - head * math.cos(ang - 0.45), y1 - head * math.sin(ang - 0.45))
    right = (x1 - head * math.cos(ang + 0.45), y1 - head * math.sin(ang + 0.45))
    fr.d.polygon([(x1, y1), left, right], fill=color)


def tag(fr, text, eid=1):
    """Scene label, like a sticky note in the top-left corner."""
    w = 40 + int(font(30, "marker").getlength(text) * 1.02)
    sprite(fr, card(w, 56, YELLOW, text, 30, INK, "marker", radius=4), 30 + w / 2, 44, angle=2, eid=eid)


def photo(img_bgr, size, border=10):
    """A printed photo: picture with a white border, as an RGBA cut-out."""
    rgb = cv2.cvtColor(cv2.resize(img_bgr, (size, size), interpolation=cv2.INTER_NEAREST), cv2.COLOR_BGR2RGB)
    im = Image.new("RGBA", (size + 2 * border, size + 2 * border), WHITE + (255,))
    im.paste(Image.fromarray(rgb), (border, border))
    return im


def colormap(a):
    """2-D array -> RGB picture (inferno colours), as BGR for photo()."""
    a = a - a.min()
    a = (a / (a.max() + 1e-6) * 255).astype(np.uint8)
    return cv2.applyColorMap(a, cv2.COLORMAP_INFERNO)


_hands = {}


def hand(n, size=200):
    """Paper cut-out of a hand showing n fingers (0 = fist)."""
    key = (n, size)
    if key in _hands:
        return _hands[key]
    S = size
    im = Image.new("RGBA", (int(S * 1.6), int(S * 2.0)), (0, 0, 0, 0))
    d = ImageDraw.Draw(im)
    cx, cy = im.width / 2, im.height * 0.66
    w = int(S * 0.17)
    fingers = []
    up = [a for a in (-24, -8, 8, 24)]
    if n == 5:
        fingers = [(a, 0.62) for a in up] + [(-68, 0.42)]
    elif n >= 1:
        picks = {1: [-8], 2: [-14, 6], 3: [-20, -2, 16], 4: up}[n]
        fingers = [(a, 0.62) for a in picks]
    for col, extra in ((SKIN_LINE, 6), (SKIN, 0)):
        for a, length in fingers:
            r = math.radians(a)
            x1 = cx + math.sin(r) * S * (0.30 + length)
            y1 = cy - math.cos(r) * S * (0.30 + length)
            d.line([(cx, cy), (x1, y1)], fill=col, width=w + extra)
            d.ellipse([x1 - (w + extra) / 2, y1 - (w + extra) / 2, x1 + (w + extra) / 2, y1 + (w + extra) / 2], fill=col)
        rx, ry = S * 0.40 + extra / 2, S * 0.42 + extra / 2
        d.ellipse([cx - rx, cy - ry, cx + rx, cy + ry], fill=col)
        d.rectangle([cx - S * 0.22 - extra / 2, cy + S * 0.25, cx + S * 0.22 + extra / 2, im.height], fill=col)
    if n == 0:
        for k in range(4):
            x = cx - S * 0.27 + k * S * 0.18
            d.arc([x - 16, cy - S * 0.42, x + 16, cy - S * 0.42 + 26], 180, 360, fill=SKIN_LINE, width=3)
    _hands[key] = im
    return im


# 5x7 digits (same as the UNO Q sketch), for the LED matrix.
BIG = {1: [0x04, 0x0C, 0x04, 0x04, 0x04, 0x04, 0x0E], 2: [0x0E, 0x11, 0x01, 0x02, 0x04, 0x08, 0x1F],
       3: [0x1F, 0x02, 0x04, 0x02, 0x01, 0x11, 0x0E], 4: [0x02, 0x06, 0x0A, 0x12, 0x1F, 0x02, 0x02],
       5: [0x1F, 0x10, 0x1E, 0x01, 0x01, 0x11, 0x0E]}


def uno_q(fr, cx, cy, led=(70, 70, 70), digit=None, entered=0, eid=300, scale=1.0, check=False):
    """The UNO Q board: LED matrix and two RGB LEDs."""
    w, h = int(230 * scale), int(120 * scale)
    sprite(fr, card(w, h, (20, 120, 120), None, radius=8, border=(10, 70, 70)), cx, cy, eid=eid, rot=0.3)
    dx, dy, _ = fr.jit(eid, 1.4, 0)
    sp = 11 * scale
    x0, y0 = cx + dx - w / 2 + 16 * scale, cy + dy - h / 2 + 22 * scale
    lit = set()
    if digit:
        for r, bits in enumerate(BIG[digit]):
            for c in range(5):
                if bits & (0x10 >> c):
                    lit.add((r, c))
    if check:
        for r, c in [(4, 2), (5, 3), (6, 4), (5, 5), (4, 6), (3, 7), (2, 8), (1, 9), (0, 10)]:
            lit.add((r, c))
    for r in range(8):
        for c in range(13):
            on = (r, c) in lit or (c in (10, 11) and r in (0, 1, 3, 4, 6, 7) and (r // 3) < entered)
            col = (255, 90, 60) if on else (10, 70, 70)
            fr.d.ellipse([x0 + c * sp - 3 * scale, y0 + r * sp - 3 * scale, x0 + c * sp + 3 * scale, y0 + r * sp + 3 * scale], fill=col)
    for k in range(2):
        lx, ly = cx + dx + w / 2 - 38 * scale, cy + dy - 22 * scale + k * 44 * scale
        if led != (70, 70, 70):
            fr.d.ellipse([lx - 22 * scale, ly - 22 * scale, lx + 22 * scale, ly + 22 * scale], fill=led + (70,))
        fr.d.ellipse([lx - 11 * scale, ly - 11 * scale, lx + 11 * scale, ly + 11 * scale], fill=led, outline=(20, 40, 40), width=2)
    write(fr, "UNO Q", cx + w / 2 - 40 * scale, cy + h / 2 - 12 * scale, int(16 * scale), (220, 240, 235), "bold", eid=eid)


def safe(fr, cx, cy, open_frac=0.0, led=(70, 70, 70), pressed=False, digit=None, entered=0, check=False, eid=200):
    """The safe: body, door, dial, servo latch, button, UNO Q on top, webcam."""
    bw, bh = 380, 300
    sprite(fr, card(bw, bh, (96, 108, 124), None, radius=14, border=(50, 58, 70), bw=5), cx, cy, eid=eid, rot=0.3)
    dx, dy, _ = fr.jit(eid, 1.4, 0)
    x0, y0 = cx - bw / 2 + 24 + dx, cy - bh / 2 + 24 + dy
    dw, dh = bw - 48, bh - 48
    fr.d.rectangle([x0, y0, x0 + dw, y0 + dh], fill=(40, 42, 48))
    for k in range(3):                                   # gold bars inside
        gx = x0 + 60 + k * 70
        fr.d.polygon([(gx, y0 + dh - 20), (gx + 56, y0 + dh - 20), (gx + 48, y0 + dh - 50), (gx + 8, y0 + dh - 50)], fill=(230, 185, 60))
    dwo = dw * (1 - 0.82 * ease(open_frac))
    fr.d.polygon([(x0, y0), (x0 + dwo, y0 + 10 * open_frac), (x0 + dwo, y0 + dh - 10 * open_frac), (x0, y0 + dh)],
                 fill=(120, 134, 152), outline=(50, 58, 70))
    if open_frac < 0.4:
        dcx, dcy = x0 + dwo * 0.55, y0 + dh / 2
        fr.d.ellipse([dcx - 42, dcy - 42, dcx + 42, dcy + 42], fill=(70, 78, 90), outline=(35, 40, 48), width=4)
        for a in range(0, 360, 30):
            fr.d.line([(dcx + 30 * math.cos(math.radians(a)), dcy + 30 * math.sin(math.radians(a))),
                       (dcx + 38 * math.cos(math.radians(a)), dcy + 38 * math.sin(math.radians(a)))], fill=(200, 200, 205), width=2)
    # servo with its arm (0 deg = locked across the door edge, 90 = up)
    sx, sy = x0 + dw + 6, cy + dy - 40
    fr.d.rectangle([sx - 14, sy - 20, sx + 14, sy + 20], fill=(30, 30, 34))
    a = math.radians(180 - 90 * ease(open_frac))
    fr.d.line([(sx, sy), (sx + 60 * math.cos(a), sy - 60 * math.sin(a))], fill=WHITE, width=9)
    write(fr, "servo", sx, sy + 38, 16, WHITE, "bold", eid=eid + 1)
    # button on the side
    bx, by = cx + bw / 2 + 36 + dx, cy + 70 + dy
    fr.d.rectangle([bx - 28, by - 22, bx + 28, by + 26], fill=(60, 60, 64))
    fr.d.ellipse([bx - 18, by - (8 if pressed else 18), bx + 18, by + (14 if pressed else 6)], fill=RED)
    write(fr, "button", bx, by + 46, 18, INK, "bold", eid=eid + 2)
    # webcam on a stick, looking down at the hand spot
    fr.d.line([(cx + bw / 2 + 120 + dx, cy + bh / 2 + dy), (cx + bw / 2 + 120 + dx, cy - bh / 2 - 60 + dy)], fill=(70, 70, 70), width=8)
    fr.d.rounded_rectangle([cx + bw / 2 + 100 + dx, cy - bh / 2 - 90 + dy, cx + bw / 2 + 200 + dx, cy - bh / 2 - 50 + dy], 8, fill=(30, 30, 30))
    fr.d.ellipse([cx + bw / 2 + 168 + dx, cy - bh / 2 - 82 + dy, cx + bw / 2 + 192 + dx, cy - bh / 2 - 58 + dy], fill=(90, 120, 160))
    write(fr, "webcam", cx + bw / 2 + 150, cy - bh / 2 - 108, 18, INK, "bold", eid=eid + 3)
    uno_q(fr, cx, cy - bh / 2 - 66, led, digit, entered, eid=eid + 10, check=check)


def cups(fr, x0, y, probs, highlight=None, eid=600, w=88, gap=18, show_pct=True, beads=None):
    for k, p in enumerate(probs):
        cx = x0 + k * (w + gap) + w / 2
        dx, dy, _ = fr.jit(eid + k, 1.2, 0)
        top, bot = y - 120 + dy, y + dy
        fr.d.polygon([(cx - w / 2 + dx, top), (cx + w / 2 + dx, top), (cx + w / 2 - 10 + dx, bot), (cx - w / 2 + 10 + dx, bot)],
                     fill=(250, 248, 240), outline=INK, width=3)
        level = 110 * p
        if level > 1:
            col = GREEN if k == highlight else BLUE
            fr.d.polygon([(cx - w / 2 + 10 + dx + 1, bot - 2), (cx + w / 2 - 10 + dx - 1, bot - 2),
                          (cx + w / 2 - 10 + 10 * level / 120 + dx, bot - level), (cx - w / 2 + 10 - 10 * level / 120 + dx, bot - level)],
                         fill=col + (200,))
        write(fr, str(k), cx, bot + 26, 30, INK, "marker", eid=eid + k)
        if show_pct:
            write(fr, f"{p:.0%}", cx, top - 18, 22, GREEN if k == highlight else INK, "bold", eid=eid + k)


def dial(fr, cx, cy, angle, eid=0, r=20):
    dx, dy, _ = fr.jit(eid, 1, 0)
    fr.d.ellipse([cx - r + dx, cy - r + dy, cx + r + dx, cy + r + dy], fill=WHITE, outline=INK, width=3)
    a = math.radians(angle)
    fr.d.line([(cx + dx, cy + dy), (cx + dx + (r - 4) * math.cos(a), cy + dy - (r - 4) * math.sin(a))], fill=RED, width=4)


def stamp(fr, text, cx, cy, color=RED, size=60, eid=0, angle=-8):
    im = card(int(font(size, "marker").getlength(text)) + 50, size + 40, (0, 0, 0, 0), text, size, color, "marker",
              border=color, bw=6)
    sprite(fr, im, cx, cy, angle=angle, eid=eid, shadow=False)


# --- The project's real data --------------------------------------------------------------

class Data:
    pass


def load_data():
    D = Data()
    model = load_model()
    D.model = model
    import glob
    best = None
    for path in sorted(glob.glob(os.path.join(ROOT, "data", "3", "*_012.png"))):
        im = cv2.imread(path)
        with torch.no_grad():
            p = F.softmax(model(standardize(images_to_batch([im]))), 1)[0].numpy()
        if p[3] > 0.97:
            best = (im, p)
            break
    D.hand_img, D.probs = best
    # The real feature maps of every conv layer for that picture.
    x = standardize(images_to_batch([D.hand_img]))
    feats = []
    with torch.no_grad():
        a = x
        for conv, norm, pool in zip(model.convs, model.norms, model.pools):
            a = F.relu(norm(conv(a)))
            feats.append(a[0].numpy())
            if pool:
                a = F.max_pool2d(a, 2)
    D.feats = feats
    order = np.argsort([-f.std() for f in feats[0]])
    D.maps1 = [colormap(feats[0][c]) for c in order]                      # 16 conv1 maps, most varied first
    D.maps4 = [colormap(np.kron(feats[3][c], np.ones((5, 5)))) for c in range(feats[3].shape[0])]
    D.means4 = np.array([feats[3][c].mean() for c in range(feats[3].shape[0])])
    # What a network with random (untrained) weights says about the same picture.
    torch.manual_seed(3)
    rnd = FingerNet(16, 4).eval()
    with torch.no_grad():
        D.probs_random = F.softmax(rnd(x), 1)[0].numpy()
    # The convolution demo: the picture at 10x10 in shades 0-9 (9 = dark = hand),
    # and a vertical-edge filter run over it.
    g = cv2.cvtColor(D.hand_img, cv2.COLOR_BGR2GRAY)
    small = cv2.resize(g, (10, 10), interpolation=cv2.INTER_AREA).astype(np.float32)
    D.grid = np.clip(np.round((small.max() - small) / (small.max() - small.min() + 1e-6) * 9), 0, 9).astype(int)
    K = np.array([[-1, 0, 1]] * 3)
    D.kernel = K
    D.conv_out = np.array([[int((D.grid[r:r + 3, c:c + 3] * K).sum()) for c in range(8)] for r in range(8)])
    pool_in = np.abs(D.conv_out[2:6, 2:6])
    if pool_in.max() == 0:
        pool_in = np.arange(16).reshape(4, 4)
    D.pool_in = pool_in
    # Pictures for the data scene.
    rng = random.Random(4)
    allp = sorted(glob.glob(os.path.join(ROOT, "data", "*", "*.png")))
    D.pics = [cv2.imread(p) for p in rng.sample(allp, 40)]
    # Grad-CAM pictures.
    wimg = cv2.imread(os.path.join(ROOT, "results", "gradcam_watch.png"))
    D.cam_watch = wimg[:, :256] if wimg is not None else None
    gimg = cv2.imread(os.path.join(ROOT, "results", "gradcam.png"))
    D.cam_good = [gimg[k * 192:(k + 1) * 192, 0:192] for k in (1, 3, 5)] if gimg is not None else []
    # Gradient descent on a bumpy hill, worked out for real (with momentum).
    f = lambda x: 0.075 * (x - 6.4) ** 2 + 0.28 * np.sin(2.1 * x) + 0.3       # noqa: E731
    fp = lambda x: 0.15 * (x - 6.4) + 0.28 * 2.1 * np.cos(2.1 * x)            # noqa: E731
    xs, x, v = [0.4], 0.4, 0.0
    for _ in range(36):
        v = 0.8 * v + fp(x)          # momentum 0.8
        x = x - 0.15 * v             # learning rate 0.15
        xs.append(x)
    grid = np.linspace(0, 11, 2000)
    D.gd_f, D.gd_xs, D.gd_min = f, xs, float(grid[np.argmin(f(grid))])
    return D


# --- The scenes ------------------------------------------------------------------------------
# Each scene is draw(fr, p, D) with p = 0..1 through the scene.

def s_title(fr, p, D):
    write(fr, "How my finger lock's", W / 2, 230, 64, INK, "marker", eid=1)
    write(fr, "neural network works", W / 2, 310, 64, INK, "marker", eid=2)
    write(fr, "a stop-motion movie  ·  ME193  ·  Fiona Gettelman", W / 2, 400, 30, (90, 80, 70), "chalk", eid=3)
    for k, n in enumerate((1, 2, 3, 4, 5)):
        sprite(fr, hand(n, 70), 300 + k * 170, 560, angle=(-8 + 4 * k), eid=10 + k,
               scale=ease(ph(p, 0.1 + 0.08 * k, 0.25 + 0.08 * k)))


def safe_sequence(fr, p, D, cx=470, cy=430, hand_x=1010, code=(1, 4, 2)):
    """Press the button, show three numbers, the safe opens (scenes 1 and 13)."""
    pressed = 0.18 < p < 0.26
    started = p >= 0.22
    entered = sum(p >= t for t in (0.42, 0.58, 0.74))
    showing = None
    for k, (a, b) in enumerate(((0.3, 0.44), (0.46, 0.6), (0.62, 0.76))):
        if a <= p < b:
            showing = code[k]
    opened = p >= 0.8
    led = GREEN if opened else (WHITE if started else (70, 70, 70))
    safe(fr, cx, cy, open_frac=ph(p, 0.8, 0.92), led=led, pressed=pressed, digit=showing,
         entered=entered, check=opened)
    if p < 0.28:
        hx = lerp(1350, cx + 300, ease(ph(p, 0.0, 0.17)))
        hy = cy + 40
        sprite(fr, hand(1, 120), hx, hy, angle=90, eid=50)          # finger points at the button
    elif showing:
        sprite(fr, hand(showing, 150), hand_x, cy - 20, angle=-6, eid=51 + showing)
        write(fr, str(showing), hand_x + 120, cy - 150, 60, ORANGE, "marker", eid=60)
    for k in range(entered):
        sprite(fr, card(60, 70, YELLOW, "*", 44, INK, "marker", radius=4),
               cx - 80 + 80 * k, cy + 230, angle=(-4 + 4 * k), eid=70 + k)
    if opened:
        write(fr, "OPEN!", cx, cy - 300, 64, GREEN, "marker", eid=80)


def s1(fr, p, D):
    tag(fr, "1 · My safe")
    safe_sequence(fr, p, D)


def s2(fr, p, D):
    tag(fr, "2 · What the computer sees")
    size = 384
    px, py = 330, 390
    sprite(fr, photo(D.hand_img, size), px, py - 400 * (1 - ease(ph(p, 0, 0.1))), angle=-2, eid=10)
    # grid lines drawn over the photo, left to right
    g = ph(p, 0.12, 0.4)
    dx, dy, _ = fr.jit(10, 1.4, 0)
    x0, y0 = px - size / 2 + dx, py - size / 2 + dy
    for k in range(0, 65):
        if k / 64 <= g:
            fr.d.line([(x0 + k * 6, y0), (x0 + k * 6, y0 + size)], fill=(40, 40, 40, 70), width=1)
            fr.d.line([(x0, y0 + k * 6), (x0 + size, y0 + k * 6)], fill=(40, 40, 40, 70), width=1)
    if g > 0:
        write(fr, "64 × 64 pixels", px, py + size / 2 + 40, 32, INK, "marker", eid=11)
    z = ph(p, 0.4, 0.75)
    if z > 0:
        r0, c0 = 24, 14                         # a 4x4 patch on the edge of the hand
        mx, my = x0 + (c0 + 2) * 6, y0 + (r0 + 2) * 6
        fr.d.ellipse([mx - 22, my - 22, mx + 22, my + 22], outline=RED, width=5)
        cs = 92
        cxx, cyy = 900, 300
        if z > 0.15:
            arrow(fr, (mx + 26, my), (cxx - 2 * cs - 20, cyy), RED, 5, eid=12)
            im = Image.new("RGBA", (4 * cs + 16, 4 * cs + 16), WHITE + (255,))
            dd = ImageDraw.Draw(im)
            for r in range(4):
                for c in range(4):
                    b, gg, rr = [int(v) for v in D.hand_img[r0 + r, c0 + c]]
                    x, y = 8 + c * cs, 8 + r * cs
                    dd.rectangle([x, y, x + cs - 2, y + cs - 2], fill=(rr, gg, b))
                    tc = (255, 255, 255) if (rr + gg + b) < 380 else INK
                    dd.multiline_text((x + cs / 2, y + cs / 2), f"R {rr}\nG {gg}\nB {b}", font=font(17, "bold"),
                                      fill=tc, anchor="mm", align="center")
            sprite(fr, im, cxx, cyy, angle=1.5, eid=13, scale=ease(ph(z, 0.15, 0.4)))
    if p > 0.75:
        write(fr, "64 × 64 × 3 = 12,288 numbers", 900, 560, 40, INK, "marker", eid=14)
        q = ph(p, 0.82, 0.95)
        for k in range(6):
            if q * 6 > k:
                sprite(fr, card(60, 60, YELLOW, str(k), 34, INK, "marker", radius=4), 690 + k * 80, 650,
                       angle=(-5 + 2 * k), eid=20 + k)
        if q > 0.9:
            write(fr, "?", 1190, 650, 50, RED, "marker", eid=27)


def s3(fr, p, D):
    tag(fr, "3 · One neuron")
    inputs = [("0.8", "× 1.2", "0.96"), ("0.3", "× -0.5", "-0.15"), ("0.5", "× 0.4", "0.20")]
    nx, ny = 560, 380
    for k, (v, w, prod) in enumerate(inputs):
        a = ph(p, 0.03 + 0.08 * k, 0.1 + 0.08 * k)
        if a <= 0:
            continue
        iy = 220 + k * 160
        sprite(fr, card(100, 80, YELLOW, v, 36, INK, "marker", radius=4), 130, iy, angle=(-3 + 3 * k), eid=10 + k, scale=ease(a))
        arrow(fr, (190, iy), (nx - 95, ny + (k - 1) * 40), INK, 5, eid=20 + k)
        mx, my = (190 + nx - 95) / 2, (iy + ny + (k - 1) * 40) / 2
        sprite(fr, card(100, 46, WHITE, w, 26, BLUE, "bold", radius=20, border=BLUE), mx, my - 30, eid=30 + k)
        t = ease(ph(p, 0.3 + 0.05 * k, 0.45 + 0.05 * k))
        if 0 < t < 1:
            sprite(fr, card(84, 40, (255, 210, 190), prod, 22, INK, "bold", radius=8), lerp(mx, nx, t), lerp(my, ny, t), eid=40 + k)
    sprite(fr, card(190, 190, (190, 220, 245), None, radius=95, border=INK, bw=5), nx, ny, eid=50)
    write(fr, "neuron", nx, ny - 120, 30, INK, "marker", eid=51)
    if p > 0.5:
        write(fr, "add up:\n1.01", nx, ny, 34, INK, "marker", eid=52)
    else:
        write(fr, "Σ", nx, ny, 70, INK, "marker", eid=52)
    r = ph(p, 0.55, 0.62)
    if r > 0:
        cx, cy = 1000, 380
        arrow(fr, (nx + 100, ny), (cx - 200, cy), INK, 5, eid=60)
        sprite(fr, card(360, 320, WHITE, None, radius=10, border=INK), cx, cy, angle=1, eid=61, scale=ease(r))
        if r >= 1:
            dx, dy, _ = fr.jit(61, 1.4, 0)
            ox, oy = cx + dx, cy + 60 + dy
            fr.d.line([(ox - 150, oy), (ox + 150, oy)], fill=GREY, width=2)
            fr.d.line([(ox, oy + 50), (ox, oy - 190)], fill=GREY, width=2)
            fr.d.line([(ox - 150, oy), (ox, oy)], fill=RED, width=7)
            fr.d.line([(ox, oy), (ox + 140, oy - 140)], fill=GREEN, width=7)
            write(fr, "ReLU", cx, cy - 125, 40, INK, "marker", eid=62)
            write(fr, "negative → 0", ox - 75, oy + 30, 20, RED, "bold", eid=63)
            write(fr, "positive → goes through", ox + 60, oy + 52, 20, GREEN, "bold", eid=64)
            b1 = ease(ph(p, 0.62, 0.75))
            bx1 = ox - 110
            by1 = lerp(oy + 40, oy - 10, 0) if b1 <= 0 else lerp(oy - 120, oy - 10, b1)
            fr.d.ellipse([bx1 - 12, by1 - 12, bx1 + 12, by1 + 12], fill=RED)
            write(fr, "-1.3", bx1, by1 - 26, 20, RED, "bold", eid=65)
            b2 = ease(ph(p, 0.78, 0.95))
            bx2 = ox + lerp(0, 100, b2)
            fr.d.ellipse([bx2 - 12, oy - (bx2 - ox) - 22, bx2 + 12, oy - (bx2 - ox) + 2], fill=GREEN)
            write(fr, "1.01", bx2 + 30, oy - (bx2 - ox) - 30, 20, GREEN, "bold", eid=66)


def grid_img(vals, cs, colors=None, numbers=True, size=20, signed=False):
    n_r, n_c = vals.shape
    im = Image.new("RGBA", (n_c * cs + 8, n_r * cs + 8), WHITE + (255,))
    d = ImageDraw.Draw(im)
    vmax = max(1, np.abs(vals).max())
    for r in range(n_r):
        for c in range(n_c):
            v = vals[r, c]
            if colors == "dark":
                g = int(250 - 22 * v)
                fill = (g, g - 4, g - 10)
            elif colors == "signed":
                t = abs(v) / vmax
                fill = (255, int(255 - 120 * t), int(255 - 200 * t)) if v > 0 else (int(255 - 170 * t), int(255 - 110 * t), 255)
            elif colors == "hot":
                t = v / vmax
                fill = (255, int(255 - 140 * t), int(255 - 220 * t))
            else:
                fill = WHITE
            x, y = 4 + c * cs, 4 + r * cs
            d.rectangle([x, y, x + cs, y + cs], fill=fill, outline=(120, 120, 120), width=1)
            if numbers:
                tc = WHITE if colors == "dark" and v >= 6 else INK
                d.text((x + cs / 2, y + cs / 2), str(v), font=font(size, "bold"), fill=tc, anchor="mm")
    return im


def s4(fr, p, D):
    tag(fr, "4 · Convolution")
    cs = 38
    if p < 0.64:
        out_x = 0 if p < 0.6 else -1400 * ease(ph(p, 0.6, 0.64))
        gx, gy = 70 + out_x, 150
        sprite(fr, grid_img(D.grid, cs, "dark", size=18), gx + 5 * cs, gy + 5 * cs, eid=10, rot=0.3)
        write(fr, "the picture (10 × 10)", gx + 5 * cs, gy - 26, 26, INK, "marker", eid=11)
        sprite(fr, grid_img(D.kernel, 46, "signed", size=22), 610 + out_x, 300, angle=2, eid=12)
        write(fr, "filter: 9 weights", 610 + out_x, 210, 26, INK, "marker", eid=13)
        ox, oy = 830 + out_x, 190
        k = min(63, int(ph(p, 0.05, 0.55) * 64)) if p >= 0.05 else -1
        shown = np.full((8, 8), 0)
        done = np.zeros((8, 8), bool)
        for j in range(k + 1):
            done[j // 8, j % 8] = True
        vals = np.where(done, D.conv_out, 0)
        im = grid_img(vals, cs, "signed", size=14)
        dd = ImageDraw.Draw(im)
        for r in range(8):
            for c in range(8):
                if not done[r, c]:
                    dd.rectangle([4 + c * cs, 4 + r * cs, 4 + (c + 1) * cs, 4 + (r + 1) * cs], fill=(240, 236, 226))
        sprite(fr, im, ox + 4 * cs, oy + 4 * cs, eid=14, rot=0.3)
        write(fr, "new map: edges light up", ox + 4 * cs, oy - 26, 26, INK, "marker", eid=15)
        if k >= 0:
            r, c = k // 8, k % 8
            dx, dy, _ = fr.jit(10, 1.4, 0)
            x0, y0 = gx + 4 + c * cs + dx, gy + 4 + r * cs + dy
            fr.d.rectangle([x0 - 3, y0 - 3, x0 + 3 * cs + 3, y0 + 3 * cs + 3], outline=ORANGE, width=6)
            fr.d.rectangle([x0, y0, x0 + 3 * cs, y0 + 3 * cs], fill=(255, 170, 60, 50))
            tx, ty = ox + 4 + c * cs + cs / 2, oy + 4 + r * cs + cs / 2
            fr.d.line([(x0 + 3 * cs, y0 + 1.5 * cs), (tx, ty)], fill=ORANGE + (180,), width=3)
        if 0.5 < p < 0.6:
            write(fr, "the same 9 weights at every spot", 640, 640, 34, BLUE, "marker", eid=16)
    elif p < 0.83:
        q = ph(p, 0.64, 0.68)
        write(fr, "real filters from my network, on my hand", W / 2, 140, 36, INK, "marker", eid=20)
        sprite(fr, photo(D.hand_img, 200), 180, 380, angle=-3, eid=21, scale=ease(q))
        for k in range(3):
            a = ease(ph(p, 0.68 + 0.03 * k, 0.72 + 0.03 * k))
            sprite(fr, photo(D.maps1[k], 200), 520 + k * 260, 380, angle=(2 - 2 * k), eid=22 + k, scale=a)
            if a > 0.5:
                arrow(fr, (300, 380), (400 + k * 260, 380 - 60 + 60 * k), GREY, 3, eid=26 + k)
        write(fr, "bright = this filter found its pattern here", W / 2, 560, 28, (90, 80, 70), "chalk", eid=30)
    else:
        a, b = ease(ph(p, 0.83, 0.87)), ease(ph(p, 0.88, 0.92))
        sprite(fr, card(420, 220, (200, 235, 200), "my first layer:\n448 weights", 44, INK, "marker", border=GREEN),
               330, 380, angle=-3, eid=40, scale=a)
        if b > 0:
            write(fr, "vs", 640, 380, 60, INK, "marker", eid=41)
        sprite(fr, card(440, 220, (250, 210, 200), "one neuron per pixel:\n≈ 200,000", 44, INK, "marker", border=RED),
               950, 380, angle=3, eid=42, scale=b)
        if p > 0.94:
            write(fr, "sharing the weights is what makes a CNN small", W / 2, 600, 32, INK, "chalk", eid=43)


def s5(fr, p, D):
    tag(fr, "5 · Width")
    n = 16
    fan = ease(ph(p, 0.05, 0.5))
    shrink = ease(ph(p, 0.55, 0.65))
    for k in range(n):
        r, c = k // 4, k % 4
        tx, ty = 560 + c * 150, 140 + r * 140
        tx = lerp(tx, 120 + c * 62, shrink)
        ty = lerp(ty, 150 + r * 60, shrink)
        x = lerp(200, tx, fan)
        y = lerp(400 + k * 2, ty, fan)
        sprite(fr, photo(D.maps1[k], 120, 6), x, y, angle=(k % 5 - 2) * (1 - fan) * 4, eid=10 + k,
               scale=lerp(1.0, 0.45, shrink))
    if p < 0.55:
        write(fr, "16 filters  →  16 feature maps", 250, 640, 34, INK, "marker", eid=40)
    if p > 0.62:
        write(fr, "conv 1", 210, 390, 26, INK, "marker", eid=41)
        for j, (cnt, lab) in enumerate(((16, "conv 1"), (32, "conv 2"), (64, "conv 3"), (128, "conv 4"))):
            a = ph(p, 0.65 + 0.07 * j, 0.72 + 0.07 * j)
            if a <= 0:
                continue
            x0 = 450 + j * 210
            layers = int(cnt / 4 * ease(a))
            dx, dy, _ = fr.jit(50 + j, 1.4, 0)
            for L in range(layers):
                fr.d.rectangle([x0 + dx + L * 0.6, 560 + dy - L * 4.5, x0 + 120 + dx + L * 0.6, 640 + dy - L * 4.5],
                               fill=(250, 248, 240), outline=(120, 110, 100))
            write(fr, f"{cnt}", x0 + 60, 680, 40, INK, "marker", eid=60 + j)
            write(fr, lab, x0 + 60, 560 - layers * 4.5 - 30, 22, (90, 80, 70), "bold", eid=70 + j)
        if p > 0.9:
            write(fr, "each layer doubles", 940, 140, 34, BLUE, "marker", eid=80)


def s6(fr, p, D):
    tag(fr, "6 · Max pooling")
    cs = 92
    if p < 0.56:
        gx, gy = 110, 170
        sprite(fr, grid_img(D.pool_in, cs, "hot", size=34), gx + 2 * cs, gy + 2 * cs, eid=10, rot=0.3)
        write(fr, "a feature map", gx + 2 * cs, gy - 30, 30, INK, "marker", eid=11)
        out = np.array([[D.pool_in[0:2, 0:2].max(), D.pool_in[0:2, 2:4].max()],
                        [D.pool_in[2:4, 0:2].max(), D.pool_in[2:4, 2:4].max()]])
        ox, oy = 820, 260
        sprite(fr, grid_img(np.zeros((2, 2), int), cs, None, numbers=False), ox + cs, oy + cs, eid=12, rot=0.3)
        write(fr, "after pooling", ox + cs, oy - 30, 30, INK, "marker", eid=13)
        q = ph(p, 0.04, 0.52) * 4
        for b in range(4):
            br, bc = b // 2, b % 2
            local = q - b
            if local < 0:
                continue
            v = out[br, bc]
            if local < 1:
                dx, dy, _ = fr.jit(10, 1.4, 0)
                x0, y0 = gx + 4 + bc * 2 * cs + dx, gy + 4 + br * 2 * cs + dy
                fr.d.rectangle([x0 - 4, y0 - 4, x0 + 2 * cs + 4, y0 + 2 * cs + 4], outline=BLUE, width=7)
            sub = np.argwhere(D.pool_in[br * 2:br * 2 + 2, bc * 2:bc * 2 + 2] == v)[0]
            sx = gx + 4 + (bc * 2 + sub[1]) * cs + cs / 2
            sy = gy + 4 + (br * 2 + sub[0]) * cs + cs / 2
            tx, ty = ox + 4 + bc * cs + cs / 2, oy + 4 + br * cs + cs / 2
            t = ease(ph(local, 0.35, 0.95))
            hop = -80 * math.sin(math.pi * t)
            sprite(fr, card(cs - 14, cs - 14, YELLOW, str(v), 38, INK, "marker", radius=4),
                   lerp(sx, tx, t), lerp(sy, ty, t) + hop, angle=-3 + 6 * t, eid=20 + b)
        if p > 0.5:
            write(fr, "keep the biggest of each 2 × 2", 640, 640, 36, BLUE, "marker", eid=30)
    elif p < 0.76:
        for k, (n, s) in enumerate(((64, 256), (32, 128), (16, 64), (8, 32))):
            a = ease(ph(p, 0.57 + 0.04 * k, 0.61 + 0.04 * k))
            x = 230 + k * 280
            sprite(fr, photo(D.maps1[0] if k == 0 else cv2.resize(D.maps1[0], (n, n), interpolation=cv2.INTER_AREA), s, 6),
                   x, 360, eid=40 + k, scale=a)
            if a > 0.5:
                write(fr, f"{n} × {n}", x, 540, 34, INK, "marker", eid=50 + k)
                if k:
                    arrow(fr, (x - 150, 360), (x - 70, 360), INK, 5, eid=60 + k)
        write(fr, "half as wide, half as tall: less to compute", W / 2, 640, 32, INK, "chalk", eid=70)
    else:
        for side, (col, label) in enumerate(((1, "finger here"), (0, "moved 1 pixel"))):
            g = np.ones((4, 4), int)
            g[:, col] = 9
            a = ease(ph(p, 0.77 + 0.06 * side, 0.82 + 0.06 * side))
            x = 330 + side * 620
            sprite(fr, grid_img(g, 60, "hot", size=26), x - 110, 360, eid=80 + side, scale=a)
            pooled = np.array([[g[0:2, 0:2].max(), g[0:2, 2:4].max()], [g[2:4, 0:2].max(), g[2:4, 2:4].max()]])
            sprite(fr, grid_img(pooled, 60, "hot", size=26), x + 160, 360, eid=82 + side, scale=a)
            if a > 0.5:
                arrow(fr, (x + 20, 360), (x + 90, 360), INK, 5, eid=84 + side)
                write(fr, label, x - 110, 210, 30, INK, "marker", eid=86 + side)
        if p > 0.9:
            stamp(fr, "same answer ✓", W / 2, 600, GREEN, 50, eid=90)


def s7(fr, p, D):
    tag(fr, "7 · Depth")
    blocks = [("conv 1", "edges", 3), ("conv 2", "bits of finger", 8), ("conv 3", "fingers", 18),
              ("conv 4", "\"3 in a row\"", 38)]
    if p < 0.64:
        size = 384
        px, py = 870, 370
        sprite(fr, photo(D.hand_img, size), px, py, angle=1.5, eid=10)
        dx, dy, _ = fr.jit(10, 1.4, 0)
        for k, (name, what, rf) in enumerate(blocks):
            a = ph(p, 0.04 + 0.13 * k, 0.12 + 0.13 * k)
            if a <= 0:
                continue
            y = 590 - k * 100 - 60 * (1 - ease(a))
            cols = [(250, 214, 165), (200, 225, 245), (210, 235, 205), (240, 200, 215)]
            sprite(fr, card(300, 90, cols[k], f"{name}: {what}", 28, INK, "marker", border=INK), 230, y, angle=(k % 2) * 3 - 1.5, eid=20 + k)
            if a >= 1:
                d = rf * size / 64
                cx, cy = px - 20 + dx, py - 10 + dy
                fr.d.ellipse([cx - d / 2, cy - d / 2, cx + d / 2, cy + d / 2], outline=[ORANGE, BLUE, GREEN, RED][k], width=5)
                if k == sum(ph(p, 0.04 + 0.13 * j, 0.12 + 0.13 * j) >= 1 for j in range(4)) - 1:
                    write(fr, f"each value sees {rf} × {rf} pixels", px, py + size / 2 + 40, 30, INK, "marker", eid=30)
    else:
        write(fr, "my tests on practice images", W / 2, 140, 36, INK, "marker", eid=40)
        res = [(2, 44), (3, 49), (4, 83), (5, 91)]
        base = 580
        for k, (dpt, acc) in enumerate(res):
            a = ease(ph(p, 0.66 + 0.06 * k, 0.74 + 0.06 * k))
            x = 380 + k * 170
            h = 3.4 * acc * a
            dx, dy, _ = fr.jit(50 + k, 1.4, 0)
            fr.d.rectangle([x - 50 + dx, base - h + dy, x + 50 + dx, base + dy], fill=(GREEN if dpt == 4 else BLUE) + (230,),
                           outline=INK, width=3)
            write(fr, f"{dpt} layers", x, base + 32, 28, INK, "marker", eid=60 + k)
            if a > 0.6:
                write(fr, f"{acc}%", x, base - h - 26, 30, INK, "bold", eid=70 + k)
        if p > 0.88:
            write(fr, "2 layers can't see\na whole hand", 380, 300, 26, RED, "marker", eid=80)
            write(fr, "my network", 720, 200, 28, GREEN, "marker", eid=81)
            arrow(fr, (720, 220), (720, base - 3.4 * 83 - 50), GREEN, 4, eid=82, head=14)


def s8(fr, p, D):
    tag(fr, "8 · The answer")
    n = 128
    squash = ease(ph(p, 0.18, 0.34))
    pour = ph(p, 0.36, 0.7)
    counts = np.round(D.probs * n).astype(int)
    counts[np.argmax(counts)] += n - counts.sum()
    target = np.repeat(np.arange(6), counts)
    cx0, cy0 = 330, 440                               # cups row start
    cupw, gap = 88, 18
    fill = np.zeros(6)
    for k in range(n):
        r, c = k // 16, k % 16
        gx, gy = 160 + c * 60, 140 + r * 46
        if pour <= 0:
            if squash < 1:
                sprite(fr, photo(D.maps4[k], 40, 2), gx, gy, eid=10 + k, scale=lerp(1, 0.35, squash), shadow=False)
            else:
                col = tuple(int(v) for v in cv2.applyColorMap(np.uint8([[min(255, D.means4[k] / (D.means4.max() + 1e-6) * 255)]]),
                                                               cv2.COLORMAP_INFERNO)[0, 0][::-1])
                dx, dy, _ = fr.jit(10 + k, 1, 0)
                fr.d.ellipse([gx - 8 + dx, gy - 8 + dy, gx + 8 + dx, gy + 8 + dy], fill=col, outline=INK)
        else:
            t = ease(min(1.0, max(0.0, pour * 1.6 - k / n * 0.6)))
            cup = target[k]
            tx = cx0 + cup * (cupw + gap) + cupw / 2
            ty = cy0 + 140 - 30
            x = lerp(gx, tx, t)
            y = lerp(gy, ty, t) - 120 * math.sin(math.pi * t)
            if t < 1:
                fr.d.ellipse([x - 8, y - 8, x + 8, y + 8], fill=ORANGE, outline=INK)
            else:
                fill[cup] += 1
    if pour > 0:
        level = np.minimum(1.0, fill / max(1, n)) if pour < 1 else D.probs
        cups(fr, cx0, cy0 + 140, level if pour < 1 else D.probs, highlight=3 if pour >= 1 else None,
             show_pct=p > 0.72)
    if p < 0.18:
        write(fr, "128 feature maps from conv 4", W / 2, 560, 34, INK, "marker", eid=400)
    elif p < 0.36:
        write(fr, "global average pooling: one number per map", W / 2, 560, 34, INK, "marker", eid=401)
    elif p > 0.72:
        write(fr, "softmax → probabilities: the answer is 3", W / 2, 160, 38, GREEN, "marker", eid=402)
    if p > 0.85:
        stamp(fr, "98,694 weights", 1080, 330, BLUE, 44, eid=403)


def pipeline(fr, p, D, back=0.0, dial_turn=None, probs=None, wrong=None):
    """Picture -> 4 conv blocks -> cups, side by side (scenes 9 and 13)."""
    sprite(fr, photo(D.hand_img, 150), 110, 330, angle=-3, eid=10)
    cols = [(250, 214, 165), (200, 225, 245), (210, 235, 205), (240, 200, 215)]
    for k in range(4):
        x = 300 + k * 150
        sprite(fr, card(120, 230, cols[k], f"conv\n{k + 1}", 30, INK, "marker", border=INK), x, 330, eid=20 + k)
        for j in range(2):
            ang = 90 + (dial_turn[k][j] if dial_turn is not None else 0)
            dial(fr, x, 430 + j * 50 - 30, ang, eid=30 + 2 * k + j)
    if probs is not None:
        cups(fr, 860, 420, probs, highlight=None, w=56, gap=10, show_pct=False)


def s9(fr, p, D):
    tag(fr, "9 · Backpropagation")
    rng = np.random.default_rng(5)
    turns = rng.uniform(-70, 70, (4, 2))
    back = ph(p, 0.32, 0.9)
    # dials turn when the backward wave reaches their block (right to left)
    dial_turn = [[turns[k][j] * ease(ph(back, (3 - k) * 0.22, (3 - k) * 0.22 + 0.2)) for j in range(2)] for k in range(4)]
    pipeline(fr, p, D, dial_turn=dial_turn, probs=D.probs_random)
    fwd = ph(p, 0.04, 0.24)
    if 0 < fwd < 1:
        x = lerp(190, 850, fwd)
        for j in range(3):
            fr.d.ellipse([x - 10 - j * 30, 320, x + 10 - j * 30, 340], fill=BLUE)
    if p > 0.24:
        guess = int(np.argmax(D.probs_random))
        write(fr, f"random weights say: {guess}", 1040, 170, 32, RED, "marker", eid=50)
        write(fr, "✗", 1210, 230, 70, RED, "marker", eid=51)
        write(fr, "loss = how wrong", 1040, 230, 26, INK, "chalk", eid=52)
    if back > 0:
        x = lerp(880, 200, back)
        arrow(fr, (x + 120, 200), (x, 200), RED, 9, eid=60)
        write(fr, "backpropagation", x + 60, 160, 32, RED, "marker", eid=61)
    if p > 0.5:
        write(fr, "chain rule: for every weight — nudge it, does the loss go up or down?", W / 2, 640, 28, INK, "chalk", eid=70)


def s10(fr, p, D):
    tag(fr, "10 · Gradient descent")
    f, xs = D.gd_f, D.gd_xs
    X0, X1 = 0.0, 11.0
    def to_px(x, y):
        return 120 + (x - X0) / (X1 - X0) * 1040, 620 - y * 150
    pts = [to_px(x, f(x)) for x in np.linspace(X0, X1, 200)]
    dx, dy, _ = fr.jit(10, 1.2, 0)
    hill = [(x + dx, y + dy) for x, y in pts] + [(1160 + dx, 680), (120 + dx, 680)]
    fr.d.polygon(hill, fill=(200, 215, 170), outline=(90, 110, 60))
    fr.d.line([(x + dx, y + dy) for x, y in pts], fill=(70, 90, 40), width=5)
    write(fr, "loss", 150, 170, 36, INK, "marker", eid=11)
    vx, vy = to_px(D.gd_min, f(D.gd_min))
    write(fr, "right answers", vx, vy + 45, 30, GREEN, "marker", eid=12)
    s = ph(p, 0.05, 0.85) * (len(xs) - 1)
    i = int(s)
    frac = s - i
    x = lerp(xs[i], xs[min(i + 1, len(xs) - 1)], ease(frac))
    bx, by = to_px(x, f(x))
    for j in range(max(0, i - 6), i + 1):                # faint trail
        tx, ty = to_px(xs[j], f(xs[j]))
        fr.d.ellipse([tx - 5, ty - 21, tx + 5, ty - 11], fill=(120, 120, 120, 90))
    fr.d.ellipse([bx - 16, by - 32, bx + 16, by], fill=(60, 90, 200), outline=INK, width=2)
    write(fr, f"step {i}", 1050, 150, 34, INK, "marker", eid=13)
    write(fr, f"loss {f(x):.2f}", 1050, 195, 28, INK, "bold", eid=14)
    if p > 0.3:
        write(fr, "each step: 64 pictures → move every weight a bit downhill", 680, 105, 28, INK, "chalk", eid=15)
    if p > 0.55:
        write(fr, "momentum keeps it rolling over bumps", 600, 250, 28, BLUE, "marker", eid=16)


def s11(fr, p, D):
    tag(fr, "11 · The data")
    rng = random.Random(9)
    if p < 0.36:
        for k, im in enumerate(D.pics):
            a = ph(p, k / 40 * 0.28, k / 40 * 0.28 + 0.03)
            if a <= 0:
                continue
            x, y = rng.uniform(140, 1140), rng.uniform(150, 600)
            sprite(fr, photo(im, 110, 6), x, y, angle=rng.uniform(-14, 14), eid=10 + k, scale=ease(a))
        if p > 0.2:
            stamp(fr, "2,472 pictures of my hand", W / 2, 660, INK, 40, eid=60, angle=-3)
    elif p < 0.66:
        im = D.pics[3]
        steps = [("turned", lambda m: cv2.warpAffine(m, cv2.getRotationMatrix2D((32, 32), 25, 1), (64, 64), borderMode=cv2.BORDER_REFLECT)),
                 ("zoomed", lambda m: cv2.resize(m[8:56, 8:56], (64, 64))),
                 ("mirrored", lambda m: m[:, ::-1]),
                 ("darker", lambda m: np.clip(m * 0.6, 0, 255).astype(np.uint8))]
        sprite(fr, photo(im, 220), 220, 380, angle=-2, eid=70)
        write(fr, "original", 220, 530, 30, INK, "marker", eid=71)
        for k, (lab, fn) in enumerate(steps):
            a = ease(ph(p, 0.38 + 0.06 * k, 0.43 + 0.06 * k))
            x = 520 + k * 200
            sprite(fr, photo(np.ascontiguousarray(fn(im)), 160), x, 380, angle=(k - 1.5) * 3, eid=72 + k, scale=a)
            if a > 0.5:
                write(fr, lab, x, 500, 30, BLUE, "marker", eid=76 + k)
        write(fr, "random changes every time, so it can't just memorise", W / 2, 640, 30, INK, "chalk", eid=80)
    else:
        sprite(fr, card(380, 240, (210, 190, 160), None, border=INK), 900, 450, eid=90)
        write(fr, "TEST: never seen in training", 900, 290, 34, INK, "marker", eid=89)
        for k in range(6):
            t = ease(ph(p, 0.67 + 0.025 * k, 0.72 + 0.025 * k))
            sprite(fr, photo(D.pics[10 + k], 90, 5), lerp(200 + k * 70, 860 + (k % 3) * 40, t),
                   lerp(300 + (k % 2) * 120, 470, t), angle=(k - 3) * 5, eid=91 + k, scale=lerp(1, 0.5, t))
        if p > 0.86:
            stamp(fr, "95% right", 360, 300, GREEN, 70, eid=100)
            stamp(fr, "1 – 5: perfect", 360, 450, GREEN, 50, eid=101, angle=4)


def s12(fr, p, D):
    tag(fr, "12 · Something new: Grad-CAM")
    if p < 0.16:
        write(fr, "Grad-CAM asks the network:", W / 2, 300, 44, INK, "marker", eid=10)
        write(fr, "\"which pixels made you say that?\"", W / 2, 380, 48, BLUE, "marker", eid=11)
        return
    if D.cam_watch is not None:
        moved = ease(ph(p, 0.58, 0.64))
        size = int(lerp(380, 240, moved))
        x = lerp(420, 220, moved)
        sprite(fr, photo(D.cam_watch, size), x, 390, angle=-3, eid=20)
        write(fr, "is 0 (a fist), says 1", x, 390 + size / 2 + 40, 30, RED, "marker", eid=21)
        if p > 0.3:
            # spotlight lands on the watch (right side of the picture)
            t = ease(ph(p, 0.3, 0.42))
            sx = lerp(x - size * 0.3, x + size * 0.28, t)
            sy = lerp(390 + size * 0.3, 390 - size * 0.08, t)
            r = size * 0.2
            fr.d.ellipse([sx - r, sy - r, sx + r, sy + r], outline=(255, 230, 60), width=7)
        if 0.42 < p < 0.6:
            write(fr, "it was looking at my watch!", 930, 330, 44, RED, "marker", eid=22)
            write(fr, "(I wore it in my \"1\" pictures)", 930, 390, 30, INK, "chalk", eid=23)
        if p > 0.36:
            write(fr, "✗", x + size / 2 - 10, 390 - size / 2 + 10, 80, RED, "marker", eid=24)
    if p > 0.64:
        write(fr, "after more pictures, with and without the watch:", 790, 170, 30, INK, "marker", eid=30)
        for k, im in enumerate(D.cam_good):
            a = ease(ph(p, 0.66 + 0.05 * k, 0.71 + 0.05 * k))
            x = 560 + k * 230
            sprite(fr, photo(im, 200), x, 390, angle=(k - 1) * 3, eid=31 + k, scale=a)
            if a > 0.5:
                write(fr, "✓", x + 90, 300, 60, GREEN, "marker", eid=35 + k)
        if p > 0.85:
            stamp(fr, "now it looks at the fingers", 790, 610, GREEN, 44, eid=40, angle=-2)


def s13(fr, p, D):
    tag(fr, "13 · On the Arduino")
    if p < 0.34:
        # laptop
        sprite(fr, card(300, 190, (200, 200, 205), None, radius=10, border=(90, 90, 95)), 260, 360, eid=10)
        sprite(fr, card(360, 30, (170, 170, 175), None, radius=6), 260, 470, eid=11)
        write(fr, "laptop: trained here", 260, 540, 28, INK, "marker", eid=12)
        uno_q(fr, 980, 360, eid=20, scale=1.4)
        write(fr, "Arduino UNO Q", 980, 470, 30, INK, "marker", eid=21)
        cols = [(250, 214, 165), (200, 225, 245), (210, 235, 205), (240, 200, 215)]
        for k in range(4):
            t = ease(ph(p, 0.04 + 0.05 * k, 0.16 + 0.05 * k))
            x = lerp(200 + k * 40, 900 + k * 50, t)
            y = lerp(360, 250, t) - 160 * math.sin(math.pi * t)
            sprite(fr, card(44, 70, cols[k], str(k + 1), 24, INK, "marker", border=INK, radius=4), x, y, angle=20 * math.sin(6 * t), eid=30 + k)
        if p > 0.25:
            write(fr, "25 ms per picture  ·  15 pictures a second", W / 2, 640, 34, BLUE, "marker", eid=40)
    else:
        safe_sequence(fr, ph(p, 0.34, 1.0), D)


def s_end(fr, p, D):
    write(fr, "Vault Lock", W / 2, 250, 80, INK, "marker", eid=1)
    write(fr, "made by Fiona Gettelman  ·  ME193", W / 2, 340, 36, (90, 80, 70), "chalk", eid=2)
    write(fr, "a CNN built from scratch in PyTorch, running on an Arduino UNO Q", W / 2, 410, 28, (90, 80, 70), "chalk", eid=3)
    sprite(fr, hand(3, 90), W / 2, 570, angle=-5, eid=4)


SCENES = [s1, s2, s3, s4, s5, s6, s7, s8, s9, s10, s11, s12, s13]


# --- Script, timing and audio --------------------------------------------------------------------

def read_script():
    """[(number, title, voice-over text)] from MOVIE_SCRIPT.md."""
    text = open(os.path.join(ROOT, "MOVIE_SCRIPT.md")).read()
    out = []
    for block in re.split(r"\n## ", text)[1:]:
        head = block.splitlines()[0]
        m = re.match(r"(\d+)\.\s*(.+?)\s*(\*\(optional\)\*)?\s*\(", head)
        if not m:
            continue
        vo = " ".join(l[1:].strip() for l in block.splitlines() if l.startswith(">"))
        out.append((int(m.group(1)), m.group(2).strip(), vo))
    return out


def speak(text, path):
    subprocess.run(["say", "-v", VOICE, "-r", str(RATE), "-o", path, "--file-format=WAVE",
                    "--data-format=LEI16@22050", text], check=True)
    with wave.open(path) as w:
        return w.getnframes() / w.getframerate()


def build_audio(script):
    """Speak each scene; returns scene durations and writes the joined guide track."""
    os.makedirs(AUDIO_DIR, exist_ok=True)
    durs, clips = [], []
    for n, title, vo in script:
        path = os.path.join(AUDIO_DIR, f"scene{n:02d}.wav")
        d = speak(vo, path)
        clips.append(path)
        durs.append(LEAD_S + d + TAIL_S)
        print(f"  scene {n:2d} {title:28s} {d:5.1f} s of narration")
    return durs, clips


def join_audio(clips, durs, title_s, end_s, path):
    rate = 22050
    with wave.open(path, "wb") as out:
        out.setnchannels(1)
        out.setsampwidth(2)
        out.setframerate(rate)
        out.writeframes(b"\0\0" * int(title_s * rate))
        for clip, dur in zip(clips, durs):
            with wave.open(clip) as w:
                data = w.readframes(w.getnframes())
            n_frames = int(round(dur * FPS)) / FPS          # scenes are whole frames long
            out.writeframes(b"\0\0" * int(LEAD_S * rate))
            out.writeframes(data)
            pad = int(n_frames * rate) - int(LEAD_S * rate) - len(data) // 2
            out.writeframes(b"\0\0" * max(0, pad))
        out.writeframes(b"\0\0" * int(end_s * rate))


def sentences(text):
    return [s.strip() for s in re.split(r"(?<=[.!?…])\s+", text) if s.strip()]


def caption_times(script, durs, title_s):
    """[(start, end, sentence)], each sentence timed by its share of the words."""
    caps, t0 = [], title_s
    for (n, title, vo), dur in zip(script, durs):
        n_frames = int(round(dur * FPS)) / FPS
        speech = n_frames - LEAD_S - TAIL_S
        ss = sentences(vo)
        words = [len(s.split()) for s in ss]
        t = t0 + LEAD_S
        for s, wds in zip(ss, words):
            dt = speech * wds / sum(words)
            caps.append((t, t + dt, s))
            t += dt
        t0 += n_frames
    return caps


def write_srt(caps, path):
    def ts(t):
        h, rem = divmod(t, 3600)
        m, s = divmod(rem, 60)
        return f"{int(h):02d}:{int(m):02d}:{int(s):02d},{int((s % 1) * 1000):03d}"
    with open(path, "w") as f:
        for k, (a, b, s) in enumerate(caps, 1):
            f.write(f"{k}\n{ts(a)} --> {ts(b)}\n{s}\n\n")


def caption(img, text):
    d = ImageDraw.Draw(img, "RGBA")
    fnt = font(28, "chalk")
    words, lines, line = text.split(), [], ""
    for w in words:
        if fnt.getlength(line + " " + w) > W - 160:
            lines.append(line)
            line = w
        else:
            line = (line + " " + w).strip()
    lines.append(line)
    h = 40 * len(lines) + 20
    d.rectangle([0, H - h, W, H], fill=(20, 20, 20, 180))
    for k, ln in enumerate(lines):
        d.text((W / 2, H - h + 30 + 40 * k), ln, font=fnt, fill=(255, 255, 255), anchor="mm")


# --- Main ---------------------------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description="Make the stop-motion movie")
    ap.add_argument("--stills", action="store_true", help="just save test pictures of each scene")
    args = ap.parse_args()

    print("Loading the network and pictures...")
    D = load_data()
    script = read_script()
    assert len(script) == len(SCENES), f"MOVIE_SCRIPT.md has {len(script)} scenes, the movie {len(SCENES)}"

    if args.stills:
        out = os.path.join(HERE, "stills")
        os.makedirs(out, exist_ok=True)
        for k, scene in enumerate([s_title] + SCENES + [s_end]):
            row = []
            for p in (0.2, 0.5, 0.8, 0.97):
                fr = Frame(k * 100 + int(p * 50))
                scene(fr, p, D)
                row.append(fr.img.resize((W // 3, H // 3)))
            sheet = Image.new("RGB", (W // 3 * 4, H // 3))
            for j, im in enumerate(row):
                sheet.paste(im, (j * W // 3, 0))
            sheet.save(os.path.join(out, f"scene{k:02d}.jpg"), quality=85)
        print(f"Saved {out}/scene*.jpg")
        return

    import imageio_ffmpeg
    print("Timing the scenes with text-to-speech...")
    durs, clips = build_audio(script)
    title_s, end_s = 3.5, 4.0
    caps = caption_times(script, durs, title_s)
    write_srt(caps, os.path.join(OUT, "vault_movie.srt"))
    audio = os.path.join(AUDIO_DIR, "guide_voice.wav")
    join_audio(clips, durs, title_s, end_s, audio)

    plan = [(s_title, title_s)] + list(zip(SCENES, durs)) + [(s_end, end_s)]
    total = sum(int(round(d * FPS)) for _, d in plan)
    print(f"Drawing {total} frames ({total / FPS / 60:.1f} minutes)...")
    silent_path = os.path.join(OUT, "vault_movie.mp4")
    guide_tmp = os.path.join(AUDIO_DIR, "guide_video.mp4")
    writers = [imageio_ffmpeg.write_frames(pth, (W, H), fps=FPS, codec="libx264", quality=8,
                                           pix_fmt_out="yuv420p", macro_block_size=16)
               for pth in (silent_path, guide_tmp)]
    for w in writers:
        w.send(None)
    i, t = 0, 0.0
    for scene, dur in plan:
        n = int(round(dur * FPS))
        for j in range(n):
            fr = Frame(i)
            scene(fr, j / max(1, n - 1), D)
            writers[0].send(fr.img.tobytes())
            g = fr.img.copy()
            now = t + j / FPS
            cur = next((s for a, b, s in caps if a <= now < b), None)
            if cur:
                caption(g, cur)
            writers[1].send(g.tobytes())
            i += 1
            if i % 240 == 0:
                print(f"  {i}/{total} frames", flush=True)
        t += n / FPS
    for w in writers:
        w.close()

    ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    guide_path = os.path.join(OUT, "vault_movie_guide.mp4")
    subprocess.run([ffmpeg, "-y", "-loglevel", "error", "-i", guide_tmp, "-i", audio, "-c:v", "copy",
                    "-c:a", "aac", "-b:a", "128k", "-shortest", guide_path], check=True)
    os.remove(guide_tmp)
    print(f"\nDone ({t / 60:.1f} minutes):\n  {os.path.relpath(silent_path)}   silent, for your voice-over"
          f"\n  {os.path.relpath(guide_path)}   computer voice + captions, to practise the timing"
          f"\n  movie/vault_movie.srt   the script as timed subtitles")


if __name__ == "__main__":
    main()
