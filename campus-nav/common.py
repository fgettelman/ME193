"""Small helpers shared by the laptop scripts."""

import json
import os
import sys
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "unoq_app", "python"))

import settings as S          # noqa: E402,F401
import store as st            # noqa: E402,F401

DATA = os.path.join(HERE, "data")          # the laptop's copy of the walks (same layout as the board)
STEP_CALIB = os.path.join(HERE, "..", "step-counter", "calibration.json")


def url(host, path):
    return f"http://{host}:{S.WEB_PORT}{path}"


def get(host, path, timeout=10):
    with urllib.request.urlopen(url(host, path), timeout=timeout) as r:
        return json.loads(r.read())


def post(host, path, body, timeout=30):
    req = urllib.request.Request(url(host, path), data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def step_lengths():
    """Walking step length per person from the step counter's 5 m walks."""
    try:
        with open(STEP_CALIB) as f:
            cal = json.load(f)
    except (OSError, ValueError):
        return {}
    out = {}
    for name, walks in cal.items():
        w = [x for x in walks if x.get("kind") == "walk" and x.get("steps")]
        if w:
            out[name] = round(sum(x["meters"] for x in w) / sum(x["steps"] for x in w), 3)
    return out
