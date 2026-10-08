"""
Camera vision: MediaPipe Pose for the paddle wrist, AprilTags for level and
start. Pose runs on the un-mirrored frame and the x-coordinates are flipped,
because AprilTags can't be decoded from a mirrored image.
"""

import os
import shutil
import ssl
import time
import urllib.request

import certifi
import cv2
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision
from pupil_apriltags import Detector

import config

LEFT_WRIST, RIGHT_WRIST = 15, 16
HERE = os.path.dirname(os.path.abspath(__file__))


def _model_path():
    path = os.path.join(HERE, config.POSE_MODEL_FILE)
    if not os.path.exists(path):
        print("Downloading MediaPipe pose model (one time)...")
        # certifi's CA bundle: python.org installs on macOS often can't verify HTTPS otherwise.
        ctx = ssl.create_default_context(cafile=certifi.where())
        with urllib.request.urlopen(config.POSE_MODEL_URL, context=ctx) as resp, open(path, "wb") as f:
            shutil.copyfileobj(resp, f)
    return path


class Vision:
    def __init__(self, start_ms):
        options = mp_vision.PoseLandmarkerOptions(
            base_options=mp_python.BaseOptions(model_asset_path=_model_path()),
            running_mode=mp_vision.RunningMode.VIDEO,
            num_poses=1,
        )
        self.landmarker = mp_vision.PoseLandmarker.create_from_options(options)
        self.start_ms = start_ms
        self._last_ts = -1
        self.tags = Detector(families=config.TAG_FAMILY)
        self.wrist_index = RIGHT_WRIST if config.PADDLE_HAND == "right" else LEFT_WRIST

    def _small(self, frame):
        h, w = frame.shape[:2]
        scale = config.PROCESS_WIDTH / w
        return cv2.resize(frame, (config.PROCESS_WIDTH, int(h * scale))), scale

    def wrist(self, frame):
        """Paddle wrist as (x, y) in normalized *mirrored* coords, or None."""
        small, _ = self._small(frame)
        rgb = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
        ts = max(self._last_ts + 1, int(time.time() * 1000) - self.start_ms)
        self._last_ts = ts
        result = self.landmarker.detect_for_video(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb), ts)
        if not result.pose_landmarks:
            return None
        lm = result.pose_landmarks[0][self.wrist_index]
        if (lm.visibility or 0) < config.MIN_WRIST_VISIBILITY:
            return None
        return (1.0 - lm.x, lm.y)

    def detect_tags(self, frame):
        """List of (tag_id, corners in mirrored full-frame pixels)."""
        small, scale = self._small(frame)
        gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
        w = frame.shape[1]
        found = []
        for d in self.tags.detect(gray):
            corners = d.corners / scale
            corners[:, 0] = w - corners[:, 0]
            found.append((d.tag_id, corners))
        return found

    def close(self):
        self.landmarker.close()


class TagStabilizer:
    """A tag only 'fires' after it's been seen N frames in a row, and only
    once until it leaves the view -- so flashing past a card does nothing."""

    def __init__(self, frames=config.TAG_STABLE_FRAMES):
        self.frames = frames
        self.counts = {}
        self.fired = set()

    def update(self, ids):
        ids = set(ids)
        self.counts = {i: self.counts.get(i, 0) + 1 for i in ids}
        self.fired &= ids
        out = []
        for i, n in self.counts.items():
            if n >= self.frames and i not in self.fired:
                self.fired.add(i)
                out.append(i)
        return out

    def progress(self, tag_id):
        return min(1.0, self.counts.get(tag_id, 0) / self.frames)
