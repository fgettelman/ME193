"""
Generate printable AprilTags (tag36h11, same family as config.TAG_FAMILY).

Tags 1, 2, 3 = Easy / Medium / Hard; showing one also starts the game. Writes one PNG per tag
plus a single sheet with all of them into ./tags/. Print at 100% scale.

Usage:
    python make_tags.py              # tags 1, 2, 3
"""

import os
import sys

import cv2
import numpy as np
from pupil_apriltags import Detector

import config

TAG_PX = 600          # size of the black/white tag itself
BORDER_PX = 100       # white quiet zone around the tag (needed for detection)
LABEL_PX = 90         # strip under the tag for the name
HERE = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(HERE, "tags")


def tag_name(tag_id):
    return config.LEVELS.get(tag_id, ("?",))[0]


def make_card(dictionary, tag_id):
    tag = cv2.aruco.generateImageMarker(dictionary, tag_id, TAG_PX)
    card = np.full((TAG_PX + 2 * BORDER_PX + LABEL_PX, TAG_PX + 2 * BORDER_PX), 255, np.uint8)
    card[BORDER_PX:BORDER_PX + TAG_PX, BORDER_PX:BORDER_PX + TAG_PX] = tag
    label = f"{tag_id}: {tag_name(tag_id)}"
    font = cv2.FONT_HERSHEY_SIMPLEX
    (tw, _), _ = cv2.getTextSize(label, font, 1.6, 3)
    cv2.putText(card, label, ((card.shape[1] - tw) // 2, card.shape[0] - LABEL_PX // 2),
                font, 1.6, 0, 3, cv2.LINE_AA)
    return card


def main():
    ids = [int(a) for a in sys.argv[1:]] or [1, 2, 3]
    os.makedirs(OUT_DIR, exist_ok=True)
    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36h11)
    detector = Detector(families=config.TAG_FAMILY)

    cards = []
    for tag_id in ids:
        card = make_card(dictionary, tag_id)
        # Sanity check: the game's detector must read back the right id.
        found = [d.tag_id for d in detector.detect(card)]
        if found != [tag_id]:
            raise RuntimeError(f"tag {tag_id} did not round-trip (detected {found})")
        path = os.path.join(OUT_DIR, f"tag_{tag_id}.png")
        cv2.imwrite(path, card)
        print(f"wrote {path}")
        cards.append(card)

    sheet = os.path.join(OUT_DIR, "tags_sheet.png")
    cv2.imwrite(sheet, np.hstack(cards))
    print(f"wrote {sheet}")


if __name__ == "__main__":
    main()
