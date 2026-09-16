# Prints every camera OpenCV can open, with its name and resolution, so you
# know what to put in CAMERA_NAME (or CAMERA_INDEX) in your scripts.
#
# If your iPhone is missing, run enable_continuity_camera.py once, then check
# the phone-side checklist in iphone_video.py.

import cv2

from camera import list_cameras

names = dict(list_cameras())
if not names:
    print('(pip install pyobjc-framework-AVFoundation to see camera names)\n')

for index in range(5):
    cap = cv2.VideoCapture(index, cv2.CAP_AVFOUNDATION)
    if not cap.isOpened():
        cap.release()
        continue

    label = names.get(index, '?')
    ok, frame = cap.read()
    if ok:
        print(f'index {index}: {label} -- {frame.shape[1]}x{frame.shape[0]}, '
              f'mean brightness {frame.mean():.1f}')
    else:
        print(f'index {index}: {label} -- opened but returned no frame')
    cap.release()

if len(names) < 2:
    print('\nOnly one camera found. If your iPhone is missing, see the checklist '
          'in iphone_video.py.')
