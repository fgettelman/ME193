# Prints every camera OpenCV can open, with its name when possible, so you know
# which index to pass to cv2.VideoCapture().
#
# Names come from AVFoundation and are listed in the same order OpenCV indexes
# them. `pip install pyobjc-framework-AVFoundation` to get names; without it you
# still get resolutions.

import cv2

try:
    import AVFoundation as AV

    names = [d.localizedName() for d in
             (AV.AVCaptureDevice.devicesWithMediaType_(AV.AVMediaTypeVideo) or [])]
except ImportError:
    names = []
    print('(pip install pyobjc-framework-AVFoundation to see camera names)\n')

for index in range(5):
    cap = cv2.VideoCapture(index, cv2.CAP_AVFOUNDATION)
    if not cap.isOpened():
        cap.release()
        continue

    label = names[index] if index < len(names) else '?'
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
