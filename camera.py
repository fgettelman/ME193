"""Open a camera by name so scripts don't have to hardcode an index.

The iPhone's index changes depending on whether it is connected when the
script starts, so `cv2.VideoCapture(1)` is fragile. Pass a name fragment
instead -- "iPhone" matches whatever your phone is called, "FaceTime"
matches the built-in camera.

Needs `pip install pyobjc-framework-AVFoundation` for name lookup; without
it, everything still works but you must pass an explicit index.

If your iPhone never shows up, run enable_continuity_camera.py once.
"""

import cv2


def list_cameras():
    """[(index, name)] of the video devices, in the order OpenCV indexes them.

    The sort matters: OpenCV's AVFoundation backend takes the same device
    list macOS hands us, then sorts it by uniqueID before indexing (it wants
    an order that doesn't shuffle between runs). AVFoundation's own order is
    built-in camera first, so on this Mac the two disagree -- the iPhone's
    uniqueID sorts ahead of the FaceTime HD camera's, putting the phone at
    OpenCV index 0 while AVFoundation lists it second. Enumerating in
    AVFoundation's order here would hand back an index that opens the other
    camera, which is exactly the silent wrong-camera bug this module exists
    to prevent.
    """
    try:
        import AVFoundation as AV
    except ImportError:
        return []

    devices = AV.AVCaptureDevice.devicesWithMediaType_(AV.AVMediaTypeVideo) or []
    devices = sorted(devices, key=lambda d: d.uniqueID())
    return [(i, d.localizedName()) for i, d in enumerate(devices)]


def find_index(name_fragment):
    """Index of the first camera whose name contains `name_fragment`, or None."""
    for index, name in list_cameras():
        if name_fragment.lower() in name.lower():
            return index
    return None


# Name fragments for the two cameras the scripts offer a choice between.
PHONE_NAME = 'iPhone'
LAPTOP_NAME = 'FaceTime'


def ask_for_camera(default='phone'):
    """Ask at the terminal which camera to use; return a name fragment.

    The answer goes straight to `open_camera(prefer=...)`, so the choice is
    still resolved by name -- this only decides *which* name to look for.
    Pressing Enter takes `default`. If nothing is on stdin (piped input, an
    IDE's run button with no console) it falls back to `default` rather than
    hanging on a prompt nobody can answer.
    """
    cameras = list_cameras()
    if not cameras:
        print('Warning: camera names unavailable (pip install '
              'pyobjc-framework-AVFoundation), so this choice cannot be '
              'resolved by name. Set CAMERA_INDEX instead.')

    def label(fragment):
        return next((n for _, n in cameras if fragment.lower() in n.lower()),
                    'not detected')

    default_key = '1' if default == 'phone' else '2'
    print('\nWhich camera should this use?')
    print(f'  1) phone  -- {label(PHONE_NAME)}')
    print(f'  2) laptop -- {label(LAPTOP_NAME)}')

    while True:
        try:
            answer = input(f'Choose 1 or 2 [{default_key}]: ').strip().lower()
        except EOFError:
            print(f'(nothing on stdin -- using the default: {default})')
            answer = ''
        answer = answer or default_key
        if answer in ('1', 'p', 'phone', 'iphone'):
            return PHONE_NAME
        if answer in ('2', 'l', 'laptop', 'mac', 'webcam', 'built-in'):
            return LAPTOP_NAME
        print('Please answer 1 (phone) or 2 (laptop).')


def open_camera(prefer='iPhone', index=None, width=None, height=None,
                required=False):
    """Open a camera and return the cv2.VideoCapture.

    `index`, when given, wins outright. Otherwise the first camera whose
    name contains `prefer` is used, falling back to index 0.

    Pass `required=True` when the wrong camera is worse than no camera --
    e.g. the tracker, where silently opening some other camera than the one
    asked for means driving the car off a view it isn't mounted on. It
    raises instead of falling back.

    Raises RuntimeError with the available cameras listed if it can't open.
    """
    cameras = list_cameras()

    if index is None:
        index = find_index(prefer)
        if index is None:
            if required:
                available = ', '.join(f'{i}: {n}' for i, n in cameras) or 'none found'
                raise RuntimeError(
                    f'No camera matching {prefer!r}. Available cameras: {available}. '
                    f'If you expected your iPhone: run enable_continuity_camera.py '
                    f'once, then check the phone-side checklist in iphone_video.py.')
            index = 0
            if cameras:
                print(f'No camera matching {prefer!r}; falling back to '
                      f'index 0 ({cameras[0][1]}).')

    cap = cv2.VideoCapture(index, cv2.CAP_AVFOUNDATION)
    if not cap.isOpened():
        available = ', '.join(f'{i}: {n}' for i, n in cameras) or 'none found'
        raise RuntimeError(
            f'Could not open camera index {index}. Available cameras: {available}')

    if width:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    if height:
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

    name = next((n for i, n in cameras if i == index), f'index {index}')
    print(f'Camera: {name}')
    return cap
