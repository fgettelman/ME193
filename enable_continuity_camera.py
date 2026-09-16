# One-time setup so OpenCV can see your iPhone (Continuity Camera) on macOS.
#
# Why this is needed: macOS hides Continuity Camera devices from any app whose
# bundle does not declare NSCameraUseContinuityCameraDeviceType. Plain `python`
# re-execs itself into the Python framework's Python.app, and that bundle does
# not declare the key -- so cv2.VideoCapture only ever finds the FaceTime HD
# camera. This script adds the key to that Python.app's Info.plist.
#
# Run it with the SAME interpreter you run your vision code with, e.g.
#   /path/to/.venv/bin/python enable_continuity_camera.py
#   /path/to/.venv/bin/python enable_continuity_camera.py --undo

import os
import plistlib
import shutil
import subprocess
import sys

KEY = 'NSCameraUseContinuityCameraDeviceType'
PLIST = os.path.join(sys.base_prefix, 'Resources', 'Python.app', 'Contents', 'Info.plist')
BACKUP = PLIST + '.pre-continuity.bak'


def interpreter_still_works():
    """Make sure the patched interpreter can still launch (signature check)."""
    try:
        return subprocess.run(
            [sys.executable, '-c', 'print("ok")'],
            capture_output=True, timeout=30,
        ).returncode == 0
    except Exception:
        return False


def main():
    if not os.path.exists(PLIST):
        sys.exit(f'No Python.app Info.plist at {PLIST} -- this interpreter is not a framework build.')

    with open(PLIST, 'rb') as f:
        info = plistlib.load(f)

    if '--undo' in sys.argv:
        if os.path.exists(BACKUP):
            shutil.copy(BACKUP, PLIST)
            os.remove(BACKUP)
        else:
            info.pop(KEY, None)
            with open(PLIST, 'wb') as f:
                plistlib.dump(info, f)
        print(f'Removed {KEY} from {PLIST}')
        return

    if info.get(KEY):
        print(f'{KEY} is already set in {PLIST} -- nothing to do.')
        return

    if not os.access(PLIST, os.W_OK):
        sys.exit(f'{PLIST} is not writable by you. Re-run with sudo.')

    shutil.copy(PLIST, BACKUP)
    info[KEY] = True
    try:
        with open(PLIST, 'wb') as f:
            plistlib.dump(info, f)
        if not interpreter_still_works():
            raise RuntimeError('interpreter failed to start after patching')
    except Exception as exc:
        shutil.copy(BACKUP, PLIST)
        sys.exit(f'Patch failed ({exc}); restored the original Info.plist.')

    print(f'Added {KEY} to {PLIST}')
    print(f'Backup of the original: {BACKUP}')
    print('Now run: python list_cameras.py')


if __name__ == '__main__':
    main()
