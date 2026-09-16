# Read video from the iPhone (Continuity Camera) with OpenCV.
#
# Setup, once:
#   1. python enable_continuity_camera.py   (lets Python see Continuity cameras)
#   2. iPhone: Settings > General > AirPlay & Continuity > Continuity Camera ON
#   3. Same Apple ID on both devices, Wi-Fi and Bluetooth on, phone near the Mac
#   4. Phone locked, screen off, held still with the rear camera pointed at the
#      scene -- iOS only offers the camera when the phone is stationary
#   5. python list_cameras.py  -> note the index next to your phone's name
#
# Then set CAMERA_INDEX below and run this file.

import cv2

CAMERA_INDEX = 0  # 0 is usually the built-in FaceTime HD camera


def open_camera(index=CAMERA_INDEX):
    """Open a camera on macOS and fail loudly if it isn't there."""
    cap = cv2.VideoCapture(index, cv2.CAP_AVFOUNDATION)
    if not cap.isOpened():
        raise RuntimeError(
            f'Could not open camera index {index}. Run list_cameras.py to see '
            'what OpenCV can actually find.')
    return cap


def main():
    cap = open_camera()
    print('Press q to quit.')
    while True:
        ok, frame = cap.read()
        if not ok:
            print('Dropped frame / camera disconnected.')
            break

        cv2.imshow('iPhone', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
