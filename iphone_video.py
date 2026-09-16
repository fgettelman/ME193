# Standalone check that the iPhone feed works before you wire it into
# apriltag_tracker.py.
#
# Setup, once:
#   1. python enable_continuity_camera.py   (lets Python see Continuity cameras)
#   2. iPhone: Settings > General > AirPlay & Continuity > Continuity Camera ON
#   3. Same Apple ID on both devices, Wi-Fi and Bluetooth on, phone near the Mac
#   4. Phone locked, screen off, held still with the rear camera pointed at the
#      scene -- iOS only offers the camera when the phone is stationary
#   5. python list_cameras.py  -> confirm the phone is listed
#
# Then just run this file.

import cv2

from camera import open_camera


def main():
    cap = open_camera(prefer='iPhone')
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
