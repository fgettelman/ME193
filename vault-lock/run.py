"""
Run the vault lock: one command for everything.

    python run.py              # the lock on the UNO Q + the live view on the laptop
    python run.py --laptop     # the whole lock on the laptop instead (webcam plugged into the laptop)
    python run.py --deploy     # upload to the board even if nothing changed

What it does (UNO Q mode):
  1. checks the laptop can reach the board (same Wi-Fi: Tufts_Robot) over SSH
  2. uploads unoq_app/ if anything changed since the last upload (code, model,
     your code.json, the sketch); otherwise just makes sure the app is running
  3. waits until the board's camera is up
  4. opens board_view.py: the board's camera, instructions, combination boxes,
     LEDs, servo and the button's wiring, live on the laptop (q closes the
     window; the lock keeps running on the board)
"""

import argparse
import json
import subprocess
import sys
import time
import urllib.request

import deploy


def say(text):
    print(f"\n==> {text}", flush=True)


def ssh_ok(host):
    r = subprocess.run(["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=6", f"arduino@{host}", "true"],
                       capture_output=True, text=True)
    return r.returncode == 0, r.stderr.strip()


def wifi_name():
    r = subprocess.run(["networksetup", "-getairportnetwork", "en0"], capture_output=True, text=True)
    return r.stdout.strip().split(": ", 1)[-1]


def app_running(host):
    r = subprocess.run(["ssh", "-o", "BatchMode=yes", f"arduino@{host}",
                        f"docker ps --format '{{{{.Names}}}}' | grep -x {deploy.CONTAINER}"],
                       capture_output=True, text=True)
    return r.returncode == 0


def board_status(host):
    try:
        with urllib.request.urlopen(f"http://{host}:8040/status.json", timeout=2) as r:
            return json.loads(r.read())
    except Exception:
        return None


def main():
    ap = argparse.ArgumentParser(description="Run the vault lock")
    ap.add_argument("--laptop", action="store_true", help="run the lock on the laptop (webcam on the laptop)")
    ap.add_argument("--deploy", action="store_true", help="upload to the board even if nothing changed")
    ap.add_argument("--host", default=deploy.BOARD_HOST)
    ap.add_argument("--camera", type=int, default=0, help="laptop mode: camera number (default 0)")
    args = ap.parse_args()

    if args.laptop:
        say(f"Running the lock on the laptop with camera {args.camera} (SPACE = button, q = quit)")
        sys.argv = ["laptop_lock.py", "--camera", str(args.camera)]
        import laptop_lock
        laptop_lock.main()
        return

    h = args.host
    say(f"1/4  Checking the board at {h}")
    ok, err = ssh_ok(h)
    if not ok:
        print(f"Can't reach the board: {err or 'no answer'}")
        print(f"  - The laptop is on Wi-Fi '{wifi_name()}': it must be on Tufts_Robot, like the board.")
        print("  - Is the board powered and booted (about 1 minute after plugging in)?")
        print("  - If its IP changed: python run.py --host <new IP>  (App Lab shows the IP)")
        if "Permission denied" in err:
            print("  - The SSH key isn't on the board: ssh-copy-id arduino@" + h)
        raise SystemExit(1)
    print("Board reachable.")

    changed = deploy.last_deployed().get("fingerprint") != deploy.app_fingerprint()
    if args.deploy or changed:
        say("2/4  Uploading to the board (something changed since last time; ~1 minute)")
        if not deploy.deploy(h):
            raise SystemExit("Upload failed (see above).")
    elif app_running(h):
        say("2/4  Board already has the latest version, and the lock is running")
    else:
        say("2/4  Board already has the latest version; starting the lock")
        deploy.ssh(h, f"{deploy.stop_others()} arduino-app-cli app start {deploy.APP_ID} 2>&1 | grep -iE 'error|started|fail'")

    say("3/4  Waiting for the board's camera")
    restarted = False
    for i in range(90):
        st = board_status(h)
        if st and st.get("mode") == "running":
            b = st.get("button", {})
            print(f"Lock running at {st.get('fps') or '~15'} frames/s, camera: {st.get('camera')}")
            print(f"Button D2 reads {'PRESSED' if b.get('pressed_now') else 'released'}, "
                  f"{b.get('presses_since_power_on')} presses counted")
            break
        if i == 20 and not restarted and not app_running(h):
            # Right after a fresh flash the app sometimes exits once; start it again.
            print("The app stopped; starting it again...")
            subprocess.run(["ssh", f"arduino@{h}", f"arduino-app-cli app start {deploy.APP_ID} >/dev/null 2>&1"])
            restarted = True
        time.sleep(1)
    else:
        print("The lock didn't come up. Its output:")
        deploy.ssh(h, f"docker logs --tail 20 {deploy.CONTAINER} 2>&1")
        raise SystemExit("Check that the webcam is on the board's USB-C hub, then try again.")

    say("4/4  Opening the live view (SPACE = button, h = hide numbers, q = close the window)")
    sys.argv = ["board_view.py", "--host", h]
    import board_view
    board_view.main()
    print("\nThe window is closed; the lock keeps running on the board. "
          "Stop it with: python deploy.py --stop")


if __name__ == "__main__":
    main()
