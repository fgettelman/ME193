"""
Put the lock on the UNO Q and (re)start it over SSH, without Arduino App Lab
(same as step-counter's deploy.py).

    python deploy.py                # copy unoq_app/ (code + trained model + code.json), restart
    python deploy.py --logs         # follow the app's output (like App Lab's console)
    python deploy.py --peek         # copy the board camera's latest picture to results/peek.jpg
    python deploy.py --check-camera # does the board see the webcam?
    python deploy.py --start / --stop
    python deploy.py --host 10.247.137.130   # if the board's IP changed

SSH asks for the board's password once per command; `ssh-copy-id
arduino@<board IP>` once stops that. Uploading recompiles and flashes the
sketch, so it takes about a minute.
"""

import argparse
import hashlib
import io
import json
import os
import subprocess
import tarfile

HERE = os.path.dirname(os.path.abspath(__file__))
APP_DIR = os.path.join(HERE, "unoq_app")
BOARD_HOST = "10.247.137.128"      # the board's IP on Tufts_Robot
APP_ID = "user:vault-lock"
REMOTE_DIR = "~/ArduinoApps/vault-lock"
CONTAINER = "vault-lock-main-1"
# Earlier apps also use the microcontroller, so they're stopped first.
OTHER_APPS = ["user:step-counter", "user:gait-identifier"]
# What was last uploaded, so run.py only re-uploads when something changed.
DEPLOYED = os.path.join(HERE, "models", "deployed.json")
SKIP = ("__pycache__", "peek.jpg")


def app_tarball():
    """unoq_app/ as a .tar.gz in memory, without caches, macOS metadata or old peeks."""
    def keep(info):
        name = os.path.basename(info.name)
        if name in SKIP or name.startswith(".") and name != ".":
            return None
        info.uid = info.gid = 0
        info.uname = info.gname = ""
        return info

    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz", format=tarfile.USTAR_FORMAT) as tar:
        tar.add(APP_DIR, arcname=".", filter=keep)
    return buf.getvalue()


def ssh(host, command, stdin=None, capture=False):
    r = subprocess.run(["ssh", f"arduino@{host}", command], input=stdin,
                       stdout=subprocess.PIPE if capture else None)
    return r if capture else r.returncode


def app_fingerprint():
    """Hash of everything deploy uploads (code, model, code.json, sketch)."""
    h = hashlib.sha256()
    for root, dirs, files in sorted(os.walk(APP_DIR)):
        dirs[:] = sorted(d for d in dirs if d not in SKIP and not d.startswith("."))
        for f in sorted(files):
            if f not in SKIP and not f.startswith("."):
                path = os.path.join(root, f)
                h.update(os.path.relpath(path, APP_DIR).encode())
                with open(path, "rb") as fh:
                    h.update(fh.read())
    return h.hexdigest()


def last_deployed():
    try:
        with open(DEPLOYED) as f:
            return json.load(f)
    except (OSError, ValueError):
        return {}


def stop_others():
    return "".join(f"arduino-app-cli app stop {a} >/dev/null 2>&1; " for a in OTHER_APPS)


def deploy(host):
    model = os.path.join(APP_DIR, "python", "models", "fingernet.json")
    if not os.path.exists(model):
        raise SystemExit("No trained model (unoq_app/python/models/fingernet.json): run train.py first.")
    has_code = os.path.exists(os.path.join(APP_DIR, "python", "code.json"))
    print(f"Uploading unoq_app/ to {host} (code: {'yours' if has_code else 'default 1-3-5, run set_code.py'})")
    print("Then restarting the app; the sketch recompiles, so this takes about a minute.")
    remote = (f"mkdir -p {REMOTE_DIR} && tar -xzf - -C {REMOTE_DIR} 2>/dev/null"
              f" && {stop_others()}"
              f" arduino-app-cli app stop {APP_ID} >/dev/null 2>&1;"
              f" arduino-app-cli app start {APP_ID} 2>&1 | grep -iE 'error|started|fail'"
              f" && sleep 15 && docker logs --tail 8 {CONTAINER} 2>&1")
    if ssh(host, remote, stdin=app_tarball()) != 0:
        return False
    os.makedirs(os.path.dirname(DEPLOYED), exist_ok=True)
    with open(DEPLOYED, "w") as f:
        json.dump({"host": host, "fingerprint": app_fingerprint(), "time": __import__("time").ctime()}, f)
    return True


def main():
    ap = argparse.ArgumentParser(description="Deploy the finger lock to the UNO Q over SSH")
    ap.add_argument("--host", default=BOARD_HOST, help="board IP (default %(default)s)")
    ap.add_argument("--logs", action="store_true", help="follow the app's output")
    ap.add_argument("--peek", action="store_true", help="fetch the board camera's latest picture")
    ap.add_argument("--check-camera", action="store_true", help="list cameras on the board and in the app")
    ap.add_argument("--start", action="store_true", help="just start the app")
    ap.add_argument("--stop", action="store_true", help="stop the app")
    args = ap.parse_args()
    h = args.host

    if args.logs:
        print("Following the board's output (Ctrl+C to stop)...")
        try:
            ssh(h, f"docker logs -f --tail 30 {CONTAINER}")
        except KeyboardInterrupt:
            pass
    elif args.peek:
        r = ssh(h, f"cat {REMOTE_DIR}/python/peek.jpg", capture=True)
        if r.returncode or not r.stdout:
            raise SystemExit("No picture yet: is the app running and does it have a camera? (--logs)")
        out = os.path.join(HERE, "results", "peek.jpg")
        os.makedirs(os.path.dirname(out), exist_ok=True)
        with open(out, "wb") as f:
            f.write(r.stdout)
        print(f"Saved {os.path.relpath(out)}: the green-box square of this is what the net sees.")
    elif args.check_camera:
        ssh(h, "echo 'On the board:'; ls -l /dev/video* 2>&1 | head; "
               f"echo 'Inside the app:'; docker exec {CONTAINER} sh -c 'ls -l /dev/video* 2>&1 | head;"
               " python3 -c \"import cv2; print(\\\"opencv\\\", cv2.__version__)\" 2>&1'")
    elif args.stop:
        if ssh(h, f"arduino-app-cli app stop {APP_ID} 2>&1 | grep -E 'error:|stopped|fail'"):
            raise SystemExit("Couldn't stop the app. Is the board on, and is the IP right?")
    elif args.start:
        if ssh(h, f"{stop_others()} arduino-app-cli app start {APP_ID} 2>&1 | grep -iE 'error|started|fail'"):
            raise SystemExit("Couldn't start the app. Is the board on, and is the IP right?")
    else:
        if not deploy(h):
            raise SystemExit("Deploy failed (see above). Is the board on, and is the IP right?")
        print("\nDone. Watch it with:  python deploy.py --logs")


if __name__ == "__main__":
    main()
