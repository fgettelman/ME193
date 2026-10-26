"""
Put the app on the UNO Q and (re)start it over SSH, without Arduino App Lab.

    python deploy.py           # copy unoq_app/ (code + trained models), restart
    python deploy.py --logs    # follow the app's output (like App Lab's console)
    python deploy.py --start   # just start the app (e.g. after the board rebooted)
    python deploy.py --host 10.247.137.130   # if the board's IP changed

App Lab finds boards by mDNS, which Tufts Wi-Fi blocks, but SSH to the board's
IP works. SSH asks for the board's password once per command. To skip that,
run `ssh-copy-id arduino@<board IP>` once.

Uploading recompiles and flashes the sketch, so it takes about a minute.
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
APP_ID = "user:gait-identifier"
REMOTE_DIR = "~/ArduinoApps/gait-identifier"
CONTAINER = "gait-identifier-main-1"
# What was last uploaded, so run.py can skip uploading the same models again.
DEPLOYED = os.path.join(HERE, "models", "deployed.json")


def app_tarball():
    """unoq_app/ as a .tar.gz in memory, without caches or macOS metadata."""
    def keep(info):
        name = os.path.basename(info.name)
        if name == "__pycache__" or name.startswith(".") and name != ".":
            return None
        info.uid = info.gid = 0
        info.uname = info.gname = ""
        return info

    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz", format=tarfile.USTAR_FORMAT) as tar:
        tar.add(APP_DIR, arcname=".", filter=keep)
    return buf.getvalue()


def ssh(host, command, stdin=None):
    return subprocess.run(["ssh", f"arduino@{host}", command], input=stdin).returncode


def app_fingerprint():
    """Hash of everything deploy uploads; changes when code or models change."""
    h = hashlib.sha256()
    for root, dirs, files in sorted(os.walk(APP_DIR)):
        dirs[:] = sorted(d for d in dirs if d != "__pycache__" and not d.startswith("."))
        for f in sorted(files):
            if not f.startswith("."):
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


def start(host):
    """Start the app that's already on the board."""
    print(f"Starting the app on {host}...")
    return ssh(host, f"arduino-app-cli app start {APP_ID} 2>&1 | grep -E 'error:|started|fail'") == 0


def deploy(host, person=None):
    """Upload unoq_app/ and restart the app. Returns True if it worked."""
    models = sorted(f for f in os.listdir(os.path.join(APP_DIR, "python", "models"))
                    if f.endswith(".json"))
    print(f"Uploading unoq_app/ to {host} (models: {', '.join(models) or 'none yet'})")
    print("Then restarting the app; the sketch recompiles, so this takes about a minute.")
    remote = (f"mkdir -p {REMOTE_DIR} && tar -xzf - -C {REMOTE_DIR} 2>/dev/null"
              f" && arduino-app-cli app stop {APP_ID} >/dev/null 2>&1;"
              f" arduino-app-cli app start {APP_ID} 2>&1 | grep -E 'error:|started|fail'"
              f" && sleep 15 && docker logs --tail 5 {CONTAINER} 2>&1")
    if ssh(host, remote, stdin=app_tarball()):
        return False
    os.makedirs(os.path.dirname(DEPLOYED), exist_ok=True)
    with open(DEPLOYED, "w") as f:
        json.dump({"host": host, "person": person, "fingerprint": app_fingerprint()}, f)
    return True


def main():
    ap = argparse.ArgumentParser(description="Deploy the gait app to the UNO Q over SSH")
    ap.add_argument("--host", default=BOARD_HOST, help="board IP (default %(default)s)")
    ap.add_argument("--logs", action="store_true", help="just follow the app's output")
    ap.add_argument("--start", action="store_true", help="just start the app")
    args = ap.parse_args()

    if args.logs:
        print("Following the board's output (Ctrl+C to stop)...")
        try:
            ssh(args.host, f"docker logs -f --tail 30 {CONTAINER}")
        except KeyboardInterrupt:
            pass
        return

    if args.start:
        if not start(args.host):
            raise SystemExit("Couldn't start the app. Is the board on, and is the IP right?")
        return
    if not deploy(args.host):
        raise SystemExit("Deploy failed (see above). Is the board on, and is the IP right?")
    print("\nDone. Watch it with:  python live_view.py")


if __name__ == "__main__":
    main()
