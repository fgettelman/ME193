"""
Start the campus navigator.

    python run.py                  # upload the app if it changed, then show the phone link
    python run.py --person fiona   # also set who's wearing it
    python run.py --host 10.247.137.130

1. Uploads unoq_app/ to the board if it changed since the last upload
   (deploy.py; about a minute, the sketch recompiles). Walks and models
   already on the board are kept.
2. Waits for the board's web page, and copies each person's step length from
   the step counter's 5 m walks (../step-counter/calibration.json) if the
   board doesn't have it yet.
3. Prints the link to open on your phone and opens it on the laptop.

The phone has to reach the board: both on Tufts_Robot. Run board_wifi.py once
per board so the board's Wi-Fi doesn't keep dropping (see README).
"""

import argparse
import time
import webbrowser

import common
import deploy


def wait_for_page(host, seconds=90):
    end = time.time() + seconds
    while time.time() < end:
        try:
            return common.get(host, "/api/state", timeout=3)
        except Exception:
            time.sleep(2)
    return None


def main():
    ap = argparse.ArgumentParser(description="Start the campus navigator")
    ap.add_argument("--host", default=deploy.BOARD_HOST)
    ap.add_argument("--person", help="who's wearing it (e.g. fiona)")
    ap.add_argument("--no-browser", action="store_true")
    args = ap.parse_args()

    print("--- Board ---")
    last = deploy.last_deployed()
    if last.get("fingerprint") == deploy.app_fingerprint() and last.get("host") == args.host:
        print("The board has the latest app.")
        if not wait_for_page(args.host, 8):
            deploy.start(args.host)
    else:
        print("Putting the latest app on the board (enter the board's password if asked).")
        if not deploy.deploy(args.host, args.person):
            raise SystemExit(f"Couldn't reach the board at {args.host}. Is it on, and on the same "
                             "network? (python run.py --host <IP> if its IP changed)")

    print("Waiting for the board's web page...")
    state = wait_for_page(args.host)
    if not state:
        raise SystemExit("The app started but the page isn't answering. Try: python deploy.py --logs")

    have = state.get("people", {})
    for name, step in common.step_lengths().items():
        if name not in have:
            common.post(args.host, "/api/person", {"name": name, "step_length_m": step})
            print(f"Step length for {name}: {step} m (from the step counter)")
    if args.person:
        step = have.get(args.person, {}).get("step_length_m") or common.step_lengths().get(args.person, 0.70)
        common.post(args.host, "/api/person", {"name": args.person, "step_length_m": step})
        print(f"Wearer: {args.person} ({step} m per step)")

    link = common.url(args.host, "/")
    print(f"\n--- Open on your phone: {link} ---")
    print("Walk = demonstrate a route · Guide = navigate (+ DAgger fixes) · Review = clean labels · Train")
    if not args.no_browser:
        webbrowser.open(link)


if __name__ == "__main__":
    main()
