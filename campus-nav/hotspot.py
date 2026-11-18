"""
Make the UNO Q its own Wi-Fi hotspot, so your phone can reach the page
anywhere on campus (Tufts_Robot only works in some buildings). The Wi-Fi is
only the link between phone and board; nothing uses it to work out where you
are.

    python hotspot.py --on              # board becomes Wi-Fi "CampusNav" for 90 min
    python hotspot.py --on --minutes 30
    python hotspot.py --off             # back to Tufts_Robot now

With --on, the board drops off Tufts_Robot (one radio), so this SSH session
ends. Join "CampusNav" on your phone (password below) and open
http://10.42.0.1:8050. After --minutes it switches back to Tufts_Robot by
itself, so you can't get locked out. To end it early: join CampusNav on the
laptop and run  python hotspot.py --off --host 10.42.0.1

Uses NetworkManager (nmcli) on the board. Not tested on our board yet: if
`nmcli` asks for permission, run the printed command with sudo over SSH.
"""

import argparse
import subprocess

import deploy

SSID = "CampusNav"
PASSWORD = "nolop2026"          # 8+ characters; change it if you like
HOME_WIFI = "Tufts_Robot"


def main():
    ap = argparse.ArgumentParser(description="Board hotspot on/off")
    ap.add_argument("--host", default=deploy.BOARD_HOST)
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--on", action="store_true")
    g.add_argument("--off", action="store_true")
    ap.add_argument("--minutes", type=int, default=90)
    args = ap.parse_args()

    back = f"nmcli connection down Hotspot; nmcli connection up id '{HOME_WIFI}'"
    if args.on:
        cmd = (f"nohup sh -c \"sleep 2; nmcli device wifi hotspot ifname wlan0 ssid {SSID} "
               f"password {PASSWORD}; sleep {args.minutes * 60}; {back}\" >/tmp/hotspot.log 2>&1 &")
        print(f"Switching the board to hotspot '{SSID}' (password {PASSWORD}) for {args.minutes} min.")
        print(f"Phone: join {SSID}, open http://10.42.0.1:8050")
    else:
        cmd = f"nohup sh -c \"sleep 2; {back}\" >/tmp/hotspot.log 2>&1 &"
        print(f"Switching the board back to {HOME_WIFI}.")
    print(f"(on the board: {cmd})")
    subprocess.run(["ssh", f"arduino@{args.host}", cmd])


if __name__ == "__main__":
    main()
