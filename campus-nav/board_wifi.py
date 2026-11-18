"""
Keep the UNO Q steadily connected to Tufts_Robot while you walk around.

    python board_wifi.py            # apply the settings below, show the connection
    python board_wifi.py --check    # just show the connection and recent drops
    python board_wifi.py --undo     # back to the defaults

Out of the box the board kept dropping off Tufts_Robot about every 2 minutes
(34 times in 1.5 h): it lost contact with a 5 GHz access point and hopped to
another one, and the phone page froze each time. Two settings fix that:

  - 2.4 GHz only: longer range and less blocked by your body (the board is on
    your thigh), and no forced channel changes (many 5 GHz channels must be
    vacated when an access point detects radar, which drops everyone)
  - Wi-Fi power saving off: the radio doesn't doze and miss the access point

The board only sends a few KB a second, so 2.4 GHz is plenty fast.
"""

import argparse
import subprocess

import deploy

CONN = "Tufts_Robot"

SHOW = (f"echo; echo 'Connection:'; nmcli -t -f IN-USE,BSSID,CHAN,SIGNAL dev wifi list | grep '^\\*' "
        f"| awk -F: '{{print \"  channel \" $(NF-1) \", signal \" $NF \"/100\"}}';"
        f" echo \"  band: $(nmcli -g 802-11-wireless.band connection show {CONN}),"
        f" power saving: $(nmcli -g 802-11-wireless.powersave connection show {CONN})\";"
        f" echo \"  Wi-Fi drops in the last hour: $(journalctl -u wpa_supplicant --since '-1 hour'"
        f" --no-pager | grep -c CTRL-EVENT-DISCONNECTED)\"")


def main():
    ap = argparse.ArgumentParser(description="Steady Tufts_Robot Wi-Fi on the board")
    ap.add_argument("--host", default=deploy.BOARD_HOST)
    g = ap.add_mutually_exclusive_group()
    g.add_argument("--check", action="store_true")
    g.add_argument("--undo", action="store_true")
    args = ap.parse_args()

    if args.check:
        cmd = SHOW
    else:
        band, ps = ("''", "0") if args.undo else ("bg", "2")
        print("Applying the settings; the board reconnects (SSH drops for a few seconds)...")
        # Reconnect in the background so it survives this SSH session dropping.
        cmd = (f"nmcli connection modify {CONN} 802-11-wireless.band {band} "
               f"802-11-wireless.powersave {ps} && "
               f"nohup sh -c 'sleep 1; nmcli connection up {CONN}' >/tmp/wifi.log 2>&1 &")
    subprocess.run(["ssh", f"arduino@{args.host}", cmd])
    if not args.check:
        import time
        time.sleep(15)
        subprocess.run(["ssh", "-o", "ConnectTimeout=20", f"arduino@{args.host}", SHOW])


if __name__ == "__main__":
    main()
