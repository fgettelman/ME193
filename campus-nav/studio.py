"""
The phone page on the laptop, without the board: review walks and train on
the copies in data/ (python sync.py first). Walk and Guide are greyed out.

    python studio.py            # http://localhost:8051
"""

import argparse
import time
import webbrowser

import common
from common import st
import server


def main():
    ap = argparse.ArgumentParser(description="Review and train on the laptop")
    ap.add_argument("--port", type=int, default=8051)
    args = ap.parse_args()
    app = server.App(st.Store(common.DATA), live=None, train_search="full")
    server.serve(app, args.port)
    link = f"http://localhost:{args.port}"
    print(f"Review/train page: {link}  (Ctrl+C to stop; python sync.py --push to send reviews back)")
    webbrowser.open(link)
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
