"""
Set the safe's code: three numbers from 1 to 5, no two in a row the same
(3 2 3 is fine, 3 3 2 isn't: holding still mustn't enter a number twice).

    python set_code.py 4 1 3

Saves unoq_app/python/code.json (a salted hash, not the numbers; Git ignores
it). laptop_lock.py uses it straight away; run deploy.py to send it to the
board. Until you set one, the code is 1 3 5.
"""

import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "unoq_app", "python"))

import lock  # noqa: E402

if len(sys.argv) != 4:
    raise SystemExit(__doc__)
try:
    lock.make_code_file(sys.argv[1:])
except ValueError as e:
    raise SystemExit(f"Not saved: {e}.")
check, _ = lock.load_checker()
assert check(tuple(int(c) for c in sys.argv[1:]))
print(f"Code saved to {os.path.relpath(lock.CODE_PATH)}. Run deploy.py to put it on the board.")
