"""Watch a trained Silly-Walk policy (or a dummy one). Same as mjlab's `play`, with our task registered.

    python play.py Silly-Walk --agent zero          # beams do nothing: checks the robot loads
    python play.py Silly-Walk --agent random        # random beam speeds
    python play.py Silly-Walk --checkpoint-file logs/rsl_rl/silly_walk/<run>/model_999.pt
"""
import sys

import walk_task  # noqa: F401  (registers "Silly-Walk")
from mjlab.scripts.play import main

if __name__ == "__main__":
  sys.exit(main())
