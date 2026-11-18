"""Train the Silly-Walk task. Same as mjlab's own `train`, but it registers our task first.

    python train.py Silly-Walk --env.scene.num-envs 4096
"""
import sys

import walk_task  # noqa: F401  (registers "Silly-Walk")
from mjlab.scripts.train import main

if __name__ == "__main__":
  sys.exit(main())
