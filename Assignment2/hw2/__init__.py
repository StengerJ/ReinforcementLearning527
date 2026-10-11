"""HW2 code.  Puts the professor-supplied ``starter/`` folder on ``sys.path`` so
``from furuta_env import FurutaEnv, discretize_state`` resolves to the
unmodified supplied file."""

import sys
from pathlib import Path

STARTER_DIR = Path(__file__).resolve().parent.parent / "starter"
if str(STARTER_DIR) not in sys.path:
    sys.path.insert(0, str(STARTER_DIR))
