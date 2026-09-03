"""Cell constants shared by robot.py / maneuver.py / rrt.py."""

import os
from pathlib import Path

import numpy as np

# Vendored menagerie UR5e MJCF: the source of the kinematic chain (fixed joint
# frames, axes, limits) that the original read off a URDF.  Resolved from the
# env var, a source checkout, or the working directory -- the installed
# package does not carry the assets.
_CANDIDATES = [
    Path(__file__).resolve().parent.parent.parent / "assets",
    Path.cwd() / "assets",
]
ASSETS = next((c for c in _CANDIDATES if c.is_dir()), _CANDIDATES[-1])
MJCF_PATH = Path(os.environ.get(
    "SDA_BFC_UR5E_MJCF", ASSETS / "universal_robots_ur5e" / "ur5e.xml"))

# Believed placement of arm B's base: (xyz, rotation vector).  Matches
# tests/test_touch_poses.BASE_OFFSET.
BASE_OFFSET = np.array([-0.24, 0.73, -0.25, 0.0, 0.0, 0.0])

FOREARM_LINK = "forearm_link"
# Contact radius of the forearm tube.  The library's FK table value
# (0.235 m circumference); the source repo's hardware measurement was 0.0378.
FOREARM_RADIUS = 0.235 / (2.0 * np.pi)

# Park pose while the other arm moves.
HOME_Q = np.array([0.0, -np.pi / 2, 0.0, -np.pi / 2, 0.0, 0.0])
