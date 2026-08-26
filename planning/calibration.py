"""Base-offset calibration from collected touches.

Prior-aware SolverNewton multistart: the bases stand on a near-common plane
(the setup prior), so Gauss-Newton is seeded on that manifold and candidates
that leave it are rejected.
"""

import numpy as np

from sda_bfc import SolverNewton

TOUCH_LINK = 3
RADIUS_INDEX = 4
PRIOR_MAX_Z = 0.3
PRIOR_MAX_TILT_DEG = 15.0
STARTS = 500


def within_prior(x):
    tilt = np.degrees(np.arccos(np.clip(x[2, 2], -1.0, 1.0)))
    return abs(x[2, 3]) <= PRIOR_MAX_Z and tilt <= PRIOR_MAX_TILT_DEG


def yaw_only_start(rng):
    yaw = rng.uniform(0.0, 2.0 * np.pi)
    radius = rng.uniform(0.2, 1.1)
    bearing = rng.uniform(0.0, 2.0 * np.pi)
    x = np.eye(4)
    c, s = np.cos(yaw), np.sin(yaw)
    x[:3, :3] = [[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]]
    x[:3, 3] = [radius * np.cos(bearing), radius * np.sin(bearing), 0.0]
    return x


def calibrate(robot, touches, radius=None, seed=12345):
    """radius is the effective touch radius of the world the touches were
    felt in (e.g. oracle.touch_radius); defaults to the calibrated tube."""
    if radius is None:
        radius = robot.get_link_radius(RADIUS_INDEX)
    As = [robot.get_cylinder_transform(TOUCH_LINK, qa) for qa, _ in touches]
    Bs = [robot.get_cylinder_transform(TOUCH_LINK, qb) for _, qb in touches]
    solver = SolverNewton(As, Bs, radius, radius)
    rng = np.random.default_rng(seed)
    best_cost, best_x = np.inf, None
    for _ in range(STARTS):
        x = solver.solve(yaw_only_start(rng), 150)
        cost = solver.cost(x)
        if cost < best_cost and within_prior(x):
            best_cost, best_x = cost, x
    return best_x
