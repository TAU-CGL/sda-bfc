"""Belief-side collision model for planning under placement uncertainty.

R1 stands at qA; its capsules are expanded by the uncertainty range (C++
expandCapsule).  R2 believes its base is at X_initial and checks its own
capsules (transformed by X_initial) against the expanded hulls with a
conservative facet-plane distance, plus its own self-collision.  The ground
truth placement never enters here.
"""

import numpy as np
from scipy.spatial import ConvexHull

from sda_bfc import UncertaintyRanges, expand_capsule


def make_ranges(range_tuple):
    ranges = UncertaintyRanges()
    (ranges.x, ranges.y, ranges.z,
     ranges.roll, ranges.pitch, ranges.yaw) = range_tuple
    return ranges


class BeliefWorld:
    def __init__(self, scene_identity, X_initial, qA, range_tuple, margin=0.0):
        self.scene = scene_identity          # TwoArmScene(robot, I)
        self.R = X_initial[:3, :3]
        self.t = X_initial[:3, 3]
        self.margin = margin
        ranges = make_ranges(range_tuple)
        self.hull_equations = [
            ConvexHull(expand_capsule(capsule, ranges)).equations
            for capsule in scene_identity.capsules(qA)
        ]

    def config_free(self, q):
        if self.scene.min_self_clearance(q) <= 0.0:
            return False
        for capsule in self.scene.capsules(q):
            a = self.R @ np.asarray(capsule.a) + self.t
            b = self.R @ np.asarray(capsule.b) + self.t
            n = max(2, int(np.linalg.norm(b - a) / 0.02) + 2)
            pts = a[None, :] + np.linspace(0.0, 1.0, n)[:, None] * (b - a)[None, :]
            for equations in self.hull_equations:
                # max over facet planes: exact inside (negative), a lower
                # bound outside -> conservative collision test.
                sd = (pts @ equations[:, :3].T + equations[:, 3]).max(axis=1)
                if sd.min() < capsule.r + self.margin:
                    return False
        return True

    def edge_free(self, q0, q1, step=0.05):
        span = np.max(np.abs(q1 - q0))
        n = max(1, int(np.ceil(span / step)))
        for k in range(n + 1):
            if not self.config_free(q0 + (q1 - q0) * (k / n)):
                return False
        return True


class StaticObstacle:
    """Expanded convex outer bounds of a fixed set of capsules (world frame);
    conservative facet-plane clearance test against query capsules."""

    def __init__(self, capsules_world, range_tuple):
        ranges = make_ranges(range_tuple)
        self.hull_equations = [
            ConvexHull(expand_capsule(capsule, ranges)).equations
            for capsule in capsules_world
        ]

    def capsule_free(self, a, b, r, margin=0.0):
        n = max(2, int(np.linalg.norm(b - a) / 0.02) + 2)
        pts = a[None, :] + np.linspace(0.0, 1.0, n)[:, None] * (b - a)[None, :]
        for equations in self.hull_equations:
            sd = (pts @ equations[:, :3].T + equations[:, 3]).max(axis=1)
            if sd.min() < r + margin:
                return False
        return True
