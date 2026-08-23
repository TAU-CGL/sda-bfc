"""Capsule-based collision model for a pair of UR5e arms.

Each link cylinder is a finite capsule (segment + radius) with DH-derived
extents; distances are exact segment-segment distances, unlike the solver's
infinite-line model.  Used both as the dance's simulated contact sensor and
to verify that recorded configurations are collision-free.
"""

import numpy as np

from sda_bfc import UR5e

robot = UR5e()

NUM_LINKS = 6
UR5E_D = [0.1625, 0.0, 0.0, 0.1333, 0.0997, 0.0996]
UR5E_A = [0.0, -0.425, -0.3922, 0.0, 0.0, 0.0]
DEFAULT_HOUSING_LENGTH = 0.12


def link_extents():
    extents = []
    for i in range(NUM_LINKS):
        a_in = UR5E_A[i - 1] if i > 0 else 0.0
        d_in = UR5E_D[i - 1] if i > 0 else UR5E_D[0]
        if abs(a_in) > 1e-6:
            extents.append((min(0.0, -a_in), max(0.0, -a_in)))
        else:
            length = abs(d_in) if abs(d_in) > 1e-6 else DEFAULT_HOUSING_LENGTH
            extents.append((-length / 2, length / 2))
    return extents


LINK_EXTENTS = link_extents()
# DH link i maps to radii index i for the base links but i+1 from the forearm
# on (the convention established by the touch-pose data: DH 3 <-> radii 4).
LINK_RADII = [robot.get_link_radius(i) for i in [0, 1, 2, 4, 5, 6]]


def segment_closest(p1, q1, p2, q2):
    d1, d2, r = q1 - p1, q2 - p2, p1 - p2
    a, e = d1 @ d1, d2 @ d2
    b, c, f = d1 @ d2, d1 @ r, d2 @ r
    denom = a * e - b * b
    s = np.clip((b * f - c * e) / denom, 0.0, 1.0) if denom > 1e-14 else 0.0
    t = (b * s + f) / e if e > 1e-14 else 0.0
    if t < 0.0:
        t, s = 0.0, (np.clip(-c / a, 0.0, 1.0) if a > 1e-14 else 0.0)
    elif t > 1.0:
        t, s = 1.0, (np.clip((b - c) / a, 0.0, 1.0) if a > 1e-14 else 0.0)
    return np.linalg.norm((p1 + s * d1) - (p2 + t * d2)), s, t


def segment_distance(p1, q1, p2, q2):
    return segment_closest(p1, q1, p2, q2)[0]


def capsules(q, base=None):
    out = []
    for i in range(NUM_LINKS):
        T = robot.get_cylinder_transform(i, q)
        if base is not None:
            T = base @ T
        p, u = T[:3, 3], T[:3, 2]
        t0, t1 = LINK_EXTENTS[i]
        out.append((p + t0 * u, p + t1 * u, LINK_RADII[i]))
    return out


class TwoArmSimulator:
    """Arm A at the origin, arm B at base transform X."""

    def __init__(self, X):
        self.X = X

    def self_clearances(self, q):
        caps = capsules(q)
        out = {}
        for i in range(NUM_LINKS):
            for j in range(i + 2, NUM_LINKS):
                p1, q1, r1 = caps[i]
                p2, q2, r2 = caps[j]
                out[(i, j)] = segment_distance(p1, q1, p2, q2) - (r1 + r2)
        return out

    def cross_clearances(self, qA, qB):
        capsA = capsules(qA)
        capsB = capsules(qB, self.X)
        out = {}
        for i, (p1, q1, r1) in enumerate(capsA):
            for j, (p2, q2, r2) in enumerate(capsB):
                out[(i, j)] = segment_distance(p1, q1, p2, q2) - (r1 + r2)
        return out

    def first_cross_contact(self, qA, qB):
        clearances = self.cross_clearances(qA, qB)
        pair = min(clearances, key=clearances.get)
        return clearances[pair], pair

    def contact_interior(self, qA, qB, pair, margin=0.03):
        """True if the closest points of the contacting pair lie strictly
        inside both segments -- a side-to-side touch, where segment distance
        equals infinite-line distance (the solver's model)."""
        i, j = pair
        p1, q1, _ = capsules(qA)[i]
        p2, q2, _ = capsules(qB, self.X)[j]
        _, s, t = segment_closest(p1, q1, p2, q2)
        return margin < s < 1.0 - margin and margin < t < 1.0 - margin

    def validate_config(self, qA, qB, touch_pair):
        issues = []
        for arm, q in [("A", qA), ("B", qB)]:
            for pair, clearance in self.self_clearances(q).items():
                if clearance < 0.0:
                    issues.append((f"self-{arm}", pair, clearance))
        touch_clearance = None
        for pair, clearance in self.cross_clearances(qA, qB).items():
            if pair == touch_pair:
                touch_clearance = clearance
            elif clearance < 0.0:
                issues.append(("cross", pair, clearance))
        return issues, touch_clearance


if __name__ == "__main__":
    import blind_touch_dance as dance

    for seed in range(5):
        rng = np.random.default_rng(seed)
        X_true = dance.sample_ground_truth(rng)
        sim = TwoArmSimulator(X_true)
        touches = dance.perform_dance(X_true, verbose=False)
        issues, gaps = [], []
        for qA, qB in touches:
            config_issues, touch_gap = sim.validate_config(qA, qB, dance.TOUCH_PAIR)
            issues.extend(config_issues)
            gaps.append(touch_gap)
        print(f"seed {seed}: {len(touches)} touches | collision issues: {len(issues)} | "
              f"touch clearance [{min(gaps)*1e6:.2f}, {max(gaps)*1e6:.2f}] um")
        for kind, pair, clearance in issues[:5]:
            print(f"    {kind} {pair}: {clearance*1000:.2f} mm")
