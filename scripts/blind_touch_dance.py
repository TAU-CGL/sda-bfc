"""Blind touch "dance" between two UR5e arms, physically executable.

Neither arm knows where the other is, and -- unlike a naive simulation --
neither arm may ever pass through the other.  Every motion in this script is
a GUARDED SEGMENT: one arm moves at a time along a straight joint-space line,
stops at the first felt contact (of ANY link pair, as a real force-sensing
arm would), and retreats only along the path it just traversed, which is
free by construction.  Placements, sweeps, posture changes, and returns to
home are all such segments, so the executed trajectory is continuous and
collision-free end to end.

Geometry ("blue upright, sweep orange"): a horizontal forearm can reach at
most z ~ 0.59 (shoulder 0.16 + upper arm 0.43 fully vertical), while a fully
upright arm's forearm STARTS at 0.59 -- so the static blue arm stands
upright-but-bent: elbow at z ~ 0.39, forearm continuing up at a slight
outward tilt (0.2-0.4 rad), spanning z ~ 0.39-0.77.  The orange arm sweeps
its horizontal forearm "finger" (radii ~ 0.13-0.66 from its base) at
altitudes 0.50-0.57: above blue's elbow, below blue's wrists, so the correct
link is what it meets.  Blue's deliberate forearm tilt also makes every
contact distance z-sensitive, which is what lets the solver observe the
z / roll / pitch of the base offset.

A contact is recorded for calibration only when the touching pair is
forearm-forearm with the contact interior to both cylinders (a real system
reads this off the joint-torque signature); other contacts just stop the
sweep and are retreated from.  Only the simulated contact sensor consults
the ground truth; the dance logic itself is blind.
"""

import numpy as np

from arm_simulator import TwoArmSimulator
from sda_bfc import SolverNewton, UR5e

robot = UR5e()
DH_LINK, RADII_LINK = 3, 4
R_LINK = robot.get_link_radius(RADII_LINK)
TOUCH_PAIR = (DH_LINK, DH_LINK)

A_DIRECTIONS = np.linspace(0.0, 2.0 * np.pi, 12, endpoint=False)
BLUE_SHOULDER = -0.55
BLUE_TILTS = [0.2, 0.4]
ORANGE_RUNGS = [-1.25, -1.0, -0.9]
ORANGE_TILTS = [-0.2, 0.2]
ORANGE_DEPLOY_YAWS = [0.0, 2.0 * np.pi / 3.0, -2.0 * np.pi / 3.0]
STEP = 0.02
SWEEP_MARGIN = 0.05
TARGET_TOUCHES = 12
MIN_TOUCHES = 6
MIN_TOUCH_DIRECTIONS = 3
MIN_TOUCH_TILTS = 2


def fold_config(yaw):
    return np.array([yaw, -np.pi / 2, 0.0, 0.0, 0.0, 0.0])


def blue_config(yaw, tilt):
    return np.array([yaw, BLUE_SHOULDER, -np.pi / 2 - BLUE_SHOULDER + tilt,
                     0.0, 0.0, 0.0])


def orange_config(yaw, rung, tilt=0.0):
    return np.array([yaw, rung, -rung + tilt, 0.0, 0.0, 0.0])


def forearm(q):
    return robot.get_cylinder_transform(DH_LINK, q)


def sample_ground_truth(rng):
    distance = rng.uniform(0.5, 0.78)
    bearing = rng.uniform(0.0, 2.0 * np.pi)
    yaw = rng.uniform(0.0, 2.0 * np.pi)
    roll, pitch = np.radians(rng.uniform(-2.0, 2.0, size=2))
    cz, sz = np.cos(yaw), np.sin(yaw)
    cy, sy = np.cos(pitch), np.sin(pitch)
    cx, sx = np.cos(roll), np.sin(roll)
    Rz = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]])
    Ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
    Rx = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]])
    X = np.eye(4)
    X[:3, :3] = Rz @ Ry @ Rx
    X[:3, 3] = [distance * np.cos(bearing), distance * np.sin(bearing),
                rng.uniform(-0.05, 0.05)]
    return X


class MotionExecutor:
    """Executes guarded joint-space segments; one arm moves at a time.

    Every step is checked for cross-arm and self contact; a segment stops at
    the first contact (bisected onto the surface) and reports it.  The frame
    stream this emits is therefore a continuous, collision-free trajectory.
    """

    def __init__(self, sim, recorder=None):
        self.sim = sim
        self.recorder = recorder
        self.q = {"A": fold_config(0.0), "B": fold_config(0.0)}

    def _clearance(self, arm, q):
        qA = q if arm == "A" else self.q["A"]
        qB = q if arm == "B" else self.q["B"]
        cross, pair = self.sim.first_cross_contact(qA, qB)
        self_clearances = self.sim.self_clearances(q)
        self_pair = min(self_clearances, key=self_clearances.get)
        if self_clearances[self_pair] < cross:
            return self_clearances[self_pair], ("self", self_pair)
        return cross, pair

    def _emit(self, event=None):
        if self.recorder is not None:
            self.recorder(self.q["A"].copy(), self.q["B"].copy(), event)

    def move(self, arm, target):
        """Move `arm` toward `target`; returns (reached, contact_pair)."""
        start = self.q[arm].copy()
        delta = np.asarray(target, float) - start
        span = np.max(np.abs(delta))
        if span < 1e-12:
            return True, None
        steps = max(1, int(np.ceil(span / STEP)))
        for k in range(1, steps + 1):
            candidate = start + delta * (k / steps)
            clearance, pair = self._clearance(arm, candidate)
            if clearance <= 0.0:
                lo, hi = self.q[arm].copy(), candidate
                for _ in range(50):
                    mid = 0.5 * (lo + hi)
                    if self._clearance(arm, mid)[0] > 0.0:
                        lo = mid
                    else:
                        hi = mid
                self.q[arm] = 0.5 * (lo + hi)
                _, contact_pair = self._clearance(arm, self.q[arm])
                self._emit()
                return False, contact_pair
            self.q[arm] = candidate
            self._emit()
        return True, None


def perform_dance(X_true, verbose=True, recorder=None):
    sim = TwoArmSimulator(X_true)
    ex = MotionExecutor(sim, recorder)
    touches = []
    classes = set()

    def done():
        directions = {c[3] for c in classes}
        tilts = {c[0] for c in classes}
        return (len(touches) >= TARGET_TOUCHES
                and len(directions) >= MIN_TOUCH_DIRECTIONS
                and len(tilts) >= MIN_TOUCH_TILTS)

    def try_touch(contact_pair, contact_class):
        qA, qB = ex.q["A"].copy(), ex.q["B"].copy()
        if contact_pair == TOUCH_PAIR and sim.contact_interior(qA, qB, TOUCH_PAIR):
            touches.append((qA, qB))
            classes.add(contact_class)
            if recorder is not None:
                recorder(qA, qB, ("touch", contact_pair))
            return True
        if recorder is not None and contact_pair is not None:
            recorder(qA, qB, ("reject", contact_pair))
        return False

    def orange_pass(theta0, rung, tilt, sign, contact_class):  # noqa: ANN001
        target = orange_config(theta0 + sign * (2.0 * np.pi - SWEEP_MARGIN),
                               rung, tilt)
        reached, pair = ex.move("B", target)
        found = False
        if not reached:
            found = try_touch(pair, contact_class)
        ex.move("B", orange_config(theta0, rung, tilt))
        return found, reached

    def orange_rounds(theta0, blue_tilt, direction):
        any_found = False
        for rung in ORANGE_RUNGS:
            reached, _ = ex.move("B", orange_config(theta0, rung))
            if not reached:
                ex.move("B", fold_config(theta0))
                continue
            tilts = [0.0]
            for tilt in tilts:
                contact_class = (blue_tilt, rung, tilt, direction)
                found_cw, done_cw = orange_pass(theta0, rung, tilt, +1,
                                                contact_class)
                found_ccw = False
                if not (done_cw and not found_cw):
                    found_ccw, _ = orange_pass(theta0, rung, tilt, -1,
                                               contact_class)
                if (found_cw or found_ccw) and tilt == 0.0:
                    tilts.extend(ORANGE_TILTS)
                    any_found = True
                if done():
                    break
            ex.move("B", orange_config(theta0, rung))
            if done():
                break
        ex.move("B", fold_config(theta0))
        return any_found

    def orange_deploy():
        for theta0 in ORANGE_DEPLOY_YAWS:
            ex.move("B", fold_config(theta0))
            reached, _ = ex.move("B", orange_config(theta0, ORANGE_RUNGS[0]))
            ex.move("B", fold_config(theta0))
            if reached:
                return theta0
        return None

    for direction in A_DIRECTIONS:
        ex.move("A", fold_config(direction))
        for blue_tilt in BLUE_TILTS:
            reached, _ = ex.move("A", blue_config(direction, blue_tilt))
            if not reached:
                ex.move("A", fold_config(direction))
                break
            theta0 = orange_deploy()
            found = False
            if theta0 is not None:
                found = orange_rounds(theta0, blue_tilt, direction)
            ex.move("A", fold_config(direction))
            if not found or done():
                break
        if done():
            break

    if verbose:
        print(f"dance: {len(touches)} touches from {len(classes)} contact classes")
    return touches


# Setup priors from the problem statement (not the ground truth): the bases
# stand on a near-common plane -- |z| small, roll/pitch a few degrees, yaw
# and xy free.  Calibration seeds Gauss-Newton on that manifold and rejects
# converged candidates that leave it.
PRIOR_MAX_Z = 0.15
PRIOR_MAX_TILT_DEG = 10.0
CALIBRATION_STARTS = 500


def within_prior(X):
    tilt = np.degrees(np.arccos(np.clip(X[2, 2], -1.0, 1.0)))
    return abs(X[2, 3]) <= PRIOR_MAX_Z and tilt <= PRIOR_MAX_TILT_DEG


def calibrate(touches):
    As = [forearm(qA) for qA, _ in touches]
    Bs = [forearm(qB) for _, qB in touches]
    solver = SolverNewton(As, Bs, R_LINK, R_LINK)
    rng = np.random.default_rng(12345)
    best_cost, best_X = np.inf, None
    for _ in range(CALIBRATION_STARTS):
        yaw = rng.uniform(0.0, 2.0 * np.pi)
        radius = rng.uniform(0.2, 1.1)
        bearing = rng.uniform(0.0, 2.0 * np.pi)
        X0 = np.eye(4)
        c, si = np.cos(yaw), np.sin(yaw)
        X0[:3, :3] = [[c, -si, 0.0], [si, c, 0.0], [0.0, 0.0, 1.0]]
        X0[:3, 3] = [radius * np.cos(bearing), radius * np.sin(bearing), 0.0]
        X = solver.solve(X0, 150)
        cost = solver.cost(X)
        if cost < best_cost and within_prior(X):
            best_cost, best_X = cost, X
    return best_X


def main(seed=0):
    rng = np.random.default_rng(seed)
    X_true = sample_ground_truth(rng)

    touches = perform_dance(X_true)
    print(f"total touch poses: {len(touches)} (need >= {MIN_TOUCHES})")
    assert len(touches) >= MIN_TOUCHES

    X = calibrate(touches)

    # ------------------------------------------------------------------
    # Verification against the ground truth.
    # ------------------------------------------------------------------
    np.set_printoptions(precision=6, suppress=True)
    print("\nrecovered X:\n", X)
    print("ground truth X:\n", X_true)
    dt = X[:3, 3] - X_true[:3, 3]
    R_delta = X[:3, :3].T @ X_true[:3, :3]
    angle = np.degrees(np.arccos(np.clip((np.trace(R_delta) - 1.0) / 2.0, -1.0, 1.0)))
    print(f"\ntranslation error: {np.linalg.norm(dt) * 1000:.4f} mm "
          f"(dx {dt[0]*1000:.4f}, dy {dt[1]*1000:.4f}, dz {dt[2]*1000:.4f})")
    print(f"rotation error: {angle:.5f} deg")
    return np.linalg.norm(dt), angle


if __name__ == "__main__":
    import sys
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 0)
