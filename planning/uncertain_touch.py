"""Generating link-link touches under placement uncertainty.

Setup: R2's true base X_gt is sampled; the robot's BELIEF X_initial is a
range-box perturbation of it (X_initial = dT^-1 . X_gt with dT sampled in
the range, so the unknown correction X_gt . X_initial^-1 = dT lies in the
range and the C++ capsule expansion covers it exactly).  X_gt is used ONLY
inside GroundTruthSimulator (the collision/touch oracle) and for reporting;
every planning computation sees X_initial and the uncertainty range alone.

One attempt:
  1. Sample a touching candidate (qA, qB) with the C++ ContactGenerator
     under the BELIEVED placement.
  2. Move R1 home -> qA, guarded by the oracle; contact => retreat, fail.
  3. Back qB's base yaw off until the config clears the uncertainty-expanded
     R1 obstacles (pre-touch config), plan R2 home -> pre-touch with RRT in
     the belief world, and execute the path guarded; premature contact =>
     retreat along the executed trajectory, fail.
  4. Guarded approach: advance R2's base yaw through qB (plus overshoot)
     until contact is FELT; success iff the touching pair is forearm-forearm
     and interior (a solver-usable touch at the TRUE placement).
"""

import time

import numpy as np

from sda_bfc import (Capsule, ContactGenerator, TwoArmScene, UR5e,
                     fold_config, sample_valid_placement)

from .collision import BeliefWorld, StaticObstacle
from .rrt import rrt_plan

TOUCH_PAIR = (3, 3)
GUARD_STEP = 0.02
APPROACH_OVERSHOOT = 0.35
PRETOUCH_DELTAS = np.arange(0.08, 0.61, 0.04)


def sample_delta(rng, ranges):
    dx, dy, dz = (rng.uniform(-r, r) for r in ranges[:3])
    roll, pitch, yaw = (rng.uniform(-r, r) for r in ranges[3:])
    cz, sz = np.cos(yaw), np.sin(yaw)
    cy, sy = np.cos(pitch), np.sin(pitch)
    cx, sx = np.cos(roll), np.sin(roll)
    Rz = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]])
    Ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
    Rx = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]])
    D = np.eye(4)
    D[:3, :3] = Rz @ Ry @ Rx
    D[:3, 3] = [dx, dy, dz]
    return D


class GroundTruthSimulator:
    """The only holder of X_gt: guarded motion + touch classification."""

    def __init__(self, robot, X_gt, recorder=None):
        self.scene = TwoArmScene(robot, X_gt)
        self.q = {"A": fold_config(0.0), "B": fold_config(0.0)}
        self.recorder = recorder

    def _emit(self):
        if self.recorder is not None:
            self.recorder(self.q["A"].copy(), self.q["B"].copy())

    def move(self, arm, target):
        start = self.q[arm].copy()
        delta = np.asarray(target, float) - start
        span = np.max(np.abs(delta))
        if span < 1e-12:
            return True, None
        steps = max(1, int(np.ceil(span / GUARD_STEP)))
        for k in range(1, steps + 1):
            candidate = start + delta * (k / steps)
            qA = candidate if arm == "A" else self.q["A"]
            qB = candidate if arm == "B" else self.q["B"]
            contact = self.scene.first_cross_contact(qA, qB)
            if contact.clearance <= 0.0:
                lo, hi = self.q[arm].copy(), candidate
                for _ in range(50):
                    mid = 0.5 * (lo + hi)
                    qA = mid if arm == "A" else self.q["A"]
                    qB = mid if arm == "B" else self.q["B"]
                    if self.scene.first_cross_contact(qA, qB).clearance > 0.0:
                        lo = mid
                    else:
                        hi = mid
                self.q[arm] = 0.5 * (lo + hi)
                self._emit()
                qA = self.q["A"] if arm == "B" else self.q[arm]
                qB = self.q["B"] if arm == "A" else self.q[arm]
                contact = self.scene.first_cross_contact(self.q["A"], self.q["B"])
                return False, (contact.i, contact.j)
            self.q[arm] = candidate
            self._emit()
        return True, None

    def touch_is_valid(self):
        contact = self.scene.first_cross_contact(self.q["A"], self.q["B"])
        return ((contact.i, contact.j) == TOUCH_PAIR
                and self.scene.contact_interior(self.q["A"], self.q["B"],
                                                *TOUCH_PAIR))


class UncertainTouchSession:
    def __init__(self, seed, ranges, robot=None, recorder=None):
        self.robot = robot if robot is not None else UR5e()
        self.ranges = tuple(ranges)
        self.rng = np.random.default_rng(seed)
        self.X_gt = sample_valid_placement(seed, self.robot)
        delta = sample_delta(self.rng, self.ranges)
        self.X_initial = np.linalg.inv(delta) @ self.X_gt
        self.sim = GroundTruthSimulator(self.robot, self.X_gt, recorder)
        self.scene_identity = TwoArmScene(self.robot, np.eye(4))
        self.contact_seed = 1000 * seed
        self.touches = []
        home_caps_world = []
        R, t = self.X_initial[:3, :3], self.X_initial[:3, 3]
        for capsule in self.scene_identity.capsules(fold_config(0.0)):
            home_caps_world.append(Capsule(R @ np.asarray(capsule.a) + t,
                                           R @ np.asarray(capsule.b) + t,
                                           capsule.r))
        # Expanded believed R2-at-home: R1 must not sweep through it.
        self.r2_home_obstacle = StaticObstacle(home_caps_world, self.ranges)

    def _r1_path_clear_in_belief(self, qA, step=0.1):
        home = fold_config(0.0)
        span = np.max(np.abs(qA - home))
        n = max(1, int(np.ceil(span / step)))
        for k in range(n + 1):
            q = home + (qA - home) * (k / n)
            for capsule in self.scene_identity.capsules(q):
                if not self.r2_home_obstacle.capsule_free(
                        np.asarray(capsule.a), np.asarray(capsule.b), capsule.r):
                    return False
        return True

    def _find_pretouch(self, world, qB):
        for delta in PRETOUCH_DELTAS:
            for joint in (0, 1, 2):
                for sign in (1.0, -1.0):
                    candidate = qB.copy()
                    candidate[joint] += sign * delta
                    if world.config_free(candidate):
                        return candidate, joint, -sign
        return None, None, None

    def _plan_candidate(self):
        """Belief-side search for a viable candidate: touching configs, a
        clear R1 path, a pre-touch config, and an RRT path for R2."""
        for _ in range(10):
            self.contact_seed += 1
            contact = ContactGenerator(self.robot, self.X_initial).generate(
                seed=self.contact_seed)
            if contact is None:
                continue
            qA, qB = np.array(contact.q_a), np.array(contact.q_b)
            if not self._r1_path_clear_in_belief(qA):
                continue
            world = BeliefWorld(self.scene_identity, self.X_initial, qA,
                                self.ranges)
            if not world.config_free(fold_config(0.0)):
                continue
            q_pre, joint, sign = self._find_pretouch(world, qB)
            if q_pre is None:
                continue
            path = rrt_plan(world, fold_config(0.0), q_pre, self.rng,
                            max_iters=2500)
            if path is None:
                continue
            return qA, qB, path, joint, sign
        return None

    def _retreat(self, arm, trail):
        for q in reversed(trail):
            self.sim.move(arm, q)

    def attempt(self):
        t_start = time.perf_counter()
        result = {"outcome": None, "plan_time": 0.0, "total_time": 0.0}

        t_plan = time.perf_counter()
        candidate = self._plan_candidate()
        result["plan_time"] = time.perf_counter() - t_plan
        if candidate is None:
            result["outcome"] = "no-viable-candidate"
            result["total_time"] = time.perf_counter() - t_start
            return result
        qA, qB, path, approach_joint, approach_sign = candidate

        reached, _ = self.sim.move("A", qA)
        if not reached:
            self.sim.move("A", fold_config(0.0))
            result["outcome"] = "r1-blocked"
            result["total_time"] = time.perf_counter() - t_start
            return result

        trail = []
        for waypoint in path[1:]:
            before = self.sim.q["B"].copy()
            reached, _ = self.sim.move("B", waypoint)
            trail.append(before)
            if not reached:
                self._retreat("B", trail)
                self.sim.move("A", fold_config(0.0))
                result["outcome"] = "path-collision"
                result["total_time"] = time.perf_counter() - t_start
                return result

        target = qB.copy()
        target[approach_joint] += approach_sign * APPROACH_OVERSHOOT
        before = self.sim.q["B"].copy()
        reached, pair = self.sim.move("B", target)
        if reached:
            self.sim.move("B", before)
            self._retreat("B", trail)
            self.sim.move("A", fold_config(0.0))
            result["outcome"] = "no-contact"
            result["total_time"] = time.perf_counter() - t_start
            return result

        if pair == TOUCH_PAIR and self.sim.touch_is_valid():
            self.touches.append((self.sim.q["A"].copy(), self.sim.q["B"].copy()))
            result["outcome"] = "success"
        else:
            result["outcome"] = f"wrong-pair-{pair}"
        self.sim.move("B", before)
        self._retreat("B", trail)
        self.sim.move("A", fold_config(0.0))
        result["total_time"] = time.perf_counter() - t_start
        return result

    def collect(self, num_touches=7, max_attempts=40):
        attempts = []
        while len(self.touches) < num_touches and len(attempts) < max_attempts:
            attempts.append(self.attempt())
        return attempts
