"""Simulation oracle backed by the calibrated capsule model (fast)."""

import numpy as np

from sda_bfc import TwoArmScene, fold_config

from .oracle import (DYNAMIC_ARM, STATIC_ARM, CollisionOracle, MoveOutcome,
                     TouchReport)

GUARD_STEP = 0.02
BISECTION_ITERATIONS = 50


class CapsuleOracle(CollisionOracle):
    def __init__(self, robot, x_true, workcell=None, recorder=None):
        self.robot = robot
        self.scene = TwoArmScene(robot, x_true)
        self.touch_radius = self.scene.link_radius(3)
        self.x_true = x_true
        self.workcell = workcell
        self.recorder = recorder
        self.q = {STATIC_ARM: fold_config(0.0), DYNAMIC_ARM: fold_config(0.0)}

    def configuration(self, arm):
        return self.q[arm].copy()

    def move(self, arm, target):
        start = self.q[arm].copy()
        delta = np.asarray(target, float) - start
        span = np.max(np.abs(delta))
        if span < 1e-12:
            return MoveOutcome(True, None)
        steps = max(1, int(np.ceil(span / GUARD_STEP)))
        for k in range(1, steps + 1):
            candidate = start + delta * (k / steps)
            if self._in_contact(arm, candidate):
                self._bisect_to_contact(arm, candidate)
                return MoveOutcome(False, self._current_contact())
            self.q[arm] = candidate
            self._emit()
        return MoveOutcome(True, None)

    def classify_touch(self):
        contact = self.scene.first_cross_contact(self.q[STATIC_ARM],
                                                 self.q[DYNAMIC_ARM])
        if contact.clearance > 1e-6:
            return None
        interior = self.scene.contact_interior(
            self.q[STATIC_ARM], self.q[DYNAMIC_ARM], contact.i, contact.j)
        return TouchReport(pair=(contact.i, contact.j), interior=interior)

    def _in_contact(self, arm, candidate):
        qA, qB = self._configs_with(arm, candidate)
        if self.scene.first_cross_contact(qA, qB).clearance <= 0.0:
            return True
        return self._hits_workcell(arm, candidate)

    def _hits_workcell(self, arm, candidate):
        if self.workcell is None:
            return False
        base = np.eye(4) if arm == STATIC_ARM else self.x_true
        R, t = base[:3, :3], base[:3, 3]
        for i, capsule in enumerate(self.scene.capsules(np.asarray(candidate))):
            if i < 2:   # the base column stands on the floor by construction
                continue
            a, b = R @ np.asarray(capsule.a) + t, R @ np.asarray(capsule.b) + t
            for normal, offset in self.workcell.halfspaces():
                if max(normal @ a, normal @ b) + capsule.r > offset:
                    return True
        return False

    def _bisect_to_contact(self, arm, colliding):
        lo, hi = self.q[arm].copy(), colliding
        for _ in range(BISECTION_ITERATIONS):
            mid = 0.5 * (lo + hi)
            if self._in_contact(arm, mid):
                hi = mid
            else:
                lo = mid
        self.q[arm] = 0.5 * (lo + hi)
        self._emit()

    def _current_contact(self):
        contact = self.scene.first_cross_contact(self.q[STATIC_ARM],
                                                 self.q[DYNAMIC_ARM])
        if contact.clearance <= 1e-6:
            return (contact.i, contact.j)
        return ("env", "workcell")

    def _configs_with(self, arm, candidate):
        qA = candidate if arm == STATIC_ARM else self.q[STATIC_ARM]
        qB = candidate if arm == DYNAMIC_ARM else self.q[DYNAMIC_ARM]
        return np.asarray(qA), np.asarray(qB)

    def _emit(self):
        if self.recorder is not None:
            self.recorder(self.q[STATIC_ARM].copy(),
                          self.q[DYNAMIC_ARM].copy())
