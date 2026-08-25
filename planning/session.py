"""Orchestration of touch collection under placement uncertainty.

TouchSession wires a belief-side planner to a collision oracle.  The oracle
is the plug-and-play boundary: pass a different oracle factory to run the
same session against the capsule simulator, the mesh simulator, or (later)
a real robot.
"""

import time
from dataclasses import dataclass
from enum import Enum

import numpy as np

from sda_bfc import UR5e, fold_config, sample_valid_placement

from .belief import TOUCH_LINK, BeliefPlanner, PlanningParams
from .capsule_oracle import CapsuleOracle
from .oracle import DYNAMIC_ARM, STATIC_ARM
from .workcell import workcell_for_placement


class Outcome(Enum):
    SUCCESS = "success"
    NO_CANDIDATE = "no-candidate"
    STATIC_BLOCKED = "static-blocked"
    PATH_COLLISION = "path-collision"
    NO_CONTACT = "no-contact"
    WRONG_PAIR = "wrong-pair"
    NON_INTERIOR = "non-interior"


@dataclass
class AttemptResult:
    outcome: Outcome
    plan_seconds: float
    total_seconds: float


def sample_belief(rng, x_true, range_tuple):
    """X_initial = dT^-1 . X_gt with dT in the range box, so the unknown
    correction X_gt . X_initial^-1 lies exactly in the expansion's range."""
    dx, dy, dz = (rng.uniform(-r, r) for r in range_tuple[:3])
    roll, pitch, yaw = (rng.uniform(-r, r) for r in range_tuple[3:])
    cz, sz, cy, sy = np.cos(yaw), np.sin(yaw), np.cos(pitch), np.sin(pitch)
    cx, sx = np.cos(roll), np.sin(roll)
    Rz = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]])
    Ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
    Rx = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]])
    delta = np.eye(4)
    delta[:3, :3] = Rz @ Ry @ Rx
    delta[:3, 3] = [dx, dy, dz]
    return np.linalg.inv(delta) @ x_true


class TouchSession:
    def __init__(self, seed, range_tuple, oracle_factory=CapsuleOracle,
                 walls=(), params=None, robot=None, recorder=None):
        self.robot = robot if robot is not None else UR5e()
        self.rng = np.random.default_rng(seed)
        self.x_true = sample_valid_placement(seed, self.robot)
        self.x_belief = sample_belief(self.rng, self.x_true, range_tuple)
        self.workcell = workcell_for_placement(self.x_true, walls)
        self.oracle = oracle_factory(self.robot, self.x_true,
                                     workcell=self.workcell, recorder=recorder)
        self.planner = BeliefPlanner(self.robot, self.x_belief, range_tuple,
                                     self.workcell,
                                     params if params else PlanningParams())
        self.plan_seed = 1000 * seed
        self.touches = []

    def attempt(self):
        started = time.perf_counter()
        plan, plan_seconds = self._timed_propose()
        if plan is None:
            return self._result(Outcome.NO_CANDIDATE, plan_seconds, started)
        outcome = self._execute(plan)
        return self._result(outcome, plan_seconds, started)

    def collect(self, num_touches=11, max_attempts=60):
        attempts = []
        while len(self.touches) < num_touches and len(attempts) < max_attempts:
            attempts.append(self.attempt())
        return attempts

    def _timed_propose(self):
        started = time.perf_counter()
        self.plan_seed += self.planner.params.candidate_budget
        plan = self.planner.propose(self.plan_seed)
        return plan, time.perf_counter() - started

    def _execute(self, plan):
        if not self.oracle.move(STATIC_ARM, plan.q_static).reached:
            self._go_home(STATIC_ARM)
            return Outcome.STATIC_BLOCKED
        trail = self._run_path(plan.path)
        if trail is None:
            self._go_home(STATIC_ARM)
            return Outcome.PATH_COLLISION
        outcome = self._approach(plan)
        self._retreat(trail)
        self._go_home(STATIC_ARM)
        return outcome

    def _run_path(self, path):
        trail = []
        for waypoint in path[1:]:
            trail.append(self.oracle.configuration(DYNAMIC_ARM))
            if not self.oracle.move(DYNAMIC_ARM, waypoint).reached:
                self._retreat(trail)
                return None
        return trail

    def _approach(self, plan):
        pre_touch = self.oracle.configuration(DYNAMIC_ARM)
        outcome = self.oracle.move(DYNAMIC_ARM, plan.approach_target)
        if outcome.reached:
            self.oracle.move(DYNAMIC_ARM, pre_touch)
            return Outcome.NO_CONTACT
        result = self._classify(plan)
        self.oracle.move(DYNAMIC_ARM, pre_touch)
        return result

    def _classify(self, plan):
        touch = self.oracle.classify_touch()
        if touch is None or touch.pair != (TOUCH_LINK, TOUCH_LINK):
            return Outcome.WRONG_PAIR
        if not touch.interior:
            return Outcome.NON_INTERIOR
        self.touches.append((self.oracle.configuration(STATIC_ARM),
                             self.oracle.configuration(DYNAMIC_ARM)))
        return Outcome.SUCCESS

    def _retreat(self, trail):
        for q in reversed(trail):
            self.oracle.move(DYNAMIC_ARM, q)

    def _go_home(self, arm):
        self.oracle.move(arm, fold_config(0.0))

    @staticmethod
    def _result(outcome, plan_seconds, started):
        return AttemptResult(outcome, plan_seconds,
                             time.perf_counter() - started)
