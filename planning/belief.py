"""Belief-side planning: everything the robot computes without X_gt.

Given the believed placement and the uncertainty range, propose a touch
plan: a static-arm config, an RRT path to a pre-touch config, and a guarded
approach whose corridor may enter only the target link's expanded hull.
"""

from dataclasses import dataclass, field

import numpy as np

from sda_bfc import (BeliefWorld, Capsule, ContactGenerator, Halfspace,
                     RRTPlanner, TwoArmScene, UncertaintyRanges,
                     cloud_intersects_capsule, expand_capsule, fold_config)

TOUCH_LINK = 3


@dataclass
class PlanningParams:
    candidate_budget: int = 10
    backoff_deltas: tuple = tuple(np.arange(0.08, 1.01, 0.04))
    approach_overshoot: float = 0.5
    max_backoff: float | None = None    # prune candidates needing deeper backoff
    corridor: str = "auto"    # "full" | "neighborhood" | "auto" | "off"
    workcell_caution: float = 0.25      # 0 = exact workcell, 1 = full margins
    home_filter_caution: float = 0.5    # range fraction for the home-sweep filter
    rrt: RRTPlanner = field(default_factory=RRTPlanner)


@dataclass(frozen=True)
class TouchPlan:
    q_static: np.ndarray
    q_touch: np.ndarray
    path: list
    approach_target: np.ndarray


def make_ranges(range_tuple):
    ranges = UncertaintyRanges()
    (ranges.x, ranges.y, ranges.z,
     ranges.roll, ranges.pitch, ranges.yaw) = range_tuple
    return ranges


class BeliefPlanner:
    def __init__(self, robot, x_belief, range_tuple, workcell, params=None):
        self.robot = robot
        self.x_belief = x_belief
        self.range_tuple = tuple(range_tuple)
        self.workcell = workcell
        self.params = params if params is not None else PlanningParams()
        self.scene = TwoArmScene(robot, np.eye(4))
        self.rng = np.random.default_rng(0)
        self._home_obstacle = self._expanded_dynamic_home()
        self._generator = self._workcell_aware_generator()

    def propose(self, seed):
        """Search candidates for a fully viable touch plan."""
        for offset in range(self.params.candidate_budget):
            contact = self._generator.generate(seed=seed + offset)
            if contact is None:
                continue
            plan = self._plan_for(np.array(contact.q_a), np.array(contact.q_b),
                                  seed + offset)
            if plan is not None:
                return plan
        return None

    def _plan_for(self, q_static, q_touch, seed):
        if not self._static_path_clear(q_static):
            return None
        world = self._build_world(q_static)
        if not world.is_free(fold_config(0.0)):
            return None
        approach = self._find_approach(world, q_touch)
        if approach is None:
            return None
        q_pre, approach_target = approach
        path = self.params.rrt.plan(world, fold_config(0.0), q_pre, seed)
        if path is None:
            return None
        return TouchPlan(q_static, q_touch, [np.array(q) for q in path],
                         approach_target)

    def _find_approach(self, world, q_touch):
        """A backed-off, strictly-free pre-touch config with an admissible
        approach corridor.  Corridor modes trade protection for yield:
        "full" admits only the target hull, "neighborhood" also its +-1
        hulls (ambiguity there is resolved by felt-pair classification),
        "off" approaches blind beyond the pre-touch, and "auto" prefers a
        protected corridor but falls back to blind when the expanded hulls
        leave none (large uncertainty)."""
        modes = {"full": ["full"], "neighborhood": ["neighborhood"],
                 "off": ["off"],
                 "auto": ["neighborhood", "off"]}[self.params.corridor]
        for mode in modes:
            approach = self._search_approach(world, q_touch, mode)
            if approach is not None:
                return approach
        return None

    def _search_approach(self, world, q_touch, mode):
        overshoot = self._overshoot()
        for delta in self.params.backoff_deltas:
            if self.params.max_backoff is not None \
                    and delta > self.params.max_backoff:
                return None
            for joint in (0, 1, 2):
                for sign in (1.0, -1.0):
                    q_pre = q_touch.copy()
                    q_pre[joint] += sign * delta
                    target = q_touch.copy()
                    target[joint] -= sign * overshoot
                    if world.is_free(q_pre) and self._corridor_admissible(
                            world, q_pre, target, mode):
                        return q_pre, target
        return None

    @staticmethod
    def _corridor_admissible(world, q_pre, target, mode):
        if mode == "off":
            return True
        if mode == "full":
            return world.corridor_edge_free(q_pre, target, TOUCH_LINK,
                                            TOUCH_LINK, TOUCH_LINK)
        return world.corridor_edge_free(q_pre, target, TOUCH_LINK - 1,
                                        TOUCH_LINK + 1, TOUCH_LINK)

    def _overshoot(self):
        """Deep enough to reach the true surface anywhere in the range."""
        _, _, _, roll, pitch, yaw = self.range_tuple
        return self.params.approach_overshoot + 2.0 * (roll + pitch + yaw)

    def _build_world(self, q_static):
        world = BeliefWorld(self.robot, self.x_belief, q_static,
                            make_ranges(self.range_tuple))
        for normal, offset in self._belief_halfspaces():
            world.add_halfspace(Halfspace(normal, offset))
        return world

    def _belief_halfspaces(self):
        if self.workcell is None:
            return []
        x, y, z, roll, pitch, yaw = self.range_tuple
        # Margins are a yield/failed-attempt trade, not a safety one: an
        # environment contact is guarded (felt, retreated from).  Caution
        # scales them; lever arms per axis as before.
        caution = self.params.workcell_caution
        return self.workcell.halfspaces(
            translation_margin=(caution * x, caution * y, caution * z),
            tilt_margins=(caution * 0.5 * (pitch + yaw),
                          caution * 0.5 * (roll + yaw),
                          caution * 0.3 * (roll + pitch)))

    def _workcell_aware_generator(self):
        """Candidates are born workcell-legal: exact halfspaces for the
        static arm, uncertainty-margined ones for the dynamic arm."""
        generator = ContactGenerator(self.robot, self.x_belief)
        if self.workcell is not None:
            for normal, offset in self.workcell.halfspaces():
                generator.add_static_halfspace(Halfspace(normal, offset))
            for normal, offset in self._belief_halfspaces():
                generator.add_dynamic_halfspace(Halfspace(normal, offset))
        return generator

    def _static_path_clear(self, q_static, step=0.1):
        """The static arm must not sweep the expanded believed dynamic home."""
        home = fold_config(0.0)
        span = np.max(np.abs(q_static - home))
        n = max(1, int(np.ceil(span / step)))
        for k in range(n + 1):
            q = home + (q_static - home) * (k / n)
            for capsule in self.scene.capsules(q):
                for cloud in self._home_obstacle:
                    if cloud_intersects_capsule(cloud, np.asarray(capsule.a),
                                                np.asarray(capsule.b), capsule.r):
                        return False
        return True

    def _expanded_dynamic_home(self):
        """Filter obstacle for the static arm's sweep.  Caution < 1 trades
        occasional guarded static-blocked attempts for candidate yield."""
        caution = self.params.home_filter_caution
        R, t = self.x_belief[:3, :3], self.x_belief[:3, 3]
        ranges = make_ranges(tuple(caution * r for r in self.range_tuple))
        clouds = []
        for capsule in self.scene.capsules(fold_config(0.0)):
            world_capsule = Capsule(R @ np.asarray(capsule.a) + t,
                                    R @ np.asarray(capsule.b) + t, capsule.r)
            clouds.append(expand_capsule(world_capsule, ranges))
        return clouds
