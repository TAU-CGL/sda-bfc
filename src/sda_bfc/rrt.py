"""AORRTC (OMPL) for the free-space legs, as a drop-in for maneuver.transit.

Ported from the source repo's rrt.py, comments trimmed.  The C-space is the
6-DOF joint space of whichever arm is moving; the other arm is static.  Only
park, carry and return come here -- the approach leg stays deterministic (its
goal is a tangency, "in collision" under any clearance test).

The planner never sees the ground-truth placement: `ctx` is the DualRobot
built from the BELIEVED base offset, padded by robot.set_uncertainty.

Determinism: OMPL's RNG is process-global and seeded once at import; a fresh
process issuing the same query sequence reproduces its results exactly (two
identical queries within one process will differ).

Not imported from __init__: requires the ompl wheel.
"""

import time

import numpy as np
from ompl import base as ob
from ompl import geometric as og
from ompl import util as ou

from .maneuver import LINE_RES, line_res, path_length, transit, transit_clearance
from .robot import joint_limits

SEED = 1  # ou.RNG rejects 0
# Dead for AORRTC (no iteration count); kept because callers pass iterations=.
ITERATIONS = 4000
# What one hard query may spend.  AORRTC finds its first solution in well
# under a second; the rest is optimisation, and the optimisation is the
# point -- stop-at-first gives paths ~30% longer.  15 s was the smallest
# budget measured to beat InformedRRTstar on both time and travel.
TRANSIT_BUDGET = 15.0
FIRST_SOLUTION_ONLY = False  # module state, flippable by a bench or a step
TIME_CAP = 120.0  # outer net; TRANSIT_BUDGET is what actually ends a solve
# Shoulder_lift is the one joint whose limit lets the sampler swing a full
# turn through where the bench would be; this box spans every value the scene
# uses, with headroom, and no turn fits inside it.
SHOULDER_LIFT = 1
SHOULDER_LIFT_BOX = (-3.4, 0.2)
BRANCH_MARGIN = np.pi / 2  # how far allow_wrap=False lets a joint stray

ou.setLogLevel(ou.LOG_ERROR)
ou.RNG.setSeed(SEED)  # must happen before any OMPL RNG is drawn from


def joint_bounds(mover, q_start, q_goal, allow_wrap=True, margin=BRANCH_MARGIN):
    """Per-joint sampler bounds as (lo, hi), always containing both endpoints.
    allow_wrap=False pins every joint near the endpoints (stay on the goal's
    own branch); the shoulder-lift box is applied to BOTH branches."""
    q_start, q_goal = np.asarray(q_start, float), np.asarray(q_goal, float)
    bounds = []
    for j in range(6):
        limit = joint_limits(mover.rid, j) or (-2 * np.pi, 2 * np.pi)
        if not allow_wrap:
            lo = min(q_start[j], q_goal[j]) - margin
            hi = max(q_start[j], q_goal[j]) + margin
        else:
            lo, hi = limit
        if j == SHOULDER_LIFT:
            lo, hi = max(lo, SHOULDER_LIFT_BOX[0]), min(hi, SHOULDER_LIFT_BOX[1])
        lo, hi = max(lo, limit[0]), min(hi, limit[1])
        # a bound that excludes an endpoint makes the query unsolvable
        bounds.append((min(lo, q_start[j], q_goal[j]),
                       max(hi, q_start[j], q_goal[j])))
    return bounds


def _segment(a, b, res=LINE_RES):
    """The straight joint segment a -> b sampled at `res`, endpoints included."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    n = max(1, int(np.ceil(np.linalg.norm(b - a) / res)))
    return a + (b - a) * np.linspace(0.0, 1.0, n + 1)[:, None]


class CapsuleMotionValidator(ob.MotionValidator):
    """Edge checker that validates a whole motion in one batched
    gaps_along call instead of crossing into Python per interpolated state."""

    def __init__(self, si, ctx, mover, obstacle, clearance):
        super().__init__(si)
        self.ctx, self.mover, self.obstacle = ctx, mover, obstacle
        self.clearance = clearance

    def _configs(self, s1, s2):
        a = np.array([s1[i] for i in range(6)])
        b = np.array([s2[i] for i in range(6)])
        if not (np.isfinite(a).all() and np.isfinite(b).all()):
            return None  # see checkMotion
        n = max(1, int(np.ceil(np.linalg.norm(b - a) / line_res(self.ctx))))
        return a + (b - a) * np.linspace(0.0, 1.0, n + 1)[:, None]

    def checkMotion(self, s1, s2):
        qs = self._configs(s1, s2)
        if qs is None:
            # OMPL's informed sampler can hand back an unfilled state when its
            # attempts are exhausted; unevaluable states are not valid ones.
            return False
        return not self.ctx.blocked_along(self.mover, self.obstacle, qs,
                                          self.clearance)


def rrt_transit(ctx, mover, obstacle, q_start, q_goal, clearance=None,
                try_cartesian=True, home=None, allow_wrap=True,
                iterations=None, seed=SEED, budget=None):
    """AORRTC in mover's joint space, obstacle static.  Same (waypoints, info)
    contract as maneuver.transit, so callers swap the two with no change.
    try_cartesian and home are accepted for signature parity (the ladder
    fallback uses them).  Does not move the arm."""
    t0 = time.perf_counter()
    iterations = ITERATIONS if iterations is None else iterations
    budget = float(getattr(ctx, "transit_budget", TRANSIT_BUDGET)
                   if budget is None else budget)
    clearance = transit_clearance(ctx, clearance)
    q_start = np.asarray(q_start, float).ravel()
    q_goal = np.asarray(q_goal, float).reshape(6)
    q_goal = np.where(~np.isnan(q_goal), q_goal, q_start)

    def fail(reason):
        return [], dict(method="none", leg="none", reason=reason, len=0.0,
                        time=time.perf_counter() - t0)

    if ctx._blocked(mover, obstacle, q_start, clearance):
        return fail("start blocked: "
                    + ctx._why_blocked(mover, obstacle, q_start, clearance))
    if ctx._blocked(mover, obstacle, q_goal, clearance):
        return fail("goal blocked: "
                    + ctx._why_blocked(mover, obstacle, q_goal, clearance))

    # A clear straight segment IS the length-optimal path.  Not just an
    # optimisation: when the direct connection is optimal the informed
    # ellipse degenerates and OMPL's sampler divides by zero (NaN states).
    direct = _segment(q_start, q_goal, line_res(ctx))
    if not ctx.blocked_along(mover, obstacle, direct, clearance):
        return [q_start, q_goal], dict(
            method="direct", leg="direct", reason="",
            len=path_length([q_start, q_goal]),
            time=time.perf_counter() - t0, iterations=0)

    bounds = joint_bounds(mover, q_start, q_goal, allow_wrap)
    space = ob.RealVectorStateSpace(6)
    rvb = ob.RealVectorBounds(6)
    for j, (lo, hi) in enumerate(bounds):
        rvb.setLow(j, lo)
        rvb.setHigh(j, hi)
    space.setBounds(rvb)

    setup = og.SimpleSetup(space)
    setup.setStateValidityChecker(
        lambda s: not ctx._blocked(mover, obstacle,
                                   [s[i] for i in range(6)], clearance))
    si = setup.getSpaceInformation()
    validator = CapsuleMotionValidator(si, ctx, mover, obstacle, clearance)
    si.setMotionValidator(validator)

    # states are built by indexing, NOT copyFromReals -- that binding segfaults
    state_space = setup.getStateSpace()
    start, goal = state_space.allocState(), state_space.allocState()
    for j in range(6):
        start[j] = float(q_start[j])
        goal[j] = float(q_goal[j])
    setup.setStartAndGoalStates(start, goal)
    setup.setOptimizationObjective(ob.PathLengthOptimizationObjective(si))

    planner = og.AORRTC(si)
    setup.setPlanner(planner)
    setup.setup()

    # AORRTC is an anytime optimal planner: the wall-clock budget is spent on
    # shortening the path, which is real time on the robot.
    stop = ob.timedPlannerTerminationCondition(budget)
    if FIRST_SOLUTION_ONLY:
        stop = ob.plannerOrTerminationCondition(
            stop, ob.exactSolnPlannerTerminationCondition(
                setup.getProblemDefinition()))
    setup.solve(stop)

    if not setup.haveExactSolutionPath():
        # An approximate path ends NEAR the goal, which would break the
        # approach leg's branch pairing -- no solution at all.  Fall back to
        # the deterministic ladder, so coverage never drops below it.
        ladder, info = transit(ctx, mover, obstacle, q_start, q_goal,
                               clearance, try_cartesian, home, allow_wrap)
        if ladder:
            info["method"] = f"ladder:{info['method']}"
            info["time"] = time.perf_counter() - t0
            return ladder, info
        return fail(f"AORRTC found no exact solution in {budget:.1f}s, and "
                    "the ladder is blocked too -- " + info["reason"])

    setup.simplifySolution()
    solution = setup.getSolutionPath()
    waypoints = [np.array([s[j] for j in range(6)])
                 for s in solution.getStates()]
    waypoints[0], waypoints[-1] = q_start, q_goal  # kill any endpoint drift
    return waypoints, dict(method="rrt", leg="rrt", reason="",
                           len=path_length(waypoints),
                           time=time.perf_counter() - t0,
                           iterations=getattr(planner, "numIterations",
                                              lambda: None)())
