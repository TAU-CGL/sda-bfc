"""Deterministic motion for the two-arm scene: retract / transit / approach.

Ported from the source repo's maneuver.py, comments trimmed.  Every motion
decomposes into legs that never sample -- same inputs, same path, every time:

  RETRACT   back straight away from the other forearm along the contact
            normal until a target gap (retract_to) or strictly free
            (retract_clear).
  TRANSIT   free-space ladder: straight forearm line, straight joint line,
            detour via home, detour via fixed VIA configs.
  APPROACH  straight back in, gap monotone, stopping at tangency.

Cartesian means the FOREARM here, never the TCP: forearm centre position is 3
constraints on the 3 joints that drive it -- square, no branches to flip.
`ctx` is the DualArm (see dual_arm.py): its _blocked/_why_blocked oracle and
_forearm_cyl are reused everywhere rather than reimplemented.
"""

import time
from contextlib import contextmanager

import numpy as np

from .robot import Robot, forearm_gaps, forearm_shafts, joint_limits

# --------------------------------------------------------------------- tuning
CREEP_RES = 0.01    # rad; joint step for retract/approach
CART_RES = 0.004    # m; cartesian step per 'cartesian'-rule step
LINE_RES = 0.05     # rad; collision-check resolution along a transit leg
GRAD_EPS = 1e-3     # rad; central-difference step
DLS_LAMBDA = 1e-3   # damping for the point-Jacobian pseudo-inverse
MAX_STEPS = 800     # safety bound on a retract
SAFE_GAP = 0.15     # m; forearm gap at which the arm counts as clear
TANGENT_TILT = np.radians(20)  # tried to each side when a retract step stalls

# Joints allowed to reach a goal +-2pi round the other way: pan + wrists.
# shoulder_lift and elbow excluded -- a full turn drives the arm through the
# floor, and this scene has no floor for the oracle to catch it with.
WRAP_JOINTS = (0, 3, 4, 5)

# Fixed detour configs, tried in order.  NOT sampled -- reproducibility is the
# point.  q123 only; the wrist rides at WRIST_FIXED.
VIA_Q123 = [
    (0.0, -1.2, 0.9),
    (1.5708, -1.5708, 0.0),
    (-1.5708, -1.5708, 0.0),
    (0.0, -2.2, 0.0),
    (0.0, -1.0, -1.0),
]


def path_length(path):
    q = np.asarray(path, float)
    return float(np.sum(np.linalg.norm(np.diff(q, axis=0), axis=1))) \
        if len(q) > 1 else 0.0


def densify(wps, res=CREEP_RES):
    """Waypoints resampled so consecutive configs are at most `res` apart."""
    wps = [np.asarray(q, float) for q in wps]
    out = [wps[0]]
    for a, b in zip(wps, wps[1:]):
        n = max(1, int(np.linalg.norm(b - a) / res))
        out.extend(a + (b - a) * i / n for i in range(1, n + 1))
    return out


def _q6(q):
    """Accept a q123 or a full 6-vector; return the 6-vector."""
    q = np.asarray(q, float).ravel()
    return q.copy() if q.size == 6 else np.concatenate([q[:3], Robot.WRIST_FIXED])


def _in_limits(mover, q6):
    return all(lim is None or lim[0] <= qi <= lim[1]
               for j, qi in enumerate(q6)
               for lim in [joint_limits(mover, j)])


# --------------------------------------------------------------- gap geometry
def _gap_of(ctx, mover, obstacle_cyl, q6):
    """Analytic surface-to-surface forearm clearance at q6.  Moves the arm."""
    mover.set_arm(q6)
    return ctx._forearm_cyl(mover).lateral_gap(obstacle_cyl)[0]


def _witness(ctx, mover, obstacle_cyl, q6):
    """(gap, c_mover, n): closest-approach point on the mover's forearm axis
    and the unit common perpendicular pointing obstacle -> mover."""
    mover.set_arm(q6)
    gap, c_obs, c_mov, _, _ = obstacle_cyl.lateral_gap(ctx._forearm_cyl(mover))
    n = c_mov - c_obs
    nn = float(np.linalg.norm(n))
    return gap, c_mov, (n / nn if nn > 1e-12 else np.array([0.0, 0.0, 1.0]))


def _gap_grad(ctx, mover, obstacle_cyl, q6):
    """d(gap)/d(q1,q2,q3) by central differences (envelope theorem: equals
    n^T J_witness, so stepping along it opens the gap along the normal)."""
    qs = np.repeat(np.asarray(q6, float)[None], 6, axis=0)
    for i in range(3):
        qs[2 * i, i] += GRAD_EPS
        qs[2 * i + 1, i] -= GRAD_EPS
    gaps = forearm_gaps(mover, qs, obstacle_cyl)
    mover.set_arm(q6)
    return (gaps[0::2] - gaps[1::2]) / (2.0 * GRAD_EPS)


def _material_points(mover, qs, u):
    """(M, 3) world positions of the material point `u` along the shaft."""
    p0, p1 = forearm_shafts(mover, qs)
    span = p1 - p0
    height = np.linalg.norm(span, axis=1)
    axis = span / np.where(height > 1e-12, height, 1.0)[:, None]
    return 0.5 * (p0 + p1) + u * axis


def _point_jacobian(ctx, mover, q6, u=0.0):
    """3x3 linear Jacobian of that material point wrt q1,q2,q3, by central
    differences on the same analytic cylinder everything else uses."""
    qs = np.repeat(np.asarray(q6, float)[None], 6, axis=0)
    for i in range(3):
        qs[2 * i, i] += GRAD_EPS
        qs[2 * i + 1, i] -= GRAD_EPS
    pts = _material_points(mover, qs, u)
    mover.set_arm(q6)
    return ((pts[0::2] - pts[1::2]) / (2.0 * GRAD_EPS)).T


def _dls(J, e, max_step):
    """Damped-least-squares joint step realising cartesian error e, capped."""
    dq = J.T @ np.linalg.solve(J @ J.T + DLS_LAMBDA * np.eye(3), e)
    m = float(np.linalg.norm(dq))
    if m < 1e-12:
        return None
    return dq if m <= max_step else dq / m * max_step


# --------------------------------------------------------- forearm cartesian
def forearm_ik(ctx, mover, target_pos, q_seed, obstacle=None, clearance=0.01,
               tol=1e-5, max_iter=200, max_step=0.2):
    """q6 putting the forearm cylinder CENTRE at target_pos, or None.  DLS
    Newton on the 3x3 point Jacobian, seeded at q_seed then the fixed VIA
    configs (a singular seed has no descent direction).  The solution set is
    discrete; pass `obstacle` to reject branches that collide."""
    target_pos = np.asarray(target_pos, float)
    for seed in [q_seed, *VIA_Q123]:
        q = _q6(seed)
        for _ in range(max_iter):
            mover.set_arm(q)
            e = target_pos - ctx._forearm_cyl(mover).center
            if float(np.linalg.norm(e)) < tol:
                if _in_limits(mover, q) and not (
                        obstacle is not None
                        and ctx._blocked(mover, obstacle, q, clearance)):
                    mover.set_arm(q)
                    return q
                break
            dq = _dls(_point_jacobian(ctx, mover, q), e, max_step)
            if dq is None:
                break
            q = q + np.concatenate([dq, np.zeros(3)])
    return None


def forearm_line(ctx, mover, obstacle, q_start, q_goal, clearance,
                 pos_res=0.01, jump_tol=0.35):
    """The moveL leg: walk the forearm centre along the straight world line,
    re-solving the 3-DOF IK each step seeded on the previous one.  [] if the
    line is infeasible (IK failure, joint jump, collision, wrong branch)."""
    q_start, q_goal = _q6(q_start), _q6(q_goal)
    mover.set_arm(q_start)
    c0 = ctx._forearm_cyl(mover).center
    mover.set_arm(q_goal)
    c1 = ctx._forearm_cyl(mover).center
    n = max(2, int(np.ceil(float(np.linalg.norm(c1 - c0)) / pos_res)))
    path, q_prev = [q_start], q_start
    for t in np.linspace(0.0, 1.0, n + 1)[1:]:
        q = forearm_ik(ctx, mover, (1 - t) * c0 + t * c1, q_seed=q_prev)
        if q is None or float(np.max(np.abs(q - q_prev))) > jump_tol:
            return []
        step = line_res(ctx)
        for s in np.linspace(0.0, 1.0,
                             max(1, int(np.linalg.norm(q - q_prev) / step)) + 1)[1:]:
            if ctx._blocked(mover, obstacle, (1 - s) * q_prev + s * q, clearance):
                return []
        path.append(q)
        q_prev = q
    d_end = float(np.linalg.norm(q_prev - q_goal))
    if d_end > 0.2:
        return []  # reached the point on a different branch
    for t in np.linspace(0.0, 1.0, max(2, int(d_end / 0.02)) + 1)[1:]:
        if ctx._blocked(mover, obstacle, (1 - t) * q_prev + t * q_goal, clearance):
            return []
    path.append(q_goal)
    return path


# ---------------------------------------------------------- retract steppers
def _step_gradient(ctx, mover, obstacle_cyl, q6, q_home):
    g = _gap_grad(ctx, mover, obstacle_cyl, q6)
    n = float(np.linalg.norm(g))
    if n < 1e-9:
        return None
    return np.concatenate([g / n * CREEP_RES, np.zeros(3)])


def _step_cartesian(ctx, mover, obstacle_cyl, q6, q_home):
    _, c_mov, n = _witness(ctx, mover, obstacle_cyl, q6)
    # Jacobian of the MATERIAL point currently at the witness, not of "the
    # witness", which slides along the shaft as the arm moves
    cyl = ctx._forearm_cyl(mover)
    u = float(np.dot(c_mov - cyl.center, cyl.axis))
    dq = _dls(_point_jacobian(ctx, mover, q6, u), n * CART_RES, CREEP_RES)
    if dq is None:
        return None
    return np.concatenate([dq, np.zeros(3)])


def _step_lerp(ctx, mover, obstacle_cyl, q6, q_home):
    """The baseline: head straight for home in joint space (wrist included)."""
    d = _q6(q_home) - q6
    m = float(np.linalg.norm(d))
    if m < 1e-9:
        return None
    return d / m * min(CREEP_RES, m)


STEPPERS = {"gradient": _step_gradient, "cartesian": _step_cartesian,
            "lerp": _step_lerp}
RETRACT_RULES = tuple(STEPPERS)


def _tangent_dir(obstacle_cyl, n):
    t = obstacle_cyl.axis - np.dot(obstacle_cyl.axis, n) * n
    tn = float(np.linalg.norm(t))
    return None if tn < 1e-9 else t / tn


def _step_tilted(ctx, mover, obstacle_cyl, q6, angle):
    """Cartesian retreat at `angle` off the normal, tilted toward the
    obstacle's axis: slides the witness sideways when the pure-normal push
    stalls, like sliding a finger along a rod instead of pushing off it."""
    _, c_mov, n = _witness(ctx, mover, obstacle_cyl, q6)
    t = _tangent_dir(obstacle_cyl, n)
    if t is None:
        return None
    d = np.cos(angle) * n + np.sin(angle) * t
    dn = float(np.linalg.norm(d))
    if dn < 1e-9:
        return None
    cyl = ctx._forearm_cyl(mover)
    u = float(np.dot(c_mov - cyl.center, cyl.axis))
    dq = _dls(_point_jacobian(ctx, mover, q6, u), d / dn * CART_RES, CREEP_RES)
    return None if dq is None else np.concatenate([dq, np.zeros(3)])


def _retract_steps(ctx, mover, obstacle, q_start, rule, q_home, stop):
    """Walk away one CREEP_RES step at a time until stop(q, gap), checking
    every step (forearm pair exempted -- that contact is the point).  On a
    stalled or colliding step, retries tilted to either side of the normal."""
    step_fn = STEPPERS[rule]
    obstacle_cyl = ctx._forearm_cyl(obstacle)
    q = _q6(q_start)
    path = [q.copy()]
    for _ in range(MAX_STEPS):
        gap = _gap_of(ctx, mover, obstacle_cyl, q)
        if stop(q, gap):
            return path
        moved = False
        for angle in (0.0, TANGENT_TILT, -TANGENT_TILT):
            dq = (step_fn(ctx, mover, obstacle_cyl, q, q_home) if angle == 0.0
                  else _step_tilted(ctx, mover, obstacle_cyl, q, angle))
            if dq is None:
                continue
            q_next = q + dq
            if not ctx._blocked(mover, obstacle, q_next, clearance=0.0,
                                ignore_forearm_pair=True):
                q, moved = q_next, True
                path.append(q.copy())
                break
        if not moved:
            return None
    return None


def retract_to(ctx, mover, obstacle, q_touch, standoff=0.10,
               rule="gradient", q_home=None):
    """Retract from the touch until the forearm gap >= standoff AND the config
    is strictly free.  Returns waypoints (q_touch first) or None."""
    def stop(q, gap):
        return gap >= standoff and not ctx._blocked(mover, obstacle, q)
    return _retract_steps(ctx, mover, obstacle, q_touch, rule, q_home, stop)


def retract_clear(ctx, mover, obstacle, q_start, rule="gradient", q_home=None,
                  clearance=0.01):
    """Leg 0: at a touch the collision hulls overlap, so back off just far
    enough to be strictly free.  [q_start] when already clear."""
    q0 = _q6(q_start)
    already = not ctx._blocked(mover, obstacle, q0, clearance)
    mover.set_arm(q0)
    if already:
        return [q0]
    return _retract_steps(
        ctx, mover, obstacle, q0, rule, q_home,
        stop=lambda q, gap: not ctx._blocked(mover, obstacle, q, clearance))


# ------------------------------------------------------------------ approach
def _approach_ok(ctx, mover, obstacle, q_pre, q_touch):
    """Straight segment is free at CREEP_RES AND the gap falls monotonically
    (a dip means the creep can trip on an earlier graze)."""
    obstacle_cyl = ctx._forearm_cyl(obstacle)
    prev = np.inf
    for q in densify([q_pre, q_touch]):
        if ctx._blocked(mover, obstacle, q, clearance=0.0,
                        ignore_forearm_pair=True):
            return False
        gap = _gap_of(ctx, mover, obstacle_cyl, q)
        if gap > prev + 1e-9:
            return False
        prev = gap
    return True


def approach_line(ctx, mover, obstacle, q_pre, q_touch, retract_path=None,
                  shrink_tries=5):
    """One straight joint segment into the touch; q_pre halved toward q_touch
    on failure (a shorter standoff beats a graze).  [q_pre, q_touch] or None."""
    q_touch = _q6(q_touch)
    q = _q6(q_pre)
    for _ in range(shrink_tries + 1):
        if _approach_ok(ctx, mover, obstacle, q, q_touch):
            return [q, q_touch]
        q = 0.5 * (q + q_touch)
    return None


def approach_retrace(ctx, mover, obstacle, q_pre, q_touch, retract_path=None):
    """The recorded retract played backwards: free and gap-monotone by
    construction, at the cost of multi-waypoint streaming on the real arms."""
    if not retract_path or len(retract_path) < 2:
        return None
    return [np.asarray(q, float) for q in retract_path[::-1]]


APPROACH_MODES = {"line": approach_line, "retrace": approach_retrace}


# ------------------------------------------------------------------- transit
def _goal_variants(mover, goal, q_start):
    """`goal` plus every in-limits single-joint +-2pi wrap of it (WRAP_JOINTS
    only), nearest to q_start first: the same pose can be a very different
    move on the other branch."""
    out = [goal]
    for i in WRAP_JOINTS:
        lim = joint_limits(mover, i)
        for turn in (2.0 * np.pi, -2.0 * np.pi):
            v = goal.copy()
            v[i] += turn
            if lim is None or lim[0] <= v[i] <= lim[1]:
                out.append(v)
    return sorted(out, key=lambda v: float(np.linalg.norm(v - q_start)))


def _line(ctx, mover, obstacle, a, b, clearance):
    """(ok, reason) for the straight joint segment a -> b."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    n = max(1, int(np.linalg.norm(b - a) / line_res(ctx)))
    for t in np.linspace(0.0, 1.0, n + 1):
        q = a + (b - a) * t
        if ctx._blocked(mover, obstacle, q, clearance):
            return False, f"t={t:.2f}: {ctx._why_blocked(mover, obstacle, q, clearance)}"
    return True, ""


# Transit-leg standoff (park/carry/return/direct only; retract and approach
# are contact motions and set their own floors).
TRANSIT_CLEARANCE = 0.010


def line_res(ctx, res=None):
    """Collision-check step: the caller's, else the scene's, else LINE_RES."""
    return float(getattr(ctx, "line_res", LINE_RES) if res is None else res)


def transit_clearance(ctx, clearance=None):
    """Transit clearance: the caller's, else the scene's.  Resolved per call
    so the knob can be turned on the scene between queries."""
    return float(getattr(ctx, "transit_clearance", TRANSIT_CLEARANCE)
                 if clearance is None else clearance)


# How close the forearms may come on a leg to or from home: a retract can
# settle under the transit clearance, and the leg must not be refused for
# standing exactly where it was asked to stand.  Every other pair keeps the
# full clearance.
HOME_FOREARM_FLOOR = 0.009


def home_forearm_floor(ctx, floor=None):
    return float(getattr(ctx, "home_forearm_floor", HOME_FOREARM_FLOOR)
                 if floor is None else floor)


@contextmanager
def relaxed_forearm(ctx, floor):
    """Hold the forearm/forearm pair to `floor` m instead of the leg
    clearance, in here only.  Lowers only."""
    outer = getattr(ctx, "forearm_floor", None)
    ctx.forearm_floor = float(floor)
    try:
        yield
    finally:
        ctx.forearm_floor = outer


def transit(ctx, mover, obstacle, q_start, q_goal, clearance=None,
            try_cartesian=True, home=None, allow_wrap=True):
    """Free-space motion: fixed ladder of legs, first clear one wins --
    fore-line, direct, via-home, via-list.  The free space is star-shaped
    around home, which is why no planner is needed.  NaN entries in q_goal
    are filled from q_start.  Returns (waypoints, info); restores the arm."""
    t0 = time.perf_counter()
    clearance = transit_clearance(ctx, clearance)
    q_start = _q6(q_start)
    q_goal = np.asarray(q_goal, float).reshape(6)
    goal = np.where(~np.isnan(q_goal), q_goal, q_start)
    home = _q6(ctx.AWAY_Q if home is None else home)

    def done(path, method, leg):
        mover.set_arm(q_start)  # planning must not move the arm
        return ([np.asarray(q, float) for q in path],
                dict(method=method, leg=leg, reason="", len=path_length(path),
                     time=time.perf_counter() - t0))

    def fail(reason):
        mover.set_arm(q_start)
        return [], dict(method="none", leg="none", reason=reason, len=0.0,
                        time=time.perf_counter() - t0)

    if ctx._blocked(mover, obstacle, q_start, clearance):
        return fail("start blocked: "
                    + ctx._why_blocked(mover, obstacle, q_start, clearance))

    # allow_wrap=False pins the EXACT goal: q_pre pairs with q_touch on its
    # own branch, and landing 2pi away turns the approach into a sweep.
    goals = _goal_variants(mover, goal, q_start) if allow_wrap else [goal]
    vias = [("via-home", home)] + \
        [(f"via-{i}", _q6(v)) for i, v in enumerate(VIA_Q123)]
    tried = []

    def ladder(q_from, allow_fore_line):
        # the forearm line is a workspace path: only the nearest branch tracks it
        if allow_fore_line:
            line = forearm_line(ctx, mover, obstacle, q_from, goals[0], clearance)
            if line:
                return line, "fore-line", "fore-line"
            tried.append("fore-line: no straight forearm line")
        why = ""
        for g in goals:
            ok, why = _line(ctx, mover, obstacle, q_from, g, clearance)
            if ok:
                return [q_from, g], "direct", "direct"
        tried.append(f"direct blocked at {why}")
        for name, via in vias:
            ok, why = _line(ctx, mover, obstacle, q_from, via, clearance)
            if not ok:
                tried.append(f"{name} out blocked at {why}")
                continue
            for g in goals:
                ok, why = _line(ctx, mover, obstacle, via, g, clearance)
                if ok:
                    return ([q_from, via, g],
                            "via-home" if name == "via-home" else "via-list",
                            name)
            tried.append(f"{name} in blocked at {why}")
        return None

    got = ladder(q_start, try_cartesian)
    if got:
        return done(*got)

    # Boxed in near the other arm: back off along the contact normal first,
    # then run the ladder from the opened-up config.
    mover.set_arm(q_start)
    if _gap_of(ctx, mover, ctx._forearm_cyl(obstacle), q_start) < SAFE_GAP:
        back = _retract_steps(ctx, mover, obstacle, q_start,
                              getattr(ctx, "retract_rule", "gradient"), home,
                              stop=lambda q, gap: gap >= SAFE_GAP
                              and not ctx._blocked(mover, obstacle, q, clearance))
        if back is None:
            tried.append(f"retract to {SAFE_GAP * 1e2:.0f}cm failed")
        else:
            got = ladder(back[-1], allow_fore_line=False)
            if got:
                rest, method, leg = got
                return done(list(back) + list(rest[1:]), method,
                            f"retract+{leg}")

    return fail("no deterministic leg is clear -- " + "; ".join(tried))
