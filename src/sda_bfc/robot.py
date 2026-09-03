"""The `ctx` maneuver.py plans against: two UR5e arms + a capsule oracle.

Ported from the source repo's kinematics/robot.py, comments trimmed.  It asks
for no physics engine: one capsule per link, batched FK, and segment-segment
distances.  The one structural change in this port: the kinematic chain is
built from the library's own DH parameters (identical to fk_ur5e.hpp) rather
than a yourdfpy URDF.  The capsule table's link frames are DH joint frames
moved to each link's proximal joint -- verified exact against the C++ FK for
the base..forearm links; the WRIST link frames are taken as the plain DH
joint frames, so confirm those capsule placements against the source URDF.

The capsule radii are the REAL arm's, not the collision STLs' (those are
padded boxes; planning against them rejects everything).  The forearm radius
is the measured contact radius.
"""

from functools import lru_cache

import numpy as np

from .config import BASE_OFFSET, FOREARM_LINK, FOREARM_RADIUS, HOME_Q
from .cylinder import Cylinder, segment_closest_points  # noqa: F401 (re-export)
from .uncertainty import inflate


def offset_matrix(offset=None):
    """4x4 world placement from an (xyz, rotation vector) offset."""
    offset = BASE_OFFSET if offset is None else np.asarray(offset, float)
    angle = float(np.linalg.norm(offset[3:]))
    T = np.eye(4)
    if angle > 1e-12:
        x, y, z = offset[3:] / angle
        K = np.array([[0, -z, y], [z, 0, -x], [-y, x, 0]])
        T[:3, :3] = np.eye(3) + np.sin(angle) * K + (1 - np.cos(angle)) * K @ K
    T[:3, 3] = offset[:3]
    return T


# Wrist re-poses that leave the forearm cylinder untouched (joints 3-5 do not
# move it); all self-collision-free at every recorded pose.
WRIST_PRESETS = {
    "as recorded": None,
    "tuck-back":   (-np.pi, -np.pi / 2, 0.0),
    "tuck-in":     (-np.pi, -np.pi, 0.0),
    "tuck-out":    (0.0, -np.pi / 2, 0.0),
    "tuck-up":     (-np.pi / 2, -np.pi / 2, 0.0),
}


def scene():
    """The two-arm cell: arm A at the world origin, arm B at BASE_OFFSET."""
    return DualRobot(Robot(name="A"), Robot(offset_matrix(), name="B"))


# One capsule per link in LINK coordinates: (p0, p1, radius).  Shafts follow
# the physical bodies (the upper arm sits 0.138 m off its origin, the forearm
# 0.007 m); radii are the real tube radii.  The gripper is the Robotiq 2F-85
# fingers-closed envelope measured off its collision meshes; the knuckle
# capsules cover the joint housings the two long links' shaft radii do not.
LINK_CAPSULES = {
    "base_link":      ((0.0, 0.0, 0.025),     (0.0, 0.0, 0.075),      0.075),
    "shoulder_link":  ((0.0, -0.008, -0.025), (0.0, -0.008, 0.025),   0.078),
    "upper_arm_link": ((0.0, 0.0, 0.138),     (-0.425, 0.0, 0.138),   0.060),
    "forearm_link":   ((0.0, 0.0, 0.007),     (-0.3922, 0.0, 0.007),  FOREARM_RADIUS),
    "wrist_1_link":   ((0.0, -0.030, -0.018), (0.0, 0.030, -0.018),   0.057),
    "wrist_2_link":   ((0.0, -0.002, -0.030), (0.0, -0.002, 0.030),   0.056),
    "wrist_3_link":   ((0.0, -0.018, -0.023), (0.0, 0.018, -0.023),   0.036),
    "gripper":        ((0.0, 0.0, 0.039),     (0.0, 0.0, 0.092),      0.0627),
    "elbow":          ((0.0054, 0.0, 0.007),  (-0.0073, 0.0, 0.007),  0.0680),
    "forearm_collar": ((-0.3882, 0.0, 0.007), (-0.3953, 0.0, 0.007),  0.0545),
    "shoulder_yoke":  ((0.0036, 0.0, 0.138),  (-0.0087, 0.0, 0.138),  0.0755),
    "elbow_yoke":     ((-0.4209, 0.0, 0.138), (-0.4281, 0.0, 0.138),  0.0864),
}
LINKS = tuple(LINK_CAPSULES)

# Links bolted to a parent rather than driven by a joint.  A fixed link must
# follow its parent in LINKS (batch_link_T walks them in one pass).
FIXED_LINKS = {
    "gripper": ("wrist_3_link", np.eye(4)),
    "elbow": ("forearm_link", np.eye(4)),
    "forearm_collar": ("forearm_link", np.eye(4)),
    "shoulder_yoke": ("upper_arm_link", np.eye(4)),
    "elbow_yoke": ("upper_arm_link", np.eye(4)),
}

KINEMATIC_LINKS = tuple(l for l in LINKS if l not in FIXED_LINKS)

# Which of one arm's own links can reach each other.  NOT "everything two
# apart": the wrist capsules overlap by construction.
SELF_PAIRS = (
    ("base_link", "forearm_link"),
    ("base_link", "wrist_1_link"),
    ("base_link", "wrist_2_link"),
    ("base_link", "wrist_3_link"),
    ("shoulder_link", "forearm_link"),
    ("shoulder_link", "wrist_1_link"),
    ("shoulder_link", "wrist_2_link"),
    ("shoulder_link", "wrist_3_link"),
    ("upper_arm_link", "wrist_1_link"),
    ("upper_arm_link", "wrist_2_link"),
    ("upper_arm_link", "wrist_3_link"),
    ("base_link", "gripper"),
    ("shoulder_link", "gripper"),
    ("upper_arm_link", "gripper"),
    ("forearm_link", "gripper"),
    ("base_link", "elbow"),
    ("shoulder_link", "elbow"),
    ("elbow", "gripper"),
    ("base_link", "forearm_collar"),
    ("shoulder_link", "forearm_collar"),
    ("forearm_collar", "gripper"),
    ("shoulder_yoke", "wrist_1_link"),
    ("shoulder_yoke", "wrist_2_link"),
    ("shoulder_yoke", "wrist_3_link"),
    ("shoulder_yoke", "gripper"),
    ("elbow_yoke", "wrist_1_link"),
    ("elbow_yoke", "wrist_2_link"),
    ("elbow_yoke", "wrist_3_link"),
    ("elbow_yoke", "gripper"),
)

# The cell bounds, each stored as the half-space that is FREE ({p: n.p >= d}).
# One plane: the floor.  Re-measure before adding walls back.
WALLS = (
    ("plane-z", np.array([0.0, 0.0, 1.0]), -0.55),
)

# Base-pose uncertainty in metres, applied by inflating the MOVER's geometry
# (uncertainty.inflate) in every cross-arm, wall and box check.  One arm
# carries it: it is the two bases' RELATIVE pose, counted once.  Self-pairs
# and the mover's own stand stay nominal -- a rigid base perturbation cannot
# change one arm's internal distances or move it relative to its own mount.
# Module state so it need not thread through every call.
_UNCERTAINTY = 0.0


def set_uncertainty(u=0.0):
    global _UNCERTAINTY
    _UNCERTAINTY = float(u)


def uncertainty():
    return _UNCERTAINTY


def _radii_2d(radii, M):
    radii = np.asarray(radii, float)
    return radii[None, :] if radii.ndim == 1 else radii


# Structure, not moving body: exempt from the wall check (the base capsule
# dips below z=0 by construction of the mount).
WALL_EXEMPT_LINKS = ("base_link",)


@lru_cache(maxsize=1)
def wall_arrays():
    return (np.stack([n for _, n, _ in WALLS]),
            np.array([d for _, _, d in WALLS], float))


@lru_cache(maxsize=1)
def wall_exempt_mask():
    return np.array([link in WALL_EXEMPT_LINKS for link in LINKS])


# Group toggles: the stands follow their arm and are trusted; the walls are
# hand-measured constants; the obstacles are whatever is on the bench today.
_WALLS_ON = True
_STANDS_ON = True
_OBSTACLES_ON = True


def set_walls(on=True):
    global _WALLS_ON
    _WALLS_ON = bool(on)


def set_stands(on=True):
    global _STANDS_ON
    _STANDS_ON = bool(on)


def set_obstacles(on=True):
    global _OBSTACLES_ON
    _OBSTACLES_ON = bool(on)


def walls_enabled():
    return _WALLS_ON


def stands_enabled():
    return _STANDS_ON


def obstacles_enabled():
    return _OBSTACLES_ON


@lru_cache(maxsize=1)
def wall_labels():
    return tuple(f"{link} vs {name}" for link in LINKS for name, _, _ in WALLS)


def wall_gaps_along(P0, P1, radii):
    """(M, n_links * W) surface gap of every link capsule against every wall.
    A disabled group keeps its columns at +inf so labels stay aligned."""
    if not _WALLS_ON:
        return np.full((len(P0), len(LINKS) * len(WALLS)), np.inf)
    N, D = wall_arrays()
    proj = np.minimum(np.einsum("mli,wi->mlw", P0, N),
                      np.einsum("mli,wi->mlw", P1, N))
    gaps = proj - D[None, None, :] - _radii_2d(radii, len(P0))[:, :, None]
    gaps[:, wall_exempt_mask(), :] = np.inf
    return gaps.reshape(len(P0), -1)


def wall_gaps(robot, q=None):
    """(n_links * W,) wall gaps for ONE arm (the oracle only walls the mover)."""
    if q is not None:
        robot.set_arm(q)
    P0, P1, R = robot.capsule_arrays()
    return wall_gaps_along(P0[None], P1[None], R)[0]


# Stands: a box under each arm, top face on the base, STAND_LENGTH deep (a
# fixed depth cannot invert, wherever the bases end up).
STAND_LENGTH = 1.0


def _stands():
    r = LINK_CAPSULES["base_link"][2] * 1.15
    bases = {"stand-A": np.zeros(3), "stand-B": np.asarray(BASE_OFFSET[:3], float)}
    return tuple(
        (name, np.array([x - r, y - r, z - STAND_LENGTH]),
         np.array([x + r, y + r, z]))
        for name, (x, y, z) in bases.items())


STANDS = _stands()

# Loose kit on the bench, placed against stand-B so it follows BASE_OFFSET.
OBSTACLE_SIZE = np.array([0.10, 0.29, 0.08])


def _obstacles():
    _, lo, hi = next(box for box in STANDS if box[0] == "stand-B")
    centre_y = 0.5 * (lo[1] + hi[1])
    near = np.array([hi[0], centre_y - OBSTACLE_SIZE[1] / 2,
                     hi[2] - OBSTACLE_SIZE[2]])
    return (("obstacle-1", near, near + OBSTACLE_SIZE),)


OBSTACLES = _obstacles()

IGNORED_BOX_COLLISIONS: set = {("base_link", "obstacle-1")}


def ignore_box_collision(link, box_name):
    """Ignore the (link, box_name) pair in every later box check."""
    IGNORED_BOX_COLLISIONS.add((link, box_name))


@lru_cache(maxsize=1)
def all_boxes():
    return STANDS + OBSTACLES


BOX_OWNER = {"stand-A": "A", "stand-B": "B"}


BOX_EXEMPT_LINKS = ("base_link",)


@lru_cache(maxsize=1)
def box_labels():
    return tuple(f"{link} vs {name}" for link in LINKS for name, _, _ in all_boxes())


@lru_cache(maxsize=1)
def box_exempt_mask():
    return np.array([link in BOX_EXEMPT_LINKS for link in LINKS])


def _point_box_distance(points, lo, hi):
    outside = np.maximum(np.maximum(lo - points, points - hi), 0.0)
    return np.linalg.norm(outside, axis=-1)


BOX_NEAR = 0.35        # where the face bound stops being trusted
BOX_SEARCH_STEPS = 25  # ternary steps; distance to a convex set is convex in t


def _face_bound(P0, P1, lo, hi):
    """(M, L) lower bound on shaft-to-box distance from the six half-spaces.
    Tight facing a face, pessimistic near a corner: a FILTER, never the answer."""
    return np.stack(
        [np.minimum(lo[k] - P0[..., k], lo[k] - P1[..., k]) for k in range(3)] +
        [np.minimum(P0[..., k] - hi[k], P1[..., k] - hi[k]) for k in range(3)],
        axis=-1).max(axis=-1)


def _exact_segment_box(P0, P1, lo, hi):
    if not len(P0):
        return np.zeros(0)
    span = P1 - P0
    a, b = np.zeros(len(P0)), np.ones(len(P0))
    for _ in range(BOX_SEARCH_STEPS):
        third = (b - a) / 3.0
        m1, m2 = a + third, b - third
        closer = (_point_box_distance(P0 + span * m1[:, None], lo, hi)
                  < _point_box_distance(P0 + span * m2[:, None], lo, hi))
        b = np.where(closer, m2, b)
        a = np.where(closer, a, m1)
    return _point_box_distance(P0 + span * (0.5 * (a + b))[:, None], lo, hi)


def box_gaps_along(P0, P1, radii, nominal=None, mover=None):
    """(M, n_links * n_boxes) gaps against every box: exact within BOX_NEAR,
    the cheap face bound elsewhere.  P0/P1/radii are the mover's (possibly
    inflated) capsules; `nominal` is its (P0, P1, radii) before inflation,
    used for the mover's OWN stand -- base and stand move together, so the
    uncertainty cannot close that gap."""
    boxes = all_boxes()
    if not boxes:
        return np.zeros((len(P0), 0))
    stands = {name for name, _, _ in STANDS}
    exempt = box_exempt_mask()
    out = []
    for name, lo, hi in boxes:
        is_stand = name in stands
        if not (_STANDS_ON if is_stand else _OBSTACLES_ON):
            out.append(np.full((len(P0), len(LINKS)), np.inf))
            continue
        own = is_stand and mover is not None and BOX_OWNER.get(name) == mover
        A0, A1, R = (nominal if (own and nominal is not None)
                     else (P0, P1, radii))
        gaps = _face_bound(A0, A1, lo, hi)
        near = gaps < BOX_NEAR
        if near.any():
            gaps = gaps.copy()
            gaps[near] = _exact_segment_box(A0[near], A1[near], lo, hi)
        gaps = gaps - _radii_2d(R, len(A0))
        if is_stand:
            gaps[:, exempt] = np.inf
        for (link, box_name) in IGNORED_BOX_COLLISIONS:
            if box_name == name and link in LINKS:
                gaps[:, LINKS.index(link)] = np.inf
        out.append(gaps)
    return np.stack(out, axis=-1).reshape(len(P0), -1)


@lru_cache(maxsize=2)
def all_labels(ignore_forearm_pair=False):
    """Every oracle column: capsule pairs, then walls, then boxes."""
    return tuple(pair_index(ignore_forearm_pair)[3]) + wall_labels() + box_labels()


WRIST_LINKS = ("wrist_1_link", "wrist_2_link", "wrist_3_link", "gripper")


@lru_cache(maxsize=2)
def wrist_free_mask(ignore_forearm_pair=False):
    """(n_columns,) True for columns no wrist preset can change.  Early
    reject only, never an acceptance."""
    return np.array([not any(w in label for w in WRIST_LINKS)
                     for label in all_labels(ignore_forearm_pair)])


@lru_cache(maxsize=1)
def link_capsules():
    return {link: (np.array(p0, float), np.array(p1, float), r)
            for link, (p0, p1, r) in LINK_CAPSULES.items()}


@lru_cache(maxsize=2)
def pair_index(ignore_forearm_pair=False):
    """(ia, ib, self_mask, labels) for every checked capsule pair: the full
    cross-arm product, then mover-side SELF_PAIRS (ib into the mover's table,
    flagged by self_mask).  Built once; a query becomes fancy indexing plus
    one segment_gaps call."""
    ia, ib, self_mask, labels = [], [], [], []
    for i, la in enumerate(LINKS):
        for j, lb in enumerate(LINKS):
            if ignore_forearm_pair and la == lb == FOREARM_LINK:
                continue
            ia.append(i)
            ib.append(j)
            self_mask.append(False)
            labels.append(f"{la} vs other/{lb}")
    for la, lb in SELF_PAIRS:
        ia.append(LINKS.index(la))
        ib.append(LINKS.index(lb))
        self_mask.append(True)
        labels.append(f"{la} vs self/{lb}")
    return (np.array(ia), np.array(ib), np.array(self_mask), labels)


def segment_gaps(P0, P1, Q0, Q1, Ra, Rb):
    """Surface gap for N capsule pairs at once -- the vectorised twin of
    cylinder.segment_closest_points (same clamped solve, same order)."""
    u, v, w = P1 - P0, Q1 - Q0, P0 - Q0
    dot = lambda x, y: np.einsum("ij,ij->i", x, y)
    a, b, c = dot(u, u), dot(u, v), dot(v, v)
    d, e = dot(u, w), dot(v, w)
    den = a * c - b * b
    s = np.where(den > 1e-12,
                 np.clip((b * e - c * d) / np.where(den > 1e-12, den, 1.0), 0, 1), 0.0)
    t = np.where(c > 1e-12,
                 np.clip((b * s + e) / np.where(c > 1e-12, c, 1.0), 0, 1), 0.0)
    s = np.where(a > 1e-12,
                 np.clip((b * t - d) / np.where(a > 1e-12, a, 1.0), 0, 1), 0.0)
    sep = (P0 + s[:, None] * u) - (Q0 + t[:, None] * v)
    return np.sqrt(dot(sep, sep)) - Ra - Rb


# DH parameters (identical to the C++ fk_ur5e.hpp) -- the port's link frames
# are DH-style: each link frame sits at its PROXIMAL joint with the shaft
# along -x, which is exactly the frame the capsule table is written in.  The
# two long links' DH frames sit at the DISTAL joint, so they get shifted back
# by their own length (LINK_SHIFT).  The wrist frames are taken as the plain
# DH joint frames -- confirm their capsule placements against the source
# repo's URDF (see the module docstring).
DH_D = np.array([0.1625, 0.0, 0.0, 0.1333, 0.0997, 0.0996])
DH_A = np.array([0.0, -0.425, -0.3922, 0.0, 0.0, 0.0])
DH_ALPHA = np.array([np.pi / 2, 0.0, 0.0, np.pi / 2, -np.pi / 2, 0.0])
JOINT_LIMITS = tuple((-np.pi, np.pi) if j == 2 else (-2 * np.pi, 2 * np.pi)
                     for j in range(6))

# DH joint-frame index of each kinematic link, plus the local origin shift
# that moves the frame to the link's proximal joint.
_FRAME_OF = {"base_link": 0, "shoulder_link": 1, "upper_arm_link": 2,
             "forearm_link": 3, "wrist_1_link": 4, "wrist_2_link": 5,
             "wrist_3_link": 6}
_LINK_SHIFT = {"upper_arm_link": np.array([0.425, 0.0, 0.0]),
               "forearm_link": np.array([0.3922, 0.0, 0.0])}


@lru_cache(maxsize=1)
def chain():
    """(limits, local_p0, local_p1, radii) plus the per-link frame table."""
    for link, (parent, _) in FIXED_LINKS.items():
        if LINKS.index(parent) >= LINKS.index(link):
            raise ValueError(f"{link} must come after its parent {parent} in LINKS")
    caps = link_capsules()
    return (JOINT_LIMITS,
            np.array([caps[l][0] for l in LINKS]),
            np.array([caps[l][1] for l in LINKS]),
            np.array([caps[l][2] for l in LINKS]))


def _batch_dh(theta, d, a, alpha):
    """(M, 4, 4) DH transform for one row, batched over theta."""
    m = len(theta)
    ct, st = np.cos(theta), np.sin(theta)
    ca, sa = np.cos(alpha), np.sin(alpha)
    T = np.zeros((m, 4, 4))
    T[:, 0, 0], T[:, 0, 1], T[:, 0, 2], T[:, 0, 3] = ct, -st * ca, st * sa, a * ct
    T[:, 1, 0], T[:, 1, 1], T[:, 1, 2], T[:, 1, 3] = st, ct * ca, -ct * sa, a * st
    T[:, 2, 1], T[:, 2, 2], T[:, 2, 3] = sa, ca, d
    T[:, 3, 3] = 1.0
    return T


def batch_link_T(base_T, qs):
    """(M, n_links, 4, 4) world transform of every link, M configs at once."""
    qs = np.atleast_2d(np.asarray(qs, float))
    m = len(qs)
    frames = [np.broadcast_to(np.asarray(base_T, float), (m, 4, 4)).copy()]
    T = frames[0]
    for j in range(6):
        T = T @ _batch_dh(qs[:, j], DH_D[j], DH_A[j], DH_ALPHA[j])
        frames.append(T)
    out = np.empty((m, len(LINKS), 4, 4))
    for i, link in enumerate(LINKS):
        if link in FIXED_LINKS:
            parent, origin = FIXED_LINKS[link]
            out[:, i] = out[:, LINKS.index(parent)] @ origin
            continue
        F = frames[_FRAME_OF[link]]
        out[:, i] = F
        shift = _LINK_SHIFT.get(link)
        if shift is not None:
            out[:, i] = F.copy()
            out[:, i, :3, 3] += np.einsum("mij,j->mi", F[:, :3, :3], shift)
    return out


FOREARM_INDEX = LINKS.index(FOREARM_LINK)
FOREARM_PAIR_INDEX = FOREARM_INDEX * len(LINKS) + FOREARM_INDEX


def forearm_shafts(robot, qs):
    """(M, 3), (M, 3) world endpoints of the forearm shaft.  Pure."""
    qs = np.atleast_2d(np.asarray(qs, float))
    *_, local_p0, local_p1, _ = chain()
    T = batch_link_T(robot.base_T, qs)[:, FOREARM_INDEX]
    rot, pos = T[:, :3, :3], T[:, :3, 3]
    return (np.einsum("mij,j->mi", rot, local_p0[FOREARM_INDEX]) + pos,
            np.einsum("mij,j->mi", rot, local_p1[FOREARM_INDEX]) + pos)


def forearm_gaps(robot, qs, obstacle_cyl, radius=FOREARM_RADIUS):
    """(M,) surface gap between the forearm and a static cylinder.  Pure."""
    p0, p1 = forearm_shafts(robot, qs)
    b0, b1 = obstacle_cyl.get_endpoints()
    m = len(p0)
    return segment_gaps(p0, p1,
                        np.broadcast_to(b0, (m, 3)), np.broadcast_to(b1, (m, 3)),
                        np.zeros(m), np.full(m, radius + obstacle_cyl.radius))


def joint_limits(rid, j):
    """(lower, upper) of arm joint j.  `rid` is the Robot (the original took
    the yourdfpy model; the argument name is kept so maneuver.py fits)."""
    return rid.limits[j]


class Robot:
    """One UR5e: batched-FK state, a world placement, and its capsules."""

    ARM_JOINTS = (0, 1, 2, 3, 4, 5)
    WRIST_FIXED = np.array([-1.5708, 0.0, 0.0])

    def __init__(self, base_T=None, name="arm"):
        self.base_T = np.eye(4) if base_T is None else np.asarray(base_T, float)
        self.name = name
        self.limits = chain()[0]
        self._q = np.zeros(6)
        self._cache_key = None
        self._cache = None
        self._array_key = None
        self._arrays = None

    @property
    def rid(self):
        return self

    @property
    def q(self):
        """The configuration this arm is posed at -- the model IS the state."""
        return self._q.copy()

    def set_arm(self, q6):
        self._q = np.asarray(q6, float).ravel().copy()

    def link_T(self, link):
        return batch_link_T(self.base_T, self._q[None])[0, LINKS.index(link)]

    def world_capsules(self):
        """Every link capsule at the current config, in world coordinates.
        Memoized on the configuration."""
        key = self._q.tobytes()
        if key != self._cache_key:
            *_, local_p0, local_p1, radii = chain()
            T = batch_link_T(self.base_T, self._q[None])[0]
            rot, pos = T[:, :3, :3], T[:, :3, 3]
            P0 = np.einsum("lij,lj->li", rot, local_p0) + pos
            P1 = np.einsum("lij,lj->li", rot, local_p1) + pos
            self._cache = [(link, P0[i], P1[i], radii[i])
                           for i, link in enumerate(LINKS)]
            self._cache_key = key
        return self._cache

    def capsule_arrays(self):
        key = self._q.tobytes()
        if key != self._array_key:
            caps = self.world_capsules()
            self._arrays = (np.array([c[1] for c in caps]),
                            np.array([c[2] for c in caps]),
                            np.array([c[3] for c in caps]))
            self._array_key = key
        return self._arrays

    def forearm_cylinder(self):
        p0, p1, r = link_capsules()[FOREARM_LINK]
        T = self.link_T(FOREARM_LINK)
        return Cylinder.from_segment(T[:3, :3] @ p0 + T[:3, 3],
                                     T[:3, :3] @ p1 + T[:3, 3], r)


class DualRobot:
    """maneuver.py's `ctx`: the collision oracle and forearm geometry."""

    AWAY_Q = HOME_Q
    retract_rule = "gradient"
    approach_mode = "line"
    transit_planner = "rrt"      # or "ladder" for maneuver.transit
    transit_clearance = 0.010    # must equal maneuver.TRANSIT_CLEARANCE
    CREEP_RES = 0.01             # must equal maneuver.CREEP_RES

    def __init__(self, a, b):
        self.a, self.b = a, b
        self.contact_floor = None
        self.forearm_floor = None

    def contact_floor_at(self, mover, obstacle, q, clearance=0.0):
        """Per-pair floor for a retract starting at a multi-contact config: a
        pair that starts in contact may stay in contact but not get worse."""
        return np.minimum(self.gaps(mover, obstacle, q, True), clearance)

    def _forearm_cyl(self, robot):
        return robot.forearm_cylinder()

    def gaps(self, mover, obstacle, q, ignore_forearm_pair=False):
        return self.gaps_along(mover, obstacle, [q], ignore_forearm_pair)[0]

    def gaps_along(self, mover, obstacle, qs, ignore_forearm_pair=False):
        """(M, n_columns) gaps for M configs of `mover` against the static
        `obstacle`, the walls and the boxes, in all_labels() order.  Fully
        batched, and pure -- it never moves the arm."""
        ia, ib, self_mask, _ = pair_index(ignore_forearm_pair)
        qs = np.atleast_2d(np.asarray(qs, float))
        *_, local_p0, local_p1, radii = chain()

        T = batch_link_T(mover.base_T, qs)
        rot, pos = T[..., :3, :3], T[..., :3, 3]
        mP0 = np.einsum("mlij,lj->mli", rot, local_p0) + pos
        mP1 = np.einsum("mlij,lj->mli", rot, local_p1) + pos
        oP0, oP1, oR = obstacle.capsule_arrays()
        # the mover's uncertainty shell; self-pairs keep the nominal geometry
        iP0, iP1, iR = inflate(mP0, mP1, radii, uncertainty())

        sm = self_mask[None, :, None]
        P0 = np.where(sm, mP0[:, ia], iP0[:, ia])
        P1 = np.where(sm, mP1[:, ia], iP1[:, ia])
        Q0 = np.where(sm, mP0[:, ib], oP0[ib])
        Q1 = np.where(sm, mP1[:, ib], oP1[ib])
        Ra = np.where(self_mask[None, :], radii[ia][None, :], iR[:, ia])
        Rb = np.where(self_mask[None, :], radii[ib][None, :], oR[ib][None, :])
        R = Ra + Rb

        n = qs.shape[0] * len(ia)
        flat = segment_gaps(P0.reshape(n, 3), P1.reshape(n, 3),
                            Q0.reshape(n, 3), Q1.reshape(n, 3),
                            np.zeros(n),
                            np.broadcast_to(R, (qs.shape[0], len(ia))).reshape(n))
        return np.concatenate([flat.reshape(qs.shape[0], len(ia)),
                               wall_gaps_along(iP0, iP1, iR),
                               box_gaps_along(iP0, iP1, iR,
                                              (mP0, mP1, radii), mover.name)],
                              axis=1)

    def _floor_vector(self, clearance, n_cols, ignore_forearm_pair=False):
        """`clearance` everywhere, except the forearm/forearm pair, held only
        to forearm_floor while one is set (a pre-contact must not be refused
        for standing exactly where it was asked to stand)."""
        if self.forearm_floor is None or ignore_forearm_pair:
            return float(clearance)
        t = np.full(int(n_cols), float(clearance))
        t[FOREARM_PAIR_INDEX] = min(float(clearance), float(self.forearm_floor))
        return t

    def _blocked(self, mover, obstacle, q, clearance=0.01,
                 ignore_forearm_pair=False):
        g = self.gaps(mover, obstacle, q, ignore_forearm_pair)
        if ignore_forearm_pair and self.contact_floor is not None:
            return bool((g < self.contact_floor).any())
        return bool((g < self._floor_vector(clearance, g.size,
                                            ignore_forearm_pair)).any())

    def blocked_along(self, mover, obstacle, qs, clearance=0.01,
                      ignore_forearm_pair=False):
        if not len(qs):
            return False
        g = self.gaps_along(mover, obstacle, qs, ignore_forearm_pair)
        return bool((g < self._floor_vector(clearance, g.shape[-1],
                                            ignore_forearm_pair)).any())

    def _why_blocked(self, mover, obstacle, q, clearance=0.01,
                     ignore_forearm_pair=False):
        g = self.gaps(mover, obstacle, q, ignore_forearm_pair)
        over = g - self._floor_vector(clearance, g.size, ignore_forearm_pair)
        if over.min() >= 0.0:
            return "clear"
        k = int(np.argmin(over))
        label = all_labels(ignore_forearm_pair)[k]
        label = label.replace("vs other/", f"vs {obstacle.name}/")
        return f"{mover.name}/{label} gap={g[k]:+.4f}"
