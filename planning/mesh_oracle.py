"""Simulation oracle backed by the UR5e URDF meshes (FCL).

Two mesh sources:
  * "collision" -- the URDF collision shells.  Padded by design (forearm
    tube ~54 mm effective radius vs the real 37.5 mm), conical toward the
    elbow; fair for collision but noisy as a touch surface.
  * "visual" -- the display meshes, which for the UR5e are the true
    surfaces (forearm tube measured 37.5 +- 0.4 mm, matching the tape-
    measured real robot).

The felt touch surface defines the effective touch radius, which the
calibration solvers must use; `touch_radius` reports it per source.
"""

import functools

import numpy as np
import trimesh.collision

from sda_bfc import TwoArmScene, fold_config

from .oracle import (DYNAMIC_ARM, STATIC_ARM, CollisionOracle, MoveOutcome,
                     TouchReport)

GUARD_STEP = 0.02
BISECTION_ITERATIONS = 40
RZ_PI = np.diag([-1.0, -1.0, 1.0, 1.0])
TOUCH_PAIR = (3, 3)

# Effective forearm tube radius of each mesh source, measured by sampling
# the tube section's surface (see docstring).
MESH_SOURCES = {
    "collision": {"touch_radius": 0.0537, "touch_tolerance": 0.014},
    "visual": {"touch_radius": 0.0375, "touch_tolerance": 0.005},
}


def link_of_node(node):
    prefixes = {"base": 0, "shoulder": 1, "upperarm": 2, "forearm": 3,
                "wrist1": 4, "wrist2": 5, "wrist3": 5}
    return prefixes[node.split(".")[0].split("_")[0]]


@functools.lru_cache(maxsize=1)
def load_urdf():
    import yourdfpy
    from robot_descriptions import ur5e_description
    from robot_descriptions._xacro import get_urdf_path
    return yourdfpy.URDF.load(get_urdf_path(ur5e_description),
                              build_collision_scene_graph=True,
                              load_collision_meshes=True)


# Backwards-compatible alias used by measurement scripts.
load_collision_urdf = load_urdf


class MeshArm:
    """One arm's meshes in a CollisionManager, posed by the shared URDF.

    Mesh unit scale (visual DAEs are in millimeters) is baked into the mesh
    copies, because FCL transforms must be rigid."""

    def __init__(self, urdf, scene, base):
        self.urdf = urdf
        self.scene = scene
        self.base = base @ RZ_PI
        self.manager = trimesh.collision.CollisionManager()
        self.scales = {}
        for node in scene.graph.nodes_geometry:
            T_node, geometry = scene.graph.get(node)
            scale = np.cbrt(np.linalg.det(T_node[:3, :3]))
            self.scales[node] = scale
            mesh = scene.geometry[geometry]
            if abs(scale - 1.0) > 1e-9:
                mesh = mesh.copy()
                mesh.apply_scale(scale)
            self.manager.add_object(node, mesh)

    def pose(self, q):
        self.urdf.update_cfg(np.asarray(q))
        for node in self.scene.graph.nodes_geometry:
            T_node, _ = self.scene.graph.get(node)
            rigid = np.eye(4)
            rigid[:3, :3] = T_node[:3, :3] / self.scales[node]
            rigid[:3, 3] = T_node[:3, 3]
            self.manager.set_transform(node, self.base @ rigid)


class MeshOracle(CollisionOracle):
    def __init__(self, robot, x_true, workcell=None, recorder=None,
                 mesh_source="collision"):
        self.robot = robot
        self.x_true = x_true
        self.workcell = workcell
        self.recorder = recorder
        self.capsule_scene = TwoArmScene(robot, x_true)
        source = MESH_SOURCES[mesh_source]
        self.touch_radius = source["touch_radius"]
        self._touch_tolerance = source["touch_tolerance"]
        urdf = load_urdf()
        scene = urdf.collision_scene if mesh_source == "collision" else urdf.scene
        self.arms = {STATIC_ARM: MeshArm(urdf, scene, np.eye(4)),
                     DYNAMIC_ARM: MeshArm(urdf, scene, x_true)}
        self.q = {STATIC_ARM: fold_config(0.0), DYNAMIC_ARM: fold_config(0.0)}
        for arm in self.arms:
            self.arms[arm].pose(self.q[arm])

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
            self._commit(arm, candidate)
        return MoveOutcome(True, None)

    def classify_touch(self):
        colliding, names = self._mesh_contact_names()
        if not colliding:
            return None
        pair = self._pair_from_names(names)
        return TouchReport(pair=pair, interior=self._touch_is_usable(pair))

    def _touch_is_usable(self, pair, margin=0.05):
        """A usable touch is interior on the tube AND consistent with the
        effective touch radius -- rejecting end-bulge contacts whose axis
        distance disagrees with the tube surface."""
        if pair != TOUCH_PAIR:
            return False
        qA, qB = self.q[STATIC_ARM], self.q[DYNAMIC_ARM]
        clearance = next(pc.clearance
                         for pc in self.capsule_scene.cross_clearances(qA, qB)
                         if (pc.i, pc.j) == TOUCH_PAIR)
        axis_distance = clearance + 2.0 * self.capsule_scene.link_radius(3)
        if abs(axis_distance - 2.0 * self.touch_radius) > self._touch_tolerance:
            return False
        return self.capsule_scene.contact_interior(qA, qB, 3, 3, margin)

    def _in_contact(self, arm, candidate):
        self.arms[arm].pose(candidate)
        in_collision = self.arms[STATIC_ARM].manager.in_collision_other(
            self.arms[DYNAMIC_ARM].manager)
        self.arms[arm].pose(self.q[arm])
        return in_collision or self._hits_workcell(arm, candidate)

    def _hits_workcell(self, arm, candidate):
        if self.workcell is None:
            return False
        base = np.eye(4) if arm == STATIC_ARM else self.x_true
        R, t = base[:3, :3], base[:3, 3]
        for i, capsule in enumerate(
                self.capsule_scene.capsules(np.asarray(candidate))):
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
        self._commit(arm, hi)  # rest on the contact side so contact is queryable

    def _current_contact(self):
        colliding, names = self._mesh_contact_names()
        if colliding:
            return self._pair_from_names(names)
        return ("env", "workcell")

    def _mesh_contact_names(self):
        self.arms[STATIC_ARM].pose(self.q[STATIC_ARM])
        self.arms[DYNAMIC_ARM].pose(self.q[DYNAMIC_ARM])
        return self.arms[STATIC_ARM].manager.in_collision_other(
            self.arms[DYNAMIC_ARM].manager, return_names=True)

    @staticmethod
    def _pair_from_names(names):
        name_a, name_b = sorted(names)[0]
        return (link_of_node(name_a), link_of_node(name_b))

    def _commit(self, arm, q):
        self.q[arm] = np.asarray(q, float)
        self.arms[arm].pose(self.q[arm])
        if self.recorder is not None:
            self.recorder(self.q[STATIC_ARM].copy(),
                          self.q[DYNAMIC_ARM].copy())
