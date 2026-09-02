"""Collision validation for sampled contact poses.

ContactSampler guarantees the chosen cylinders touch but knows nothing about
the rest of either arm.  CollisionChecker rebuilds the two-arm scene with the
full MuJoCo collision model (vendored mujoco_menagerie UR5e MJCF) and flags
any pose where a link pair other than the intended touching one is closer
than `margin` -- cross-arm penetrations and self-collisions alike.

This module is intentionally not imported from __init__ so the C++ core
stays free of the mujoco dependency.
"""

import os

import mujoco
import numpy as np

DEFAULT_MJCF = os.environ.get("SDA_BFC_UR5E_MJCF",
                              "assets/universal_robots_ur5e/ur5e.xml")
PREFIXES = ("r1/", "r2/")
# Cylinder-axis index (getCylinderTransform) -> MJCF body carrying that link.
BODY_FOR_AXIS = {2: "upper_arm_link", 3: "forearm_link"}


def _build_model(X, mjcf_path, margin):
    spec = mujoco.MjSpec()
    spec.option.integrator = mujoco.mjtIntegrator.mjINT_IMPLICITFAST
    for prefix, T in zip(PREFIXES, (np.eye(4), X)):
        arm = mujoco.MjSpec.from_file(mjcf_path)
        for key in list(arm.keys):
            arm.delete(key)
        quat = np.empty(4)
        mujoco.mju_mat2Quat(quat, np.asarray(T[:3, :3], dtype=float).ravel())
        frame = spec.worldbody.add_frame(pos=T[:3, 3], quat=quat)
        spec.attach(arm, prefix=prefix, frame=frame)
    model = spec.compile()
    model.geom_margin[:] = margin  # detect proximity, not only penetration
    return model


class CollisionChecker:
    """Validates q = (q_a, q_b) against the full collision geometry.

    A pose is valid when every contact reported by MuJoCo either involves
    the intended touching pair (r1 link idx_a, r2 link idx_b) or is farther
    than `margin` apart.
    """

    def __init__(self, X, idx_a, idx_b, mjcf_path=DEFAULT_MJCF, margin=5e-3):
        self.idx_a, self.idx_b, self.margin = idx_a, idx_b, margin
        self.model = _build_model(X, mjcf_path, margin)
        self.data = mujoco.MjData(self.model)
        self.qadr = {
            prefix: np.array([self.model.joint(j).qposadr[0]
                              for j in range(self.model.njnt)
                              if self.model.joint(j).name.startswith(prefix)])
            for prefix in PREFIXES}
        self.allowed = frozenset((
            self.model.body(f"r1/{BODY_FOR_AXIS[idx_a]}").id,
            self.model.body(f"r2/{BODY_FOR_AXIS[idx_b]}").id))

    def first_violation(self, q_a, q_b):
        """None if valid, else (body_name, body_name, distance)."""
        self.data.qpos[self.qadr["r1/"]] = q_a
        self.data.qpos[self.qadr["r2/"]] = q_b
        mujoco.mj_kinematics(self.model, self.data)
        mujoco.mj_collision(self.model, self.data)
        for c in self.data.contact[:self.data.ncon]:
            bodies = (self.model.geom_bodyid[c.geom1],
                      self.model.geom_bodyid[c.geom2])
            if frozenset(bodies) == self.allowed:
                continue
            if c.dist < self.margin:
                return (self.model.body(bodies[0]).name,
                        self.model.body(bodies[1]).name, c.dist)
        return None

    def is_valid(self, q_a, q_b):
        return self.first_violation(q_a, q_b) is None


def sample_valid_contact(sampler, checker, seed=0, max_attempts=200,
                         distal_attempts=20, joint_range=np.pi):
    """Rejection-sample a touching pose that is also collision-valid.

    The distal joints (idx.. of each arm) cannot move the contact, so an
    invalid pose is first retried with fresh distal joints before a new
    contact is sampled.  Returns (q_a, q_b) or None.
    """
    rng = np.random.default_rng(seed)
    for _ in range(max_attempts):
        pose = sampler.sample(int(rng.integers(1 << 31)))
        if pose is None:
            continue
        q_a, q_b = np.array(pose.q_a), np.array(pose.q_b)
        for _ in range(distal_attempts):
            if checker.is_valid(q_a, q_b):
                return q_a, q_b
            q_a[checker.idx_a:] = rng.uniform(-joint_range, joint_range,
                                              6 - checker.idx_a)
            q_b[checker.idx_b:] = rng.uniform(-joint_range, joint_range,
                                              6 - checker.idx_b)
    return None
