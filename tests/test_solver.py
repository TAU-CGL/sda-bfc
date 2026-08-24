import numpy as np
import pytest

import test_touch_poses as touch
from sda_bfc import (CylinderPose, SolverAdam, SolverAnnealingLP,
                     SolverNewton, SolverSMC, UR5e)


def build_solver():
    robot = UR5e()
    radius = robot.get_link_radius(touch.RADII_LINK_INDEX)
    As = [robot.get_cylinder_transform(touch.DH_LINK_INDEX, np.array(q1))
          for q1, _ in touch.TOUCH_POSES]
    Bs = [robot.get_cylinder_transform(touch.DH_LINK_INDEX, np.array(q2))
          for _, q2 in touch.TOUCH_POSES]
    return SolverNewton(As, Bs, radius, radius), As, Bs, radius


def test_implicit_touch_condition_consistent_with_signed_distance():
    rng = np.random.default_rng(0)
    for _ in range(100):
        u = rng.normal(size=3)
        u /= np.linalg.norm(u)
        cp = CylinderPose(rng.uniform(-2.0, 2.0, size=3), u, rng.uniform(0.01, 0.5))
        other_r = rng.uniform(0.01, 0.5)
        sd = cp.signed_distance(other_r)
        f = cp.implicit_touch_condition(other_r)
        denom = u[0] ** 2 + u[1] ** 2
        s = cp.r + other_r
        assert f == pytest.approx(denom * ((sd + s) ** 2 - s ** 2), abs=1e-12)


def test_solver_zeros_touch_constraints():
    solver, As, Bs, radius = build_solver()
    X = solver.solve_multistart()
    for At, Bt in zip(As, Bs):
        cylinder = CylinderPose.from_se3(np.linalg.inv(At) @ X @ Bt, radius)
        assert cylinder.signed_distance(radius) == pytest.approx(0.0, abs=2e-3)


def test_annealing_lp_recovers_base_offset():
    robot = UR5e()
    radius = robot.get_link_radius(touch.RADII_LINK_INDEX)
    As = [robot.get_cylinder_transform(touch.DH_LINK_INDEX, np.array(q1))
          for q1, _ in touch.TOUCH_POSES]
    Bs = [robot.get_cylinder_transform(touch.DH_LINK_INDEX, np.array(q2))
          for _, q2 in touch.TOUCH_POSES]
    X = SolverAnnealingLP(As, Bs, radius, radius).solve()

    # ------------------------------------------------------------------
    # Ground truth: used ONLY for verification below, never in the solve.
    # ------------------------------------------------------------------
    X_gt = touch.base_transform()

    translation_error = np.linalg.norm(X[:3, 3] - X_gt[:3, 3])
    R_delta = X[:3, :3].T @ X_gt[:3, :3]
    rotation_error = np.degrees(np.arccos(np.clip((np.trace(R_delta) - 1.0) / 2.0, -1.0, 1.0)))
    assert translation_error < 0.01
    assert rotation_error < 1.0


def test_adam_recovers_base_offset():
    solver, As, Bs, radius = build_solver()
    X = SolverAdam(As, Bs, radius, radius).solve_multistart()

    # ------------------------------------------------------------------
    # Ground truth: used ONLY for verification below, never in the solve.
    # ------------------------------------------------------------------
    X_gt = touch.base_transform()

    translation_error = np.linalg.norm(X[:3, 3] - X_gt[:3, 3])
    R_delta = X[:3, :3].T @ X_gt[:3, :3]
    rotation_error = np.degrees(np.arccos(np.clip((np.trace(R_delta) - 1.0) / 2.0, -1.0, 1.0)))
    assert translation_error < 0.01
    assert rotation_error < 1.0


def test_smc_recovers_base_offset():
    robot = UR5e()
    radius = robot.get_link_radius(touch.RADII_LINK_INDEX)
    As = [robot.get_cylinder_transform(touch.DH_LINK_INDEX, np.array(q1))
          for q1, _ in touch.TOUCH_POSES]
    Bs = [robot.get_cylinder_transform(touch.DH_LINK_INDEX, np.array(q2))
          for _, q2 in touch.TOUCH_POSES]
    X = SolverSMC(As, Bs, radius, radius).solve()

    # ------------------------------------------------------------------
    # Ground truth: used ONLY for verification below, never in the solve.
    # ------------------------------------------------------------------
    X_gt = touch.base_transform()

    translation_error = np.linalg.norm(X[:3, 3] - X_gt[:3, 3])
    R_delta = X[:3, :3].T @ X_gt[:3, :3]
    rotation_error = np.degrees(np.arccos(np.clip((np.trace(R_delta) - 1.0) / 2.0, -1.0, 1.0)))
    assert translation_error < 0.01
    assert rotation_error < 1.0


def test_solver_recovers_base_offset():
    solver, *_ = build_solver()
    X = solver.solve_multistart()

    # ------------------------------------------------------------------
    # Ground truth: used ONLY for verification below, never in the solve.
    # ------------------------------------------------------------------
    X_gt = touch.base_transform()

    translation_error = np.linalg.norm(X[:3, 3] - X_gt[:3, 3])
    R_delta = X[:3, :3].T @ X_gt[:3, :3]
    rotation_error = np.degrees(np.arccos(np.clip((np.trace(R_delta) - 1.0) / 2.0, -1.0, 1.0)))
    assert translation_error < 0.01
    assert rotation_error < 1.0
