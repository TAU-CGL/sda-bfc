from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("mujoco")

from sda_bfc import ContactSampler, CylinderPose, UR5e
from sda_bfc.collision import CollisionChecker, sample_valid_contact

MJCF = str(Path(__file__).resolve().parent.parent
           / "assets" / "universal_robots_ur5e" / "ur5e.xml")


def placement(seed):
    rng = np.random.default_rng(seed)
    yaw = rng.uniform(-np.pi, np.pi)
    heading = rng.uniform(-np.pi, np.pi)
    dist = rng.uniform(0.55, 0.8)
    X = np.eye(4)
    X[:3, :3] = [[np.cos(yaw), -np.sin(yaw), 0],
                 [np.sin(yaw), np.cos(yaw), 0], [0, 0, 1]]
    X[:3, 3] = [dist * np.cos(heading), dist * np.sin(heading), 0.0]
    return X


@pytest.mark.parametrize("idx_a,idx_b", [(3, 3), (2, 3)])
def test_valid_contact_touches_and_is_collision_free(idx_a, idx_b):
    robot = UR5e()
    X = placement(0)
    sampler = ContactSampler(robot, X, idx_a, idx_b)
    checker = CollisionChecker(X, idx_a, idx_b, mjcf_path=MJCF)
    result = sample_valid_contact(sampler, checker, seed=0)
    assert result is not None
    q_a, q_b = result
    assert checker.first_violation(q_a, q_b) is None

    A = robot.get_cylinder_transform(idx_a, q_a, robot.get_link_z_offset(idx_a))
    B = X @ robot.get_cylinder_transform(idx_b, q_b, robot.get_link_z_offset(idx_b))
    sd = CylinderPose.from_se3(A, robot.get_cylinder_radius(idx_a)) \
        .signed_distance(CylinderPose.from_se3(B, robot.get_cylinder_radius(idx_b)))
    assert sd == pytest.approx(0.0, abs=1e-9)


def test_overlapping_bases_are_flagged():
    X = np.eye(4)
    X[0, 3] = 0.05  # bases interpenetrate regardless of joint config
    checker = CollisionChecker(X, 3, 3, mjcf_path=MJCF)
    violation = checker.first_violation(np.zeros(6), np.zeros(6))
    assert violation is not None
