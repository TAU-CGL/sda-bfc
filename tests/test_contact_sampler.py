import numpy as np
import pytest

from sda_bfc import ContactSampler, ContactParams, CylinderPose, UR5e

UR5E_A = [0.0, -0.425, -0.3922, 0.0, 0.0, 0.0]
INTERIOR_MARGIN = 0.02


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


def segment_closest(p1, q1, p2, q2):
    d1, d2, r = q1 - p1, q2 - p2, p1 - p2
    a, e, f = d1 @ d1, d2 @ d2, d2 @ r
    c, b = d1 @ r, d1 @ d2
    denom = a * e - b * b
    s = np.clip((b * f - c * e) / denom, 0.0, 1.0) if denom > 1e-12 else 0.0
    t = np.clip((b * s + f) / e, 0.0, 1.0)
    s = np.clip((b * t - c) / a, 0.0, 1.0)
    return np.linalg.norm(p1 + s * d1 - p2 - t * d2), s, t


@pytest.mark.parametrize("idx_a,idx_b", [(3, 3), (2, 3), (2, 2)])
@pytest.mark.parametrize("seed", [0, 1, 2])
def test_generated_contact_touches_on_finite_cylinders(idx_a, idx_b, seed):
    robot = UR5e()
    X = placement(seed)
    params = ContactParams()
    params.max_restarts = 20000  # (2, 2) contacts are rare at far placements
    pose = ContactSampler(robot, X, idx_a, idx_b, params).sample(seed)
    assert pose is not None

    A = robot.get_cylinder_transform(idx_a, pose.q_a, robot.get_link_z_offset(idx_a))
    B = X @ robot.get_cylinder_transform(idx_b, pose.q_b, robot.get_link_z_offset(idx_b))
    ra, rb = robot.get_cylinder_radius(idx_a), robot.get_cylinder_radius(idx_b)
    sd = CylinderPose.from_se3(A, ra).signed_distance(CylinderPose.from_se3(B, rb))
    assert sd == pytest.approx(0.0, abs=1e-9)

    dist, s, t = segment_closest(A[:3, 3], A[:3, 3] - UR5E_A[idx_a - 1] * A[:3, 2],
                                 B[:3, 3], B[:3, 3] - UR5E_A[idx_b - 1] * B[:3, 2])
    assert dist - (ra + rb) == pytest.approx(0.0, abs=1e-6)
    assert min(s, 1.0 - s) * abs(UR5E_A[idx_a - 1]) >= INTERIOR_MARGIN
    assert min(t, 1.0 - t) * abs(UR5E_A[idx_b - 1]) >= INTERIOR_MARGIN


def test_zero_length_link_rejected():
    with pytest.raises(Exception):
        ContactSampler(UR5e(), np.eye(4), 1, 3)
