import numpy as np

from sda_bfc import UR5e

UR5E_D = [0.1625, 0.0, 0.0, 0.1333, 0.0997, 0.0996]
UR5E_A = [0.0, -0.425, -0.3922, 0.0, 0.0, 0.0]
UR5E_ALPHA = [np.pi / 2, 0.0, 0.0, np.pi / 2, -np.pi / 2, 0.0]

Z_TO_X = np.array([
    [0.0, 0.0, 1.0, 0.0],
    [0.0, 1.0, 0.0, 0.0],
    [-1.0, 0.0, 0.0, 0.0],
    [0.0, 0.0, 0.0, 1.0],
])


def dh_transform(theta, d, a, alpha):
    ct, st = np.cos(theta), np.sin(theta)
    ca, sa = np.cos(alpha), np.sin(alpha)
    return np.array([
        [ct, -st * ca, st * sa, a * ct],
        [st, ct * ca, -ct * sa, a * st],
        [0.0, sa, ca, d],
        [0.0, 0.0, 0.0, 1.0],
    ])


def reference_cylinder_pose(link_index, q, z_offset=0.007):
    T = np.eye(4)
    for i in range(link_index):
        T = T @ dh_transform(q[i], UR5E_D[i], UR5E_A[i], UR5E_ALPHA[i])
    T[:3, 3] += z_offset * T[:3, 2]
    return T @ Z_TO_X


def test_cylinder_pose_matches_reference():
    robot = UR5e()
    q = np.array([0.1, -0.5, 0.3, 1.2, -0.7, 0.4])
    for link_index in range(7):
        np.testing.assert_allclose(
            robot.get_cylinder_transform(link_index, q),
            reference_cylinder_pose(link_index, q),
            atol=1e-12,
        )


def test_z_offset():
    robot = UR5e()
    q = np.zeros(6)
    np.testing.assert_allclose(
        robot.get_cylinder_transform(0, q, z_offset=0.0),
        Z_TO_X,
        atol=1e-12,
    )


def test_link_radii():
    robot = UR5e()
    expected = [0.0755, 0.0601, 0.0601, 0.0578, 0.235 / (2 * np.pi), 0.0393, 0.0376]
    for link_index, radius in enumerate(expected):
        assert robot.get_link_radius(link_index) == radius


def test_pose_is_rigid_transform():
    robot = UR5e()
    q = np.array([0.5, 0.5, 0.5, 0.5, 0.5, 0.5])
    T = robot.get_cylinder_transform(6, q)
    R = T[:3, :3]
    np.testing.assert_allclose(R @ R.T, np.eye(3), atol=1e-12)
    assert np.isclose(np.linalg.det(R), 1.0)
    np.testing.assert_allclose(T[3], [0.0, 0.0, 0.0, 1.0], atol=1e-12)
