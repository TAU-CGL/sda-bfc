import numpy as np
import pytest

from sda_bfc import CylinderPose

E_Z = np.array([0.0, 0.0, 1.0])


def random_unit_vector(rng):
    v = rng.normal(size=3)
    return v / np.linalg.norm(v)


def test_construct_and_attributes():
    cp = CylinderPose(np.array([1.0, 2.0, 3.0]), E_Z, 0.5)
    np.testing.assert_allclose(cp.p, [1.0, 2.0, 3.0])
    np.testing.assert_allclose(cp.u, E_Z)
    assert cp.r == 0.5


def test_to_se3_is_rigid_transform():
    rng = np.random.default_rng(0)
    for _ in range(100):
        cp = CylinderPose(rng.normal(size=3), random_unit_vector(rng), 0.1)
        T = cp.to_se3()
        R = T[:3, :3]
        np.testing.assert_allclose(R @ R.T, np.eye(3), atol=1e-12)
        assert np.isclose(np.linalg.det(R), 1.0)
        np.testing.assert_allclose(T[:3, 2], cp.u, atol=1e-12)
        np.testing.assert_allclose(T[:3, 3], cp.p, atol=1e-12)
        np.testing.assert_allclose(T[3], [0.0, 0.0, 0.0, 1.0], atol=1e-12)


def test_se3_roundtrip():
    rng = np.random.default_rng(1)
    for _ in range(100):
        cp = CylinderPose(rng.normal(size=3), random_unit_vector(rng), 0.2)
        back = CylinderPose.from_se3(cp.to_se3(), cp.r)
        np.testing.assert_allclose(back.p, cp.p, atol=1e-12)
        np.testing.assert_allclose(back.u, cp.u, atol=1e-12)
        assert back.r == cp.r


def test_signed_distance_known_values():
    a = CylinderPose(np.zeros(3), np.array([1.0, 0.0, 0.0]), 0.1)
    b = CylinderPose(np.array([0.0, 0.0, 2.0]), np.array([0.0, 1.0, 0.0]), 0.3)
    assert a.signed_distance(b) == pytest.approx(1.6)
    c = CylinderPose(np.array([0.0, 1.0, 0.0]), np.array([1.0, 0.0, 0.0]), 0.6)
    assert a.signed_distance(c) == pytest.approx(0.3)
    d = CylinderPose(np.array([0.0, 0.0, 0.1]), np.array([0.0, 1.0, 0.0]), 0.2)
    assert a.signed_distance(d) == pytest.approx(-0.2)


def test_signed_distance_symmetric():
    rng = np.random.default_rng(2)
    for _ in range(100):
        a = CylinderPose(rng.normal(size=3), random_unit_vector(rng), rng.uniform(0.01, 0.5))
        b = CylinderPose(rng.normal(size=3), random_unit_vector(rng), rng.uniform(0.01, 0.5))
        assert a.signed_distance(b) == pytest.approx(b.signed_distance(a), abs=1e-14)


def test_symmetric_representation_has_zero_distance():
    rng = np.random.default_rng(4)
    for _ in range(1000):
        cp = CylinderPose(rng.uniform(-5.0, 5.0, size=3), random_unit_vector(rng), rng.uniform(0.01, 0.5))
        T = cp.to_se3()
        T2 = T.copy()
        T2[:3, 0] *= -1.0
        T2[:3, 2] *= -1.0
        T2[:3, 3] += rng.uniform(-1000.0, 1000.0) * T[:3, 2]
        R = T2[:3, :3]
        np.testing.assert_allclose(R @ R.T, np.eye(3), atol=1e-12)
        assert np.isclose(np.linalg.det(R), 1.0)
        flipped = CylinderPose.from_se3(T2, cp.r)
        np.testing.assert_allclose(flipped.u, -cp.u, atol=1e-12)
        axis_distance = cp.signed_distance(flipped) + cp.r + flipped.r
        assert axis_distance == pytest.approx(0.0, abs=1e-9)


def test_signed_distance_upright_matches_general():
    rng = np.random.default_rng(3)
    cases = []
    for _ in range(10000):
        cases.append((
            rng.uniform(-5.0, 5.0, size=3),
            random_unit_vector(rng),
            rng.uniform(0.0, 1.0),
            rng.uniform(0.0, 1.0),
        ))
    for tilt in [1e-1, 1e-3, 1e-5, 1e-7]:
        for phi in np.linspace(0.0, 2 * np.pi, 8, endpoint=False):
            u = np.array([tilt * np.cos(phi), tilt * np.sin(phi), 1.0])
            u /= np.linalg.norm(u)
            cases.append((np.array([1.0, -2.0, 3.0]), u, 0.25, 0.4))
            cases.append((np.array([1.0, -2.0, 3.0]), -u, 0.25, 0.4))
    for p, u, r, other_r in [
        (np.zeros(3), np.array([1.0, 0.0, 0.0]), 0.1, 0.1),
        (np.array([0.0, 0.0, 7.0]), np.array([0.0, 1.0, 0.0]), 0.0, 0.0),
        (np.array([3.0, 4.0, 0.0]), np.array([0.6, 0.0, 0.8]), 0.5, 0.2),
        (np.array([1e6, 1e6, 0.0]), random_unit_vector(rng), 0.3, 0.3),
        (np.array([1e-8, 0.0, 0.0]), np.array([0.0, 1.0, 0.0]), 0.1, 0.1),
    ]:
        cases.append((p, u, r, other_r))
    for p, u, r, other_r in cases:
        cp = CylinderPose(p, u, r)
        upright = CylinderPose(np.zeros(3), E_Z, other_r)
        specialized = cp.signed_distance(other_r)
        general = cp.signed_distance(upright)
        scale = max(1.0, np.linalg.norm(p))
        assert specialized == pytest.approx(general, abs=1e-12 * scale), (p, u, r, other_r)


def test_signed_distance_upright_vertical_axis():
    cp = CylinderPose(np.array([2.0, 0.0, 5.0]), E_Z, 0.1)
    upright = CylinderPose(np.zeros(3), E_Z, 0.3)
    assert cp.signed_distance(upright) == pytest.approx(1.6)
    assert cp.signed_distance(0.3) == pytest.approx(1.6)
