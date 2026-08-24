import numpy as np
import pytest

import test_touch_poses as touch
from sda_bfc import (ContactGenerator, CylinderPose, SolverNewton, TwoArmScene,
                     UR5e, generate_experiment, sample_placement,
                     sample_valid_placement, segment_closest, segment_distance)

EXPECTED_EXTENTS = [(-0.08125, 0.08125), (-0.08125, 0.08125), (0.0, 0.425),
                    (0.0, 0.3922), (-0.06665, 0.06665), (-0.04985, 0.04985)]
RADIUS_INDEX = [0, 1, 2, 4, 5, 6]


def random_segments(rng, kind):
    p1 = rng.uniform(-1.0, 1.0, 3)
    d1 = rng.uniform(-1.0, 1.0, 3)
    if kind == "generic":
        p2 = rng.uniform(-1.0, 1.0, 3)
        d2 = rng.uniform(-1.0, 1.0, 3)
    elif kind == "parallel":
        p2 = p1 + rng.uniform(-0.5, 0.5, 3)
        d2 = d1 * rng.uniform(0.5, 2.0)
    elif kind == "collinear":
        p2 = p1 + rng.uniform(0.5, 2.0) * d1
        d2 = d1 * rng.uniform(-2.0, 2.0)
    else:
        p2 = rng.uniform(-1.0, 1.0, 3)
        d2 = np.zeros(3)
    return p1, p1 + d1, p2, p2 + d2


def test_segment_closest_vs_brute_force():
    rng = np.random.default_rng(0)
    grid = np.linspace(0.0, 1.0, 400)
    for trial in range(50):
        kind = ["generic", "generic", "parallel", "collinear", "degenerate"][trial % 5]
        p1, q1, p2, q2 = random_segments(rng, kind)
        sc = segment_closest(p1, q1, p2, q2)
        assert 0.0 <= sc.s <= 1.0 and 0.0 <= sc.t <= 1.0
        recomputed = np.linalg.norm((p1 + sc.s * (q1 - p1)) - (p2 + sc.t * (q2 - p2)))
        assert recomputed == pytest.approx(sc.distance, abs=1e-12)
        pts1 = p1 + grid[:, None] * (q1 - p1)
        pts2 = p2 + grid[:, None] * (q2 - p2)
        brute = np.min(np.linalg.norm(pts1[:, None, :] - pts2[None, :, :], axis=2))
        assert sc.distance <= brute + 1e-12
        assert brute - sc.distance <= 0.02


def test_capsule_model_matches_prototype():
    robot = UR5e()
    scene = TwoArmScene(robot, np.eye(4))
    for i in range(6):
        lo, hi = scene.link_extents(i)
        assert lo == pytest.approx(EXPECTED_EXTENTS[i][0], abs=1e-12)
        assert hi == pytest.approx(EXPECTED_EXTENTS[i][1], abs=1e-12)
        assert scene.link_radius(i) == robot.get_link_radius(RADIUS_INDEX[i])


def test_contact_jacobian_matches_finite_differences():
    robot = UR5e()
    rng = np.random.default_rng(1)
    for seed in range(20):
        X = sample_valid_placement(seed, robot)
        generator = ContactGenerator(robot, X)
        theta = rng.uniform(-np.pi, np.pi, 6)
        f, J = generator.residual_with_jacobian(theta)
        assert f == pytest.approx(generator.residual(theta), abs=1e-15)
        h = 1e-6
        J_fd = np.zeros(6)
        for i in range(6):
            e = np.zeros(6)
            e[i] = h
            J_fd[i] = (generator.residual(theta + e) - generator.residual(theta - e)) / (2 * h)
        np.testing.assert_allclose(J, J_fd, rtol=1e-5, atol=1e-9)


def test_placement_distribution_and_home_clearance():
    robot = UR5e()
    for seed in range(20):
        X = sample_valid_placement(seed, robot)
        assert 0.5 <= np.linalg.norm(X[:2, 3]) <= 0.78
        assert abs(X[2, 3]) <= 0.05
        tilt = np.degrees(np.arccos(np.clip(X[2, 2], -1.0, 1.0)))
        assert tilt <= 2.0 * np.sqrt(2.0) + 1e-9
        assert TwoArmScene(robot, X).collision_free_at_home()


def test_scene_reproduces_recorded_touches():
    robot = UR5e()
    scene = TwoArmScene(robot, touch.base_transform())
    for q1, q2 in touch.TOUCH_POSES:
        clearances = {(pc.i, pc.j): pc.clearance
                      for pc in scene.cross_clearances(np.array(q1), np.array(q2))}
        assert abs(clearances[(3, 3)]) < touch.TOUCH_TOLERANCE


def test_generated_contacts_are_valid():
    robot = UR5e()
    r = robot.get_link_radius(4)
    for seed in range(3):
        experiment = generate_experiment(6, seed=seed)
        scene = TwoArmScene(robot, experiment.X)
        assert len(experiment.q_as) == 6
        for qA, qB in zip(experiment.q_as, experiment.q_bs):
            for pc in scene.cross_clearances(qA, qB):
                if (pc.i, pc.j) == (3, 3):
                    assert abs(pc.clearance) < 1e-6
                else:
                    assert pc.clearance > 5e-3
            assert scene.min_self_clearance(qA) > 0.0
            assert scene.min_self_clearance(qB) > 0.0
            assert scene.contact_interior(qA, qB, 3, 3)
            a = CylinderPose.from_se3(robot.get_cylinder_transform(3, qA), r)
            b = CylinderPose.from_se3(
                experiment.X @ robot.get_cylinder_transform(3, qB), r)
            assert a.signed_distance(b) == pytest.approx(0.0, abs=1e-9)


def test_end_to_end_zero_noise_recovery():
    robot = UR5e()
    r = robot.get_link_radius(4)
    experiment = generate_experiment(12, seed=0)
    As = [robot.get_cylinder_transform(3, q) for q in experiment.q_as]
    Bs = [robot.get_cylinder_transform(3, q) for q in experiment.q_bs]
    solver = SolverNewton(As, Bs, r, r)
    assert solver.cost(experiment.X) < 1e-20
    X = solver.solve_multistart()
    translation_error = np.linalg.norm(X[:3, 3] - experiment.X[:3, 3])
    R_delta = X[:3, :3].T @ experiment.X[:3, :3]
    rotation_error = np.degrees(np.arccos(np.clip((np.trace(R_delta) - 1.0) / 2.0, -1.0, 1.0)))
    assert translation_error < 1e-3
    assert rotation_error < 0.1


def test_determinism():
    a = generate_experiment(4, seed=42)
    b = generate_experiment(4, seed=42)
    assert np.array_equal(a.X, b.X)
    for qa1, qa2 in zip(a.q_as, b.q_as):
        assert np.array_equal(qa1, qa2)
    for qb1, qb2 in zip(a.q_bs, b.q_bs):
        assert np.array_equal(qb1, qb2)
    c = generate_experiment(4, seed=43)
    assert not np.array_equal(a.X, c.X)


def test_generate_contact_optional():
    robot = UR5e()
    X = sample_valid_placement(0, robot)
    contact = ContactGenerator(robot, X).generate(seed=0)
    assert contact is not None
    assert np.all(np.abs(contact.q_a[3:]) <= np.pi)
    assert np.all(np.abs(contact.q_b[3:]) <= np.pi)
