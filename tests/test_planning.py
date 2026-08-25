import numpy as np
import pytest

from sda_bfc import (BeliefWorld, Halfspace, Planner, RRTPlanner,
                     UncertaintyRanges, UR5e, cloud_intersects_capsule,
                     fold_config, sample_valid_placement)

from planning import (CapsuleOracle, MeshOracle, Outcome, TouchSession, Wall,
                      calibrate)
from planning.workcell import Workcell

RANGES = (0.02, 0.02, 0.02, np.radians(2), np.radians(2), np.radians(2))


def make_ranges(values=RANGES):
    ranges = UncertaintyRanges()
    ranges.x, ranges.y, ranges.z, ranges.roll, ranges.pitch, ranges.yaw = values
    return ranges


def slsqp_distance(points, a, b):
    from scipy.optimize import minimize
    n = len(points)
    x0 = np.concatenate([np.full(n, 1.0 / n), [0.5]])
    result = minimize(
        lambda x: np.sum((x[:n] @ points - (a + x[n] * (b - a))) ** 2),
        x0, method="SLSQP", bounds=[(0, 1)] * (n + 1),
        constraints={"type": "eq", "fun": lambda x: x[:n].sum() - 1},
        options={"maxiter": 300, "ftol": 1e-14})
    return np.sqrt(max(result.fun, 0.0))


def test_gjk_matches_optimization_oracle():
    rng = np.random.default_rng(3)
    for _ in range(60):
        points = rng.uniform(-1, 1, (rng.integers(4, 10), 3))
        a, b = rng.uniform(-2, 2, 3), rng.uniform(-2, 2, 3)
        r = rng.uniform(0.05, 0.7)
        exact = slsqp_distance(points, a, b) <= r
        if abs(slsqp_distance(points, a, b) - r) < 1e-6:
            continue
        assert cloud_intersects_capsule(points, a, b, r) == exact


def belief_world(seed=1):
    robot = UR5e()
    x = sample_valid_placement(seed, robot)
    return robot, BeliefWorld(robot, x, fold_config(0.5), make_ranges())


def test_rrt_inherits_planner_and_returns_valid_path():
    robot, world = belief_world()
    planner = RRTPlanner(max_iterations=2000)
    assert isinstance(planner, Planner)
    goal = np.array([1.2, -1.2, 0.6, 0.3, 0.2, 0.1])
    if not world.is_free(goal):
        pytest.skip("sampled goal not free for this placement")
    path = planner.plan(world, fold_config(0.0), goal, seed=0)
    assert path is not None
    np.testing.assert_allclose(path[0], fold_config(0.0))
    np.testing.assert_allclose(path[-1], goal)
    for q0, q1 in zip(path, path[1:]):
        assert world.edge_free(np.asarray(q0), np.asarray(q1))


def test_halfspace_blocks_configurations():
    _, world = belief_world()
    assert world.is_free(fold_config(0.0))
    world.add_halfspace(Halfspace(np.array([0.0, 0.0, 1.0]), 0.2))
    assert not world.is_free(fold_config(0.0))  # fold reaches above z=0.2


def test_corridor_exemption_is_monotone():
    _, world = belief_world()
    rng = np.random.default_rng(0)
    for _ in range(50):
        q = rng.uniform(-np.pi, np.pi, 6)
        if world.is_free(q):
            assert world.free_except(q, 2, 4, 3)


def test_wall_halfspace_free_side_contains_origin():
    for offset in (0.8, -0.8):
        normal, bound = Wall("x", offset).halfspace()
        assert normal @ np.array([0.0, 0.0, 0.0]) < bound
        assert normal @ np.array([offset * 1.1, 0.0, 0.0]) > bound


def test_session_collects_touches_with_capsule_oracle():
    session = TouchSession(seed=1, range_tuple=RANGES,
                           oracle_factory=CapsuleOracle)
    attempts = session.collect(num_touches=3, max_attempts=15)
    assert len(session.touches) >= 3
    assert any(a.outcome is Outcome.SUCCESS for a in attempts)


def test_calibration_recovers_placement_from_touches():
    session = TouchSession(seed=1, range_tuple=RANGES,
                           oracle_factory=CapsuleOracle)
    session.collect(num_touches=7, max_attempts=25)
    assert len(session.touches) >= 7
    x_est = calibrate(session.robot, session.touches)
    assert x_est is not None
    assert np.linalg.norm(x_est[:3, 3] - session.x_true[:3, 3]) < 5e-3


def test_mesh_oracle_smoke():
    pytest.importorskip("fcl")
    session = TouchSession(seed=1, range_tuple=RANGES,
                           oracle_factory=MeshOracle)
    attempts = session.collect(num_touches=1, max_attempts=8)
    assert attempts
    assert len(session.touches) >= 1


def test_wall_blocks_session_workspace():
    session = TouchSession(seed=1, range_tuple=RANGES,
                           oracle_factory=CapsuleOracle,
                           walls=[Wall("x", 0.05)])
    session.collect(num_touches=1, max_attempts=5)
    for qa, qb in session.touches:
        for i, capsule in enumerate(session.planner.scene.capsules(qa)):
            if i >= 2:
                assert max(capsule.a[0], capsule.b[0]) + capsule.r <= 0.05 + 1e-6
