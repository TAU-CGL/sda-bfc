import numpy as np
import pytest

from sda_bfc import robot as rb
from sda_bfc.maneuver import densify, transit
from sda_bfc.robot import DualRobot, Robot, offset_matrix, scene, set_uncertainty
from sda_bfc.uncertainty import inflate, inflated_cylinder

Q_STATIC = np.array([0.0, -1.2, 1.0, -1.5, -1.5, 0.0])


@pytest.fixture(autouse=True)
def reset_uncertainty():
    yield
    set_uncertainty(0.0)


def test_inflate_grows_radius_and_endpoints():
    p0, p1, r = np.zeros(3), np.array([1.0, 0.0, 0.0]), 0.05
    q0, q1, rr = inflated_cylinder(p0, p1, r, 0.02)
    np.testing.assert_allclose(q0, [-0.02, 0, 0])
    np.testing.assert_allclose(q1, [1.02, 0, 0])
    assert rr == pytest.approx(0.07)
    same = inflate(p0[None], p1[None], np.array([r]), 0.0)
    np.testing.assert_array_equal(same[0][0], p0)
    assert same[2][0] == r


def test_oracle_pads_cross_pairs_only():
    ctx = scene()
    ctx.b.set_arm(Q_STATIC)
    q = np.array([0.5, -1.0, 0.8, -1.0, 1.0, 0.0])
    g0 = ctx.gaps(ctx.a, ctx.b, q)
    set_uncertainty(0.02)
    g1 = ctx.gaps(ctx.a, ctx.b, q)
    labels = rb.all_labels()
    for k, label in enumerate(labels):
        if not (np.isfinite(g0[k]) and np.isfinite(g1[k])):
            continue
        if "vs self/" in label:
            assert g1[k] == pytest.approx(g0[k], abs=1e-12), label
        elif "vs other/" in label:
            # radius alone shrinks the gap by u; endpoint growth only more so
            assert g1[k] <= g0[k] - 0.02 + 1e-9, label


def test_padded_plan_survives_true_base_error():
    """Plan at the belief with u >= |delta|; execution at the true base
    (belief + delta) must be collision-free -- the whole point of the pad."""
    delta = np.array([0.012, -0.012, 0.008, 0, 0, 0])  # |dt| ~ 19 mm
    u = 0.03
    ctx_hat = scene()
    ctx_true = DualRobot(Robot(name="A"),
                         Robot(offset_matrix(rb.BASE_OFFSET + delta), name="B"))
    for ctx in (ctx_hat, ctx_true):
        ctx.b.set_arm(Q_STATIC)

    rng = np.random.default_rng(3)
    planned = 0
    for _ in range(20):
        q_start, q_goal = rng.uniform(-np.pi, np.pi, (2, 6))
        q_start[1], q_goal[1] = np.clip([q_start[1], q_goal[1]], -3.0, 0.0)
        set_uncertainty(u)
        if ctx_hat._blocked(ctx_hat.a, ctx_hat.b, q_start) or \
                ctx_hat._blocked(ctx_hat.a, ctx_hat.b, q_goal):
            continue
        waypoints, info = transit(ctx_hat, ctx_hat.a, ctx_hat.b, q_start, q_goal)
        if not waypoints:
            continue
        planned += 1
        set_uncertainty(0.0)  # the true world is checked nominally
        assert not ctx_true.blocked_along(ctx_true.a, ctx_true.b,
                                          densify(waypoints), clearance=0.0), \
            info["method"]
    assert planned >= 2  # the guarantee was actually exercised
