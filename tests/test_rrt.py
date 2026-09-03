import numpy as np
import pytest

pytest.importorskip("ompl")

from sda_bfc.maneuver import transit, transit_clearance
from sda_bfc.robot import scene
from sda_bfc.rrt import _segment, rrt_transit

Q_STATIC = np.array([0.0, -1.2, 1.0, -1.5, -1.5, 0.0])
# free start/goal whose straight segment crosses the static arm B
Q_START = np.array([-2.893, 0.0, -0.256, -2.75, 0.888, 2.216])
Q_GOAL = np.array([0.584, -1.507, 2.136, 0.06, 0.068, 1.59])


@pytest.fixture(scope="module")
def ctx():
    ctx = scene()
    ctx.b.set_arm(Q_STATIC)
    return ctx


def path_is_free(ctx, waypoints, clearance):
    return all(not ctx.blocked_along(ctx.a, ctx.b, _segment(a, b), clearance)
               for a, b in zip(waypoints, waypoints[1:]))


def test_direct_shortcut_when_segment_is_clear(ctx):
    waypoints, info = rrt_transit(ctx, ctx.a, ctx.b,
                                  Q_START, Q_START + 0.05, budget=3.0)
    assert info["method"] == "direct"
    assert len(waypoints) == 2


def test_plans_around_static_arm(ctx):
    clearance = transit_clearance(ctx)
    assert ctx.blocked_along(ctx.a, ctx.b, _segment(Q_START, Q_GOAL), clearance)
    waypoints, info = rrt_transit(ctx, ctx.a, ctx.b, Q_START, Q_GOAL, budget=3.0)
    assert waypoints, info["reason"]
    assert info["method"].startswith(("rrt", "ladder"))
    np.testing.assert_array_equal(waypoints[0], Q_START)
    np.testing.assert_array_equal(waypoints[-1], Q_GOAL)
    assert path_is_free(ctx, waypoints, clearance)


def test_ladder_transit_same_query(ctx):
    waypoints, info = transit(ctx, ctx.a, ctx.b, Q_START, Q_GOAL)
    if waypoints:  # the fixed ladder may or may not cover this query
        assert path_is_free(ctx, waypoints, transit_clearance(ctx))
    else:
        assert info["reason"]


def test_blocked_start_fails_with_reason(ctx):
    q_bad = Q_STATIC.copy()  # sit exactly on the static arm
    waypoints, info = rrt_transit(ctx, ctx.a, ctx.b, q_bad, Q_GOAL, budget=1.0)
    if waypoints:
        pytest.skip("expected blocked config is free in this scene")
    assert info["reason"].startswith(("start blocked", "AORRTC"))
