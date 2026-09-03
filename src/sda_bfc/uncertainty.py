"""How far a link's cylinder can be off -- one additive number, everywhere.

The model: whatever the arm's true pose is, every point of every link is
within `u` metres of where the nominal kinematics puts it.  The shape that
certainly contains the link is the nominal cylinder grown by `u` in every
direction (a Minkowski sum with a ball); the smallest cylinder containing
that is the obvious one -- radius r + u, ends pushed out by u along the axis.
Growing BOTH matters for flat-capped cylinders, whose end planes a
radius-only pad leaves uncovered.  (For the capsule oracle the endpoint
extension is slightly conservative -- a capsule's caps already reach r past
the endpoints -- which is the safe direction.)

This replaced the split translation/rotation budget with its lever-arm taper
(swept_radii_along): one additive number is coarser at the shoulder and
honest everywhere, and it is a number anyone can read off a measurement.
"""

import numpy as np


def inflate(P0, P1, radii, u):
    """(P0', P1', R'): every cylinder grown by `u` metres in every direction.

    P0/P1 are (..., 3) endpoints; `radii` broadcasts against their leading
    shape.  Zero `u` returns the inputs untouched, so "uncertainty off" means
    exactly the nominal cylinders rather than nominal plus float dust.
    """
    P0, P1 = np.asarray(P0, float), np.asarray(P1, float)
    radii = np.asarray(radii, float)
    u = float(u)
    if not u:
        return P0, P1, np.broadcast_to(radii, P0.shape[:-1])
    axis = P1 - P0
    n = np.linalg.norm(axis, axis=-1, keepdims=True)
    step = u * np.divide(axis, n, out=np.zeros_like(axis), where=n > 1e-12)
    return P0 - step, P1 + step, np.broadcast_to(radii, P0.shape[:-1]) + u


def inflated_cylinder(p0, p1, radius, u):
    """(p0', p1', r') for ONE cylinder -- the scalar form, for viewer shells."""
    P0, P1, R = inflate(np.asarray(p0, float)[None], np.asarray(p1, float)[None],
                        np.asarray([radius], float), u)
    return P0[0], P1[0], float(R[0])
