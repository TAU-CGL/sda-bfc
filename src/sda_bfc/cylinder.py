"""Finite cylinders and segment-to-segment closest points (scalar forms).

robot.segment_gaps is the vectorised twin of segment_closest_points: same
clamped solve in the same order, so they agree to floating-point noise.
"""

import numpy as np


def segment_closest_points(p0, p1, q0, q1):
    """(distance, s, t, c_self, c_other) between segments [p0,p1] and [q0,q1]."""
    d1, d2, r = p1 - p0, q1 - q0, p0 - q0
    a, e = float(d1 @ d1), float(d2 @ d2)
    b, c, f = float(d1 @ d2), float(d1 @ r), float(d2 @ r)
    denom = a * e - b * b
    s = np.clip((b * f - c * e) / denom, 0.0, 1.0) if denom > 1e-12 else 0.0
    t = np.clip((b * s + f) / e, 0.0, 1.0) if e > 1e-12 else 0.0
    s = np.clip((b * t - c) / a, 0.0, 1.0) if a > 1e-12 else 0.0
    c_self, c_other = p0 + s * d1, q0 + t * d2
    return float(np.linalg.norm(c_self - c_other)), s, t, c_self, c_other


class Cylinder:
    """center, unit axis, half_length, radius."""

    def __init__(self, center, axis, half_length, radius):
        self.center, self.axis = np.asarray(center, float), np.asarray(axis, float)
        self.half_length, self.radius = float(half_length), float(radius)

    @classmethod
    def from_segment(cls, p0, p1, radius):
        p0, p1 = np.asarray(p0, float), np.asarray(p1, float)
        span = p1 - p0
        h = float(np.linalg.norm(span))
        axis = span / h if h > 1e-12 else np.array([0.0, 0.0, 1.0])
        return cls(0.5 * (p0 + p1), axis, 0.5 * h, radius)

    def get_endpoints(self):
        return (self.center - self.half_length * self.axis,
                self.center + self.half_length * self.axis)

    def lateral_gap(self, other):
        """(gap, c_self, c_other, s, t): surface gap, witness points ON THE
        AXES.  The axes are separated by at least the two radii, so the
        witness direction is always well defined."""
        p0, p1 = self.get_endpoints()
        q0, q1 = other.get_endpoints()
        dist, s, t, c_self, c_other = segment_closest_points(p0, p1, q0, q1)
        return dist - self.radius - other.radius, c_self, c_other, s, t
