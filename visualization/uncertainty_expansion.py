"""Thin wrapper over the C++ uncertainty expansion (sim/capsule.hpp).

The C++ side produces the expanded point cloud (an outer bound of the
capsule under every placement in the uncertainty range, rotations sampled at
{-max, 0, +max} with a sagitta radius pad); scipy's convex hull here only
turns it into a drawable mesh.
"""

from scipy.spatial import ConvexHull

from sda_bfc import UncertaintyRanges, expand_capsule


def make_ranges(x, y, z, roll, pitch, yaw):
    ranges = UncertaintyRanges()
    ranges.x, ranges.y, ranges.z = x, y, z
    ranges.roll, ranges.pitch, ranges.yaw = roll, pitch, yaw
    return ranges


def expand_arm(robot, scene, q, range_tuple, num_links=6):
    """Expanded hulls (vertices, faces) for every capsule of the static arm."""
    ranges = make_ranges(*range_tuple)
    hulls = []
    for capsule in scene.capsules(q):
        points = expand_capsule(capsule, ranges)
        hull = ConvexHull(points)
        hulls.append((hull.points, hull.simplices))
    return hulls
