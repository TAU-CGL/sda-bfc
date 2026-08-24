"""Uncertainty-expanded collision geometry for the static arm R1.

The placement X of R2 relative to R1 is uncertain within a +-range box in
(x, y, z, roll, pitch, yaw).  With X_true = dT . X_nominal, R1 as seen from
R2's planning frame is displaced by dT^-1 -- so R1's capsules are expanded:
for every corner dT of the uncertainty box and every vertex of each
capsule's oriented bounding box, collect dT^-1 . vertex, and take the convex
hull.  Planning R2's motion against these hulls with the nominal X is then
safe for every placement in the range (up to the corner sampling of the
rotation set, adequate for small angles).
"""

import itertools

import numpy as np
from scipy.spatial import ConvexHull


def capsule_obb_vertices(T, radius, t0, t1):
    """8 vertices of the oriented bounding box of a capsule whose axis is
    column 2 of T, spanning [t0, t1] with the given radius (caps included)."""
    p = T[:3, 3]
    ex, ey, axis = T[:3, 0], T[:3, 1], T[:3, 2]
    vertices = []
    for sx, sy in itertools.product([-1.0, 1.0], repeat=2):
        for h in [t0 - radius, t1 + radius]:
            vertices.append(p + sx * radius * ex + sy * radius * ey + h * axis)
    return np.array(vertices)


def range_corner_transforms(ranges):
    """Inverse transforms dT^-1 for all corners of the +-range box.

    ranges = (dx, dy, dz, roll, pitch, yaw), translations in meters, angles
    in radians; dT = trans(t) . Rz(yaw) . Ry(pitch) . Rx(roll)."""
    transforms = []
    for signs in itertools.product([-1.0, 1.0], repeat=6):
        dx, dy, dz, roll, pitch, yaw = (s * r for s, r in zip(signs, ranges))
        cz, sz = np.cos(yaw), np.sin(yaw)
        cy, sy = np.cos(pitch), np.sin(pitch)
        cx, sx = np.cos(roll), np.sin(roll)
        Rz = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]])
        Ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
        Rx = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]])
        R = Rz @ Ry @ Rx
        t = np.array([dx, dy, dz])
        transforms.append((R.T, -R.T @ t))
    return transforms


def expand_capsule(T, radius, t0, t1, ranges):
    """Convex hull (vertices, faces) of the capsule OBB swept over the
    inverse uncertainty corners."""
    obb = capsule_obb_vertices(T, radius, t0, t1)
    points = np.vstack([obb @ R_inv.T + t_inv
                        for R_inv, t_inv in range_corner_transforms(ranges)])
    hull = ConvexHull(points)
    return hull.points, hull.simplices


def expand_arm(robot, scene, q, ranges, num_links=6):
    """Expanded hulls for every capsule of the static arm at config q."""
    hulls = []
    for i in range(num_links):
        t0, t1 = scene.link_extents(i)
        T = robot.get_cylinder_transform(i, q, scene.link_z_offset(i))
        hulls.append(expand_capsule(T, scene.link_radius(i), t0, t1, ranges))
    return hulls
