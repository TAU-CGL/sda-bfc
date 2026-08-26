"""Generate forearm-forearm contact configurations for a known base transform.

Usage: python3 scripts/generate_contacts.py [--transform x y z rx ry rz]
                                            [-n COUNT] [--seed SEED] [--out FILE]
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from sda_bfc import (ContactGenerator, CylinderPose, Halfspace,  # noqa: E402
                     UR5e)

DEFAULT_TRANSFORM = (0.500658, -0.917333, 0.272303,
                     -0.004370, -0.000456, 0.000200)
TOUCH_LINK = 3
RADII_LINK_INDEX = 4


def se3_from_pose(pose):
    """4x4 from [x, y, z, rx, ry, rz] with (rx, ry, rz) a rotation vector."""
    rvec = np.asarray(pose[3:], dtype=float)
    theta = np.linalg.norm(rvec)
    R = np.eye(3)
    if theta > 1e-12:
        k = rvec / theta
        K = np.array([[0.0, -k[2], k[1]], [k[2], 0.0, -k[0]], [-k[1], k[0], 0.0]])
        R = np.eye(3) + np.sin(theta) * K + (1.0 - np.cos(theta)) * (K @ K)
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = pose[:3]
    return T


def touch_residual(robot, X, q_a, q_b):
    """Signed distance between the two forearm cylinders; zero at contact."""
    radius = robot.get_link_radius(RADII_LINK_INDEX)
    c_a = CylinderPose.from_se3(robot.get_cylinder_transform(TOUCH_LINK, q_a), radius)
    c_b = CylinderPose.from_se3(X @ robot.get_cylinder_transform(TOUCH_LINK, q_b), radius)
    return c_a.signed_distance(c_b)


def generate_contacts(robot, X, count, seed, planes):
    """Collision-free interior forearm touches, one per accepted seed."""
    generator = ContactGenerator(robot, X)
    for p in planes:
        half_space = Halfspace(p[1], p[0])
        generator.add_static_halfspace(half_space)
        generator.add_dynamic_halfspace(half_space)
    contacts, offset = [], 0
    while len(contacts) < count:
        contact = generator.generate(seed=seed + offset)
        offset += 1
        if contact is not None:
            contacts.append((np.array(contact.q_a), np.array(contact.q_b)))
    return contacts


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--transform", type=float, nargs=6, default=DEFAULT_TRANSFORM)
    parser.add_argument("-n", "--count", type=int, default=20)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=Path,
                        default=Path(__file__).resolve().parent.parent / "contacts.json")
    args = parser.parse_args()

    robot = UR5e()
    X = se3_from_pose(np.asarray(args.transform, dtype=float))
    # contacts = generate_contacts(robot, X, args.count, args.seed, args.floor, args.wall_x, args.wall_y)
    planes = [
        (0.0, np.array([0.0, 0.0, -1.0])),    # floor:  z >= 0
        (0.285, np.array([0.0, 1.0, 0.0])),   # wall:   y <= 0.285
        (0.430, np.array([-1.0, 0.0, 0.0])),  # wall:   x >= -0.430
    ]
    contacts = generate_contacts(robot, X, args.count, args.seed, planes)

    for i, (q_a, q_b) in enumerate(contacts):
        residual = touch_residual(robot, X, q_a, q_b)
        print(f"{i:3d}  qA={np.round(q_a, 4).tolist()}  "
              f"qB={np.round(q_b, 4).tolist()}  d={residual:+.2e}")

    if args.out:
        args.out.write_text(json.dumps(
            {"transform": list(args.transform),
             "contacts": [{"q_a": a.tolist(), "q_b": b.tolist()} for a, b in contacts]},
            indent=2))
        print(f"\nwrote {len(contacts)} contacts to {args.out}")


if __name__ == "__main__":
    main()
