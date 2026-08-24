"""Viser visualization of the two-arm simulator scene.

Arm A (blue) sits at the origin, arm B (orange) at a random placement X from
the C++ simulator.  Buttons:
  * "randomize placement" -- new collision-free placement (clears touches)
  * "new touching pose"   -- runs the C++ contact generator and records the
                             touching configuration (green contact marker)
  * "<" / ">"             -- step through the recorded touching poses

Usage:  python3 visualization/two_arm_scene.py   then open http://localhost:8080
"""

import time

import numpy as np
import viser
from scipy.spatial.transform import Rotation

from sda_bfc import (ContactGenerator, TwoArmScene, UR5e, fold_config,
                     sample_valid_placement, segment_closest)

NUM_LINKS = 6
TOUCH_LINK = 3
A_COLOR, B_COLOR = (70, 120, 190), (230, 140, 40)
TOUCH_COLOR = (60, 200, 90)

robot = UR5e()


def cylinder_mesh(radius, t0, t1, n=24):
    angles = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False)
    ring = np.stack([radius * np.cos(angles), radius * np.sin(angles)], axis=1)
    bottom = np.hstack([ring, np.full((n, 1), t0)])
    top = np.hstack([ring, np.full((n, 1), t1)])
    vertices = np.vstack([bottom, top, [[0, 0, t0]], [[0, 0, t1]]])
    faces = []
    for i in range(n):
        j = (i + 1) % n
        faces += [[i, j, n + i], [j, n + j, n + i]]
        faces += [[2 * n, j, i], [2 * n + 1, n + i, n + j]]
    return vertices, np.array(faces)


class SceneState:
    """Placement + recorded touching poses; no GUI dependencies (testable)."""

    def __init__(self):
        self.placement_seed = 0
        self.contact_seed = 0
        self.X = sample_valid_placement(self.placement_seed, robot)
        self.touches = []
        self.index = -1

    def randomize_placement(self):
        self.placement_seed += 1
        self.X = sample_valid_placement(self.placement_seed, robot)
        self.touches = []
        self.index = -1

    def new_touch(self, max_tries=5):
        generator = ContactGenerator(robot, self.X)
        for _ in range(max_tries):
            self.contact_seed += 1
            contact = generator.generate(seed=self.contact_seed)
            if contact is not None:
                self.touches.append((np.array(contact.q_a), np.array(contact.q_b)))
                self.index = len(self.touches) - 1
                return True
        return False

    def step(self, delta):
        if self.touches:
            self.index = (self.index + delta) % len(self.touches)

    def configs(self):
        if 0 <= self.index < len(self.touches):
            return self.touches[self.index]
        return fold_config(0.0), fold_config(0.0)

    def contact_point(self):
        if not (0 <= self.index < len(self.touches)):
            return None
        qA, qB = self.touches[self.index]
        scene = TwoArmScene(robot, self.X)
        t0, t1 = scene.link_extents(TOUCH_LINK)
        TA = robot.get_cylinder_transform(TOUCH_LINK, qA)
        TB = self.X @ robot.get_cylinder_transform(TOUCH_LINK, qB)
        pA, uA = TA[:3, 3], TA[:3, 2]
        pB, uB = TB[:3, 3], TB[:3, 2]
        sc = segment_closest(pA + t0 * uA, pA + t1 * uA, pB + t0 * uB, pB + t1 * uB)
        a = (pA + t0 * uA) + sc.s * (t1 - t0) * uA
        b = (pB + t0 * uB) + sc.t * (t1 - t0) * uB
        return 0.5 * (a + b)


def main():
    state = SceneState()
    server = viser.ViserServer()
    server.scene.add_grid("/ground", width=4.0, height=4.0)
    server.scene.add_frame("/baseA", axes_length=0.15, axes_radius=0.004)
    base_b = server.scene.add_frame("/baseB", axes_length=0.15, axes_radius=0.004)

    scene = TwoArmScene(robot, np.eye(4))
    handles = {}
    for arm, color in [("A", A_COLOR), ("B", B_COLOR)]:
        for i in range(NUM_LINKS):
            t0, t1 = scene.link_extents(i)
            vertices, faces = cylinder_mesh(scene.link_radius(i), t0, t1)
            handles[(arm, i)] = server.scene.add_mesh_simple(
                f"/arm{arm}/link{i}", vertices=vertices, faces=faces, color=color)
    marker = server.scene.add_icosphere("/contact", radius=0.015,
                                        color=TOUCH_COLOR, visible=False)

    btn_placement = server.gui.add_button("randomize placement")
    btn_touch = server.gui.add_button("new touching pose")
    btn_prev = server.gui.add_button("<")
    btn_next = server.gui.add_button(">")
    status = server.gui.add_text("touches", initial_value="0 recorded", disabled=True)

    def redraw():
        base_b.wxyz = Rotation.from_matrix(state.X[:3, :3]).as_quat(scalar_first=True)
        base_b.position = state.X[:3, 3]
        qA, qB = state.configs()
        for arm, q, base in [("A", qA, None), ("B", qB, state.X)]:
            for i in range(NUM_LINKS):
                T = robot.get_cylinder_transform(i, q)
                if base is not None:
                    T = base @ T
                handles[(arm, i)].wxyz = Rotation.from_matrix(T[:3, :3]).as_quat(
                    scalar_first=True)
                handles[(arm, i)].position = T[:3, 3]
        point = state.contact_point()
        marker.visible = point is not None
        if point is not None:
            marker.position = point
        n = len(state.touches)
        current = f"{state.index + 1}/{n}" if n else "0"
        status.value = f"{current} recorded" if n else "0 recorded"

    @btn_placement.on_click
    def _(_):
        state.randomize_placement()
        redraw()

    @btn_touch.on_click
    def _(_):
        if not state.new_touch():
            status.value = "no touch found, try again"
            return
        redraw()

    @btn_prev.on_click
    def _(_):
        state.step(-1)
        redraw()

    @btn_next.on_click
    def _(_):
        state.step(1)
        redraw()

    redraw()
    print("open http://localhost:8080")
    while True:
        time.sleep(1.0)


if __name__ == "__main__":
    main()
