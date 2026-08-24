"""Viser visualization of the two-arm simulator scene.

Renders both arms with the official UR5e meshes (via robot_descriptions /
ViserUrdf) when available, falling back to the capsule collision model
otherwise.  The URDF joint convention matches our DH model exactly, up to a
fixed Rz(pi) base rotation (ROS ur_description quirk), verified numerically.

Arm A sits at the origin, arm B at a random placement X from the C++
simulator.  Buttons:
  * "randomize placement" -- new collision-free placement (clears touches)
  * "new touching pose"   -- runs the C++ contact generator and records the
                             touching configuration (green contact marker)
  * "<" / ">"             -- step through the recorded touching poses
  * "show capsules"       -- overlay the collision capsules

Usage:  python3 visualization/two_arm_scene.py   then open http://localhost:8080
"""

import time

import numpy as np
import viser
from scipy.spatial.transform import Rotation

from sda_bfc import (ContactGenerator, TwoArmScene, UR5e, fold_config,
                     sample_valid_placement, segment_closest)
from uncertainty_expansion import expand_arm

NUM_LINKS = 6
TOUCH_LINK = 3
A_COLOR, B_COLOR = (70, 120, 190), (230, 140, 40)
TOUCH_COLOR = (60, 200, 90)
EXPANDED_COLOR = (200, 70, 70)
RZ_PI = Rotation.from_euler("z", np.pi)

robot = UR5e()


def load_urdf():
    try:
        from robot_descriptions.loaders.yourdfpy import load_robot_description
        return load_robot_description("ur5e_description")
    except Exception as error:
        print(f"UR5e meshes unavailable ({error}); showing capsules only")
        return None


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

    urdf = load_urdf()
    urdf_vis = {}
    urdf_roots = {}
    if urdf is not None:
        from viser.extras import ViserUrdf
        for arm in ["A", "B"]:
            urdf_roots[arm] = server.scene.add_frame(f"/urdf{arm}", show_axes=False)
            urdf_vis[arm] = ViserUrdf(server, urdf, root_node_name=f"/urdf{arm}")
        urdf_roots["A"].wxyz = RZ_PI.as_quat(scalar_first=True)

    scene = TwoArmScene(robot, np.eye(4))
    capsules = {}
    for arm, color in [("A", A_COLOR), ("B", B_COLOR)]:
        for i in range(NUM_LINKS):
            t0, t1 = scene.link_extents(i)
            vertices, faces = cylinder_mesh(scene.link_radius(i), t0, t1)
            capsules[(arm, i)] = server.scene.add_mesh_simple(
                f"/caps{arm}/link{i}", vertices=vertices, faces=faces,
                color=color, opacity=0.4, visible=urdf is None)
    marker = server.scene.add_icosphere("/contact", radius=0.015,
                                        color=TOUCH_COLOR, visible=False)

    btn_placement = server.gui.add_button("randomize placement")
    btn_touch = server.gui.add_button("new touching pose")
    btn_prev = server.gui.add_button("<")
    btn_next = server.gui.add_button(">")
    status = server.gui.add_text("touches", initial_value="0 recorded", disabled=True)
    show_caps = server.gui.add_checkbox("show capsules", initial_value=urdf is None)
    show_expanded = server.gui.add_checkbox("show expanded collision",
                                            initial_value=False)
    with server.gui.add_folder("uncertainty range (+-)"):
        sliders = {
            "x": server.gui.add_slider("x (cm)", 0.0, 10.0, 0.25, 2.0),
            "y": server.gui.add_slider("y (cm)", 0.0, 10.0, 0.25, 2.0),
            "z": server.gui.add_slider("z (cm)", 0.0, 10.0, 0.25, 2.0),
            "roll": server.gui.add_slider("roll (deg)", 0.0, 10.0, 0.25, 2.0),
            "pitch": server.gui.add_slider("pitch (deg)", 0.0, 10.0, 0.25, 2.0),
            "yaw": server.gui.add_slider("yaw (deg)", 0.0, 10.0, 0.25, 2.0),
        }
    expanded_handles = []
    scene_for_extents = scene

    def ranges():
        return (sliders["x"].value / 100.0, sliders["y"].value / 100.0,
                sliders["z"].value / 100.0, np.radians(sliders["roll"].value),
                np.radians(sliders["pitch"].value), np.radians(sliders["yaw"].value))

    def redraw_expansion():
        for handle in expanded_handles:
            handle.remove()
        expanded_handles.clear()
        if not show_expanded.value:
            return
        qA, _ = state.configs()
        for i, (vertices, faces) in enumerate(
                expand_arm(robot, scene_for_extents, qA, ranges())):
            expanded_handles.append(server.scene.add_mesh_simple(
                f"/expanded/link{i}", vertices=vertices, faces=faces,
                color=EXPANDED_COLOR, opacity=0.3, side="double"))

    def redraw():
        base_b.wxyz = Rotation.from_matrix(state.X[:3, :3]).as_quat(scalar_first=True)
        base_b.position = state.X[:3, 3]
        qA, qB = state.configs()
        if urdf is not None:
            R_b = Rotation.from_matrix(state.X[:3, :3]) * RZ_PI
            urdf_roots["B"].wxyz = R_b.as_quat(scalar_first=True)
            urdf_roots["B"].position = state.X[:3, 3]
            urdf_vis["A"].update_cfg(qA)
            urdf_vis["B"].update_cfg(qB)
        for arm, q, base in [("A", qA, None), ("B", qB, state.X)]:
            for i in range(NUM_LINKS):
                T = robot.get_cylinder_transform(i, q, scene.link_z_offset(i))
                if base is not None:
                    T = base @ T
                capsules[(arm, i)].wxyz = Rotation.from_matrix(T[:3, :3]).as_quat(
                    scalar_first=True)
                capsules[(arm, i)].position = T[:3, 3]
                capsules[(arm, i)].visible = show_caps.value
        point = state.contact_point()
        marker.visible = point is not None
        if point is not None:
            marker.position = point
        n = len(state.touches)
        status.value = f"{state.index + 1}/{n} recorded" if n else "0 recorded"
        redraw_expansion()

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

    @show_caps.on_update
    def _(_):
        redraw()

    @show_expanded.on_update
    def _(_):
        redraw_expansion()

    for slider in sliders.values():
        @slider.on_update
        def _(_):
            redraw_expansion()

    redraw()
    print("open http://localhost:8080")
    while True:
        time.sleep(1.0)


if __name__ == "__main__":
    main()
