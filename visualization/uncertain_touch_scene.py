"""Viser visualization of touch generation under placement uncertainty.

Arm A at the origin; arm B rendered at its TRUE placement X_gt.  The belief
frame X_initial is drawn as a ghost frame; the uncertainty sliders drive
both the belief sampling and the expanded-collision preview.

Buttons:
  * "randomize placement" -- samples X_gt and a belief X_initial inside the
                             current uncertainty range
  * "get touching path"   -- runs one full uncertain-touch attempt (candidate
                             under the belief, guarded R1 placement, RRT for
                             R2 against expanded obstacles, guarded approach);
                             draws the executed forearm trace and animates it
  * "show expanded collision" -- the belief-side obstacles R2 planned against

Usage:  python3 visualization/uncertain_touch_scene.py  ->  http://localhost:8080
"""

import sys
import time
from pathlib import Path

import numpy as np
import viser
from scipy.spatial.transform import Rotation

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sda_bfc import TwoArmScene, UR5e, fold_config  # noqa: E402
from planning import CapsuleOracle, MeshOracle, TouchSession, Wall  # noqa: E402
from uncertainty_expansion import expand_arm  # noqa: E402

NUM_LINKS = 6
TOUCH_LINK = 3
A_COLOR, B_COLOR = (70, 120, 190), (230, 140, 40)
PATH_COLOR = (240, 200, 60)
EXPANDED_COLOR = (200, 70, 70)
BELIEF_COLOR = (150, 150, 220)
RZ_PI = Rotation.from_euler("z", np.pi)

robot = UR5e()


def load_urdf():
    try:
        from robot_descriptions.loaders.yourdfpy import load_robot_description
        return load_robot_description("ur5e_description")
    except Exception as error:
        print(f"UR5e meshes unavailable ({error})")
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


def forearm_tip(q, base):
    T = base @ robot.get_cylinder_transform(TOUCH_LINK, q)
    t0, t1 = SCENE_I.link_extents(TOUCH_LINK)
    return T[:3, 3] + 0.5 * (t0 + t1) * T[:3, 2]


SCENE_I = TwoArmScene(robot, np.eye(4))


class ViewerState:
    def __init__(self):
        self.placement_seed = 0
        self.session = None
        self.frames = []
        self.walls = []
        self.use_mesh_oracle = True
        self.attempt_frames = []
        self.status = "press randomize placement"

    def randomize(self, ranges):
        self.placement_seed += 1
        self._build_session(ranges)
        self.status = f"placement #{self.placement_seed}"

    def rebuild_with_walls(self, ranges, walls):
        """Same placement seed => same X_gt/X_belief; touches reset."""
        self.walls = list(walls)
        if self.session is not None:
            self._build_session(ranges)

    def _build_session(self, ranges):
        self.frames = []
        oracle = MeshOracle if self.use_mesh_oracle else CapsuleOracle
        self.session = TouchSession(
            seed=self.placement_seed, range_tuple=ranges,
            oracle_factory=oracle, walls=self.walls,
            robot=robot, recorder=self._record)

    def _record(self, qA, qB):
        self.frames.append((qA, qB))

    def get_touching_path(self):
        if self.session is None:
            return "randomize first"
        start = len(self.frames)
        result = self.session.attempt()
        self.attempt_frames = self.frames[start:]
        return (f"{result.outcome.value} "
                f"({len(self.session.touches)} touches, "
                f"plan {result.plan_seconds:.1f}s)")


def main():
    state = ViewerState()
    server = viser.ViserServer()
    floor = server.scene.add_grid("/floor", width=4.0, height=4.0)
    wall_boxes = {}
    for axis in ("x", "y"):
        wall_boxes[axis] = server.scene.add_box(
            f"/wall_{axis}", color=(160, 160, 160),
            dimensions=(0.04, 3.0, 2.0) if axis == "x" else (3.0, 0.04, 2.0),
            visible=False, opacity=0.35)
    server.scene.add_frame("/baseA", axes_length=0.15, axes_radius=0.004)
    base_gt = server.scene.add_frame("/baseB_true", axes_length=0.15,
                                     axes_radius=0.004)
    base_belief = server.scene.add_frame("/baseB_belief", axes_length=0.22,
                                         axes_radius=0.002)

    urdf = load_urdf()
    urdf_vis, urdf_roots = {}, {}
    if urdf is not None:
        from viser.extras import ViserUrdf
        for arm in ["A", "B"]:
            urdf_roots[arm] = server.scene.add_frame(f"/urdf{arm}", show_axes=False)
            urdf_vis[arm] = ViserUrdf(server, urdf, root_node_name=f"/urdf{arm}")
        urdf_roots["A"].wxyz = RZ_PI.as_quat(scalar_first=True)

    capsules = {}
    for arm, color in [("A", A_COLOR), ("B", B_COLOR)]:
        for i in range(NUM_LINKS):
            t0, t1 = SCENE_I.link_extents(i)
            vertices, faces = cylinder_mesh(SCENE_I.link_radius(i), t0, t1)
            capsules[(arm, i)] = server.scene.add_mesh_simple(
                f"/caps{arm}/link{i}", vertices=vertices, faces=faces,
                color=color, opacity=0.4, visible=urdf is None)
    # The belief ghost: R2's capsule model at X_initial -- exactly what the
    # planner reasons about (the robot "thinks" it is here).
    ghost = {}
    for i in range(NUM_LINKS):
        t0, t1 = SCENE_I.link_extents(i)
        vertices, faces = cylinder_mesh(SCENE_I.link_radius(i), t0, t1)
        ghost[i] = server.scene.add_mesh_simple(
            f"/belief/link{i}", vertices=vertices, faces=faces,
            color=BELIEF_COLOR, opacity=0.35, visible=True)

    btn_randomize = server.gui.add_button("randomize placement")
    btn_path = server.gui.add_button("get touching path")
    status = server.gui.add_text("status", initial_value=state.status,
                                 disabled=True)
    playing = server.gui.add_checkbox("play animation", initial_value=False)
    frame_slider = server.gui.add_slider("frame", 0, 1, 1, 0)
    show_expanded = server.gui.add_checkbox("show expanded collision", False)
    show_belief = server.gui.add_checkbox("show belief ghost", True)
    with server.gui.add_folder("walls (persist across placements)"):
        wall_controls = {
            "x": (server.gui.add_checkbox("wall x", False),
                  server.gui.add_slider("x offset (m)", -1.5, 1.5, 0.05, 1.0)),
            "y": (server.gui.add_checkbox("wall y", False),
                  server.gui.add_slider("y offset (m)", -1.5, 1.5, 0.05, 1.0)),
        }
    with server.gui.add_folder("uncertainty range (+-)"):
        sliders = {
            "x": server.gui.add_slider("x (cm)", 0.0, 10.0, 0.25, 2.0),
            "y": server.gui.add_slider("y (cm)", 0.0, 10.0, 0.25, 2.0),
            "z": server.gui.add_slider("z (cm)", 0.0, 10.0, 0.25, 2.0),
            "roll": server.gui.add_slider("roll (deg)", 0.0, 10.0, 0.25, 2.0),
            "pitch": server.gui.add_slider("pitch (deg)", 0.0, 10.0, 0.25, 2.0),
            "yaw": server.gui.add_slider("yaw (deg)", 0.0, 10.0, 0.25, 2.0),
        }
    path_handle = [None]
    expanded_handles = []

    def current_walls():
        return [Wall(axis, slider.value)
                for axis, (checkbox, slider) in wall_controls.items()
                if checkbox.value]

    def redraw_workcell():
        if state.session is None:
            return
        floor.position = (0.0, 0.0, state.session.workcell.floor_z)
        for axis, (checkbox, slider) in wall_controls.items():
            wall_boxes[axis].visible = checkbox.value
            center = [0.0, 0.0, 1.0]
            center[0 if axis == "x" else 1] = slider.value
            wall_boxes[axis].position = center

    def on_wall_change(_):
        state.rebuild_with_walls(ranges(), current_walls())
        status.value = "walls applied (touches reset)"
        frame_slider.max = 1
        redraw_placement()

    def ranges():
        return (sliders["x"].value / 100.0, sliders["y"].value / 100.0,
                sliders["z"].value / 100.0, np.radians(sliders["roll"].value),
                np.radians(sliders["pitch"].value),
                np.radians(sliders["yaw"].value))

    def set_config(qA, qB):
        X = state.session.X_gt
        if urdf is not None:
            R_b = Rotation.from_matrix(X[:3, :3]) * RZ_PI
            urdf_roots["B"].wxyz = R_b.as_quat(scalar_first=True)
            urdf_roots["B"].position = X[:3, 3]
            urdf_vis["A"].update_cfg(np.asarray(qA))
            urdf_vis["B"].update_cfg(np.asarray(qB))
        for arm, q, base in [("A", qA, np.eye(4)), ("B", qB, X)]:
            for i in range(NUM_LINKS):
                T = base @ robot.get_cylinder_transform(
                    i, np.asarray(q), SCENE_I.link_z_offset(i))
                capsules[(arm, i)].wxyz = Rotation.from_matrix(
                    T[:3, :3]).as_quat(scalar_first=True)
                capsules[(arm, i)].position = T[:3, 3]
                capsules[(arm, i)].visible = urdf is None
        Xb = state.session.X_initial
        for i in range(NUM_LINKS):
            T = Xb @ robot.get_cylinder_transform(
                i, np.asarray(qB), SCENE_I.link_z_offset(i))
            ghost[i].wxyz = Rotation.from_matrix(T[:3, :3]).as_quat(
                scalar_first=True)
            ghost[i].position = T[:3, 3]
            ghost[i].visible = show_belief.value

    def show_frame(index):
        if not state.frames:
            return
        index = int(np.clip(index, 0, len(state.frames) - 1))
        set_config(*state.frames[index])

    def redraw_placement():
        X, Xb = state.session.X_gt, state.session.X_initial
        base_gt.wxyz = Rotation.from_matrix(X[:3, :3]).as_quat(scalar_first=True)
        base_gt.position = X[:3, 3]
        base_belief.wxyz = Rotation.from_matrix(Xb[:3, :3]).as_quat(scalar_first=True)
        base_belief.position = Xb[:3, 3]
        set_config(fold_config(0.0), fold_config(0.0))
        redraw_workcell()
        if path_handle[0] is not None:
            path_handle[0].remove()
            path_handle[0] = None
        redraw_expansion()

    def redraw_expansion():
        for handle in expanded_handles:
            handle.remove()
        expanded_handles.clear()
        if not show_expanded.value or state.session is None:
            return
        qA = state.frames[-1][0] if state.frames else fold_config(0.0)
        for i, (vertices, faces) in enumerate(
                expand_arm(robot, SCENE_I, np.asarray(qA), ranges())):
            expanded_handles.append(server.scene.add_mesh_simple(
                f"/expanded/link{i}", vertices=vertices, faces=faces,
                color=EXPANDED_COLOR, opacity=0.3, side="double"))

    @btn_randomize.on_click
    def _(_):
        state.walls = current_walls()
        state.randomize(ranges())
        status.value = state.status
        frame_slider.max = 1
        redraw_placement()

    for checkbox, slider in wall_controls.values():
        checkbox.on_update(on_wall_change)
        slider.on_update(on_wall_change)

    @btn_path.on_click
    def _(_):
        if state.session is None:
            status.value = "randomize first"
            return
        status.value = "planning..."
        message = state.get_touching_path()
        status.value = message
        if state.frames:
            frame_slider.max = len(state.frames) - 1
            frame_slider.value = len(state.frames) - 1
            show_frame(frame_slider.value)
            trace = np.array([forearm_tip(qB, state.session.X_gt)
                              for _, qB in state.frames[-len(state.attempt_frames):]])
            if path_handle[0] is not None:
                path_handle[0].remove()
            if len(trace) >= 2:
                path_handle[0] = server.scene.add_spline_catmull_rom(
                    "/touch_path", trace, color=PATH_COLOR, line_width=3.0)
        redraw_expansion()

    @frame_slider.on_update
    def _(_):
        if not playing.value:
            show_frame(frame_slider.value)

    @show_expanded.on_update
    def _(_):
        redraw_expansion()

    @show_belief.on_update
    def _(_):
        if state.frames:
            show_frame(frame_slider.value)
        elif state.session is not None:
            set_config(fold_config(0.0), fold_config(0.0))

    print("open http://localhost:8080")
    while True:
        if playing.value and state.frames:
            frame_slider.value = (frame_slider.value + 4) % len(state.frames)
            show_frame(frame_slider.value)
        time.sleep(1.0 / 30.0)


if __name__ == "__main__":
    main()
