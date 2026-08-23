"""Viser animation of the blind touch dance.

Records every configuration the dance actually visits (via the recorder hook
in blind_touch_dance) and replays it in a viser 3D scene: arm A in blue, arm
B in orange, green markers where a forearm-forearm touch was recorded and red
markers where a contact was rejected (wrong pair or end-cap).

Usage:  python3 dance_viser.py [seed]   then open http://localhost:8080
"""

import time

import numpy as np
import viser
from scipy.spatial.transform import Rotation

import blind_touch_dance as dance
from arm_simulator import LINK_EXTENTS, LINK_RADII, NUM_LINKS, capsules, segment_closest
from sda_bfc import UR5e

robot = UR5e()

FRAME_STRIDE = 3
FPS = 30.0
A_COLOR, B_COLOR = (70, 120, 190), (230, 140, 40)
TOUCH_COLOR, REJECT_COLOR = (60, 200, 90), (220, 60, 60)


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


def record_frames(seed):
    rng = np.random.default_rng(seed)
    X_true = dance.sample_ground_truth(rng)
    frames, events = [], []

    def recorder(qA, qB, event):
        if event is None:
            if len(frames) % FRAME_STRIDE == 0:
                frames.append((qA.copy(), qB.copy(), None))
            else:
                frames.append(None)
        else:
            frames.append((qA.copy(), qB.copy(), event))

    dance.perform_dance(X_true, verbose=True, recorder=recorder)
    kept = [f for f in frames if f is not None]
    return X_true, kept


def contact_point(X_true, qA, qB):
    p1, q1, _ = capsules(qA)[dance.DH_LINK]
    p2, q2, _ = capsules(qB, X_true)[dance.DH_LINK]
    _, s, t = segment_closest(p1, q1, p2, q2)
    return 0.5 * ((p1 + s * (q1 - p1)) + (p2 + t * (q2 - p2)))


def main(seed=0):
    print("recording dance...")
    X_true, frames = record_frames(seed)
    print(f"{len(frames)} animation frames")

    server = viser.ViserServer()
    server.scene.add_grid("/ground", width=4.0, height=4.0)
    server.scene.add_frame("/baseA", axes_length=0.15, axes_radius=0.004)
    server.scene.add_frame(
        "/baseB", axes_length=0.15, axes_radius=0.004,
        wxyz=Rotation.from_matrix(X_true[:3, :3]).as_quat(scalar_first=True),
        position=X_true[:3, 3])

    handles = {}
    for arm, color in [("A", A_COLOR), ("B", B_COLOR)]:
        for i in range(NUM_LINKS):
            vertices, faces = cylinder_mesh(LINK_RADII[i], *LINK_EXTENTS[i])
            handles[(arm, i)] = server.scene.add_mesh_simple(
                f"/arm{arm}/link{i}", vertices=vertices, faces=faces, color=color)

    def set_frame(index):
        qA, qB, event = frames[index]
        for arm, q, base in [("A", qA, None), ("B", qB, X_true)]:
            for i, T in enumerate(
                    [robot.get_cylinder_transform(i, q) for i in range(NUM_LINKS)]):
                if base is not None:
                    T = base @ T
                handle = handles[(arm, i)]
                handle.wxyz = Rotation.from_matrix(T[:3, :3]).as_quat(scalar_first=True)
                handle.position = T[:3, 3]

    marker_count = 0
    shown_events = set()

    def show_marker(index):
        nonlocal marker_count
        if index in shown_events:
            return
        qA, qB, event = frames[index]
        if event is None:
            return
        shown_events.add(index)
        kind, _ = event
        server.scene.add_icosphere(
            f"/contacts/{marker_count}", radius=0.012,
            color=TOUCH_COLOR if kind == "touch" else REJECT_COLOR,
            position=contact_point(X_true, qA, qB))
        marker_count += 1

    playing = server.gui.add_checkbox("play", initial_value=True)
    speed = server.gui.add_slider("speed", min=1, max=20, step=1, initial_value=5)
    slider = server.gui.add_slider("frame", min=0, max=len(frames) - 1, step=1,
                                   initial_value=0)
    status = server.gui.add_text("event", initial_value="", disabled=True)

    @slider.on_update
    def _(_):
        set_frame(slider.value)
        show_marker(slider.value)
        event = frames[slider.value][2]
        if event is not None:
            status.value = f"{event[0]} pair {event[1]}"

    set_frame(0)
    print("open http://localhost:8080")
    while True:
        if playing.value:
            slider.value = (slider.value + speed.value) % len(frames)
        time.sleep(1.0 / FPS)


if __name__ == "__main__":
    import sys
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 0)
