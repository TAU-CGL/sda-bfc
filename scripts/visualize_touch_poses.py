import importlib.util
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.widgets import Button

from sda_bfc import UR5e

_spec = importlib.util.spec_from_file_location(
    "touch_poses", Path(__file__).resolve().parent.parent / "tests" / "test_touch_poses.py")
_touch = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_touch)

TOUCH_POSES = _touch.TOUCH_POSES
BASE_TRANSFORM = _touch.base_transform()

NUM_LINKS = 6
ARM_COLORS = ["tab:blue", "tab:orange"]

UR5E_D = [0.1625, 0.0, 0.0, 0.1333, 0.0997, 0.0996]
UR5E_A = [0.0, -0.425, -0.3922, 0.0, 0.0, 0.0]
DEFAULT_HOUSING_LENGTH = 0.12

robot = UR5e()


def link_extents():
    extents = []
    for i in range(NUM_LINKS):
        a_in = UR5E_A[i - 1] if i > 0 else 0.0
        d_in = UR5E_D[i - 1] if i > 0 else UR5E_D[0]
        if abs(a_in) > 1e-6:
            extents.append((min(0.0, -a_in), max(0.0, -a_in)))
        else:
            length = abs(d_in) if abs(d_in) > 1e-6 else DEFAULT_HOUSING_LENGTH
            extents.append((-length / 2, length / 2))
    return extents


LINK_EXTENTS = link_extents()


def cylinder_surface(T, radius, extent, n_theta=24, n_axis=2):
    a, b, u, p = T[:3, 0], T[:3, 1], T[:3, 2], T[:3, 3]
    theta = np.linspace(0.0, 2 * np.pi, n_theta)
    t = np.linspace(extent[0], extent[1], n_axis)
    theta, t = np.meshgrid(theta, t)
    circle = radius * (np.multiply.outer(np.cos(theta), a) + np.multiply.outer(np.sin(theta), b))
    axis = np.multiply.outer(t, u)
    pts = p + circle + axis
    return pts[..., 0], pts[..., 1], pts[..., 2]


def arm_transforms(q, base=np.eye(4)):
    return [base @ robot.get_cylinder_transform(i, q) for i in range(NUM_LINKS)]


class Viewer:
    def __init__(self):
        self.index = 0
        self.fig = plt.figure(figsize=(9, 8))
        self.ax = self.fig.add_subplot(projection="3d")
        ax_prev = self.fig.add_axes([0.35, 0.02, 0.1, 0.05])
        ax_next = self.fig.add_axes([0.55, 0.02, 0.1, 0.05])
        self.btn_prev = Button(ax_prev, "<")
        self.btn_next = Button(ax_next, ">")
        self.btn_prev.on_clicked(lambda _: self.step(-1))
        self.btn_next.on_clicked(lambda _: self.step(1))
        self.draw()

    def step(self, delta):
        self.index = (self.index + delta) % len(TOUCH_POSES)
        self.draw()

    def draw(self):
        self.ax.clear()
        q1, q2 = TOUCH_POSES[self.index]
        arms = [
            arm_transforms(np.array(q1)),
            arm_transforms(np.array(q2), BASE_TRANSFORM),
        ]
        for transforms, color in zip(arms, ARM_COLORS):
            for i, T in enumerate(transforms):
                X, Y, Z = cylinder_surface(T, robot.get_link_radius(i), LINK_EXTENTS[i])
                self.ax.plot_surface(X, Y, Z, color=color, alpha=0.7, shade=True)
        self.ax.set_title(f"Touch pose {self.index + 1}/{len(TOUCH_POSES)}")
        self.ax.set_xlabel("x")
        self.ax.set_ylabel("y")
        self.ax.set_zlabel("z")
        self.ax.set_box_aspect((1, 1, 1))
        self.ax.set_xlim(-0.7, 1.2)
        self.ax.set_ylim(-1.4, 0.5)
        self.ax.set_zlim(-0.5, 1.0)
        self.fig.canvas.draw_idle()


if __name__ == "__main__":
    Viewer()
    plt.show()
