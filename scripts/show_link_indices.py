"""Visual check of UR5e cylinder-link indices.

Renders the MJCF visual model at a fixed config and overlays the analytic
cylinder of each link index i (axis frame get_cylinder_transform(i) shifted
by get_link_z_offset(i), radius get_cylinder_radius(i), length |dhA[i - 1]|), labeled "link i".  Zero-length
links are drawn as thin stubs so their axis is still visible.  Open the
printed viser URL: each translucent colored cylinder should hug its link.
"""

import time
from pathlib import Path

import mujoco
import numpy as np
import trimesh
import viser

from sda_bfc import UR5e

MJCF = str(Path(__file__).resolve().parent.parent
           / "assets" / "universal_robots_ur5e" / "ur5e.xml")
UR5E_A = [0.0, -0.425, -0.3922, 0.0, 0.0, 0.0]
Q = np.array([0.0, -1.2, 1.0, -1.5, -1.5, 0.0])
COLORS = [(180, 180, 180), (230, 80, 80), (80, 180, 80),
          (80, 120, 230), (230, 180, 60), (180, 80, 230)]

robot = UR5e()
server = viser.ViserServer()

# The MJCF visual model at the same config, as ground truth to overlay on.
model = mujoco.MjModel.from_xml_path(MJCF)
data = mujoco.MjData(model)
data.qpos[:] = Q
mujoco.mj_kinematics(model, data)
for g in range(model.ngeom):
    visual = model.geom_contype[g] == 0 and model.geom_conaffinity[g] == 0
    if not visual or model.geom_type[g] != mujoco.mjtGeom.mjGEOM_MESH:
        continue
    mid = model.geom_dataid[g]
    va, vn = model.mesh_vertadr[mid], model.mesh_vertnum[mid]
    fa, fn = model.mesh_faceadr[mid], model.mesh_facenum[mid]
    R = data.geom_xmat[g].reshape(3, 3)
    server.scene.add_mesh_simple(
        f"/robot/{g}", model.mesh_vert[va:va + vn] @ R.T + data.geom_xpos[g],
        model.mesh_face[fa:fa + fn], color=(150, 150, 155))

# z maps onto -x: turns a zero-length stub from the (undefined) tube axis
# onto the joint rotation axis, which is what the physical housing follows.
ROT_TO_JOINT_AXIS = np.array([[0.0, 0.0, -1.0], [0.0, 1.0, 0.0], [1.0, 0.0, 0.0]])

for i in range(6):
    T = robot.get_cylinder_transform(i, Q, robot.get_link_z_offset(i))
    axis, p = T[:3, 2], T[:3, 3]
    a = UR5E_A[i - 1] if i > 0 else 0.0
    radius = robot.get_cylinder_radius(i)
    R = T[:3, :3]
    if abs(a) > 1e-9:
        length = abs(a)
        center = p - 0.5 * a * axis  # segment spans [p, p - a * axis]
    else:
        length = 0.12  # zero-length link: stub along the joint axis
        center = p
        R = R @ ROT_TO_JOINT_AXIS
    mesh = trimesh.creation.cylinder(radius=radius, height=length)
    server.scene.add_mesh_simple(
        f"/link_{i}", mesh.vertices @ R.T + center, mesh.faces,
        color=COLORS[i], opacity=0.55, side="double")
    server.scene.add_label(f"/label_{i}", f"link {i}",
                           position=center + np.array([0.0, 0.0, 0.06]))

server.scene.add_frame("/base", axes_length=0.2, axes_radius=0.005)
print(f"config q = {Q}; translucent cylinder i should hug the matching link")
while True:
    time.sleep(1.0)
