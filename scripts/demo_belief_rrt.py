"""Self-contained demo: belief-space RRT for two UR5e arms (MuJoCo + viser).

Robot R1 sits at the origin (pose I); robot R2 sits at a random pose X.
The planner never sees X -- only a noisy belief X_hat ~ N(X, sigma).
R1 is frozen at a random collision-free config.  An RRT plans a path for R2
between two random collision-free configs *in the believed scene*; the path
is then executed (kinematic playback) in the *true* scene, stopping at the
first contact and reporting which links collided and where.

MuJoCo runs headless as the collision oracle; viser serves the visualization
in the browser (open the printed URL).  GUI buttons: Play (resumes, or
replays a finished run), Pause, and Randomize (samples + plans a fresh
scenario).

Dependencies: pip install mujoco viser numpy
(the UR5e MJCF is vendored from mujoco_menagerie under assets/)
Usage: python scripts/demo_belief_rrt.py [--seed N] [--sigma-t 0.03] [--no-viz]
"""

import argparse
import time
from pathlib import Path

import mujoco
import numpy as np

UR5E_MJCF = str(Path(__file__).resolve().parent.parent
                / "assets" / "universal_robots_ur5e" / "ur5e.xml")
PREFIXES = ("r1/", "r2/")


# ------------------------------------------------------------- SE(3) helpers

def rotvec_to_mat(w):
    theta = np.linalg.norm(w)
    if theta < 1e-12:
        return np.eye(3)
    k = w / theta
    K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    return np.eye(3) + np.sin(theta) * K + (1 - np.cos(theta)) * (K @ K)


def mat_to_quat(R):
    quat = np.empty(4)
    mujoco.mju_mat2Quat(quat, np.asarray(R, dtype=np.float64).ravel())
    return quat


def sample_placement(rng, min_dist=0.55, max_dist=0.8):
    """Random SE(3) pose of R2's base: planar offset from R1 plus a yaw."""
    heading = rng.uniform(-np.pi, np.pi)
    dist = rng.uniform(min_dist, max_dist)
    X = np.eye(4)
    X[:3, :3] = rotvec_to_mat(np.array([0.0, 0.0, rng.uniform(-np.pi, np.pi)]))
    X[:3, 3] = [dist * np.cos(heading), dist * np.sin(heading), 0.0]
    return X


def perturb_placement(X, rng, sigma_t, sigma_r):
    """X_hat ~ N(X): Gaussian translation + Gaussian rotation-vector noise."""
    Xh = X.copy()
    Xh[:3, :3] = rotvec_to_mat(rng.normal(0.0, sigma_r, 3)) @ X[:3, :3]
    Xh[:3, 3] += rng.normal(0.0, sigma_t, 3)
    return Xh


# ----------------------------------------------------------- MuJoCo two-arm sim

def build_model(X2):
    """Compile one MuJoCo model holding both arms: R1 at I, R2 at X2."""
    spec = mujoco.MjSpec()
    spec.option.integrator = mujoco.mjtIntegrator.mjINT_IMPLICITFAST
    for prefix, X in zip(PREFIXES, (np.eye(4), X2)):
        arm = mujoco.MjSpec.from_file(UR5E_MJCF)
        for key in list(arm.keys):  # duplicated keyframes break the second attach
            arm.delete(key)
        frame = spec.worldbody.add_frame(pos=X[:3, 3], quat=mat_to_quat(X[:3, :3]))
        spec.attach(arm, prefix=prefix, frame=frame)
    return spec.compile()


class TwoArmSim:
    """Kinematic collision oracle: set both configs, query contacts."""

    def __init__(self, X2):
        self.model = build_model(X2)
        self.data = mujoco.MjData(self.model)
        jids = {prefix: [j for j in range(self.model.njnt)
                         if self.model.joint(j).name.startswith(prefix)]
                for prefix in PREFIXES}
        self.qadr = {prefix: self.model.jnt_qposadr[ids]
                     for prefix, ids in jids.items()}
        self.lo, self.hi = np.clip(
            self.model.jnt_range[jids["r2/"]].T, -np.pi, np.pi)

    def set_config(self, q1, q2):
        self.data.qpos[self.qadr["r1/"]] = q1
        self.data.qpos[self.qadr["r2/"]] = q2
        mujoco.mj_kinematics(self.model, self.data)
        mujoco.mj_collision(self.model, self.data)

    def is_free(self, q1, q2):
        self.set_config(q1, q2)
        return self.data.ncon == 0

    def contacts(self):
        """List of (body_a, body_b, world position, penetration depth)."""
        out = []
        for c in self.data.contact[:self.data.ncon]:
            body = lambda g: self.model.body(self.model.geom_bodyid[g]).name
            out.append((body(c.geom1), body(c.geom2), c.pos.copy(), -c.dist))
        return out


# ---------------------------------------------------------------------- RRT

def interpolate(qa, qb, step):
    n = max(int(np.ceil(np.max(np.abs(qb - qa)) / step)), 1)
    return [qa + s * (qb - qa) for s in np.linspace(0.0, 1.0, n + 1)]


def densify(path, step=0.02):
    qs = [path[0]]
    for qa, qb in zip(path, path[1:]):
        qs.extend(interpolate(qa, qb, step)[1:])
    return qs


class BeliefWorld:
    """Collision checking as the planner sees it: R2's base at X_hat."""

    def __init__(self, sim, q1_static):
        self.sim = sim
        self.q1 = q1_static

    def is_free(self, q2):
        return self.sim.is_free(self.q1, q2)

    def edge_free(self, qa, qb, step=0.05):
        return all(self.is_free(q) for q in interpolate(qa, qb, step))


def rrt(world, start, goal, rng, max_iters=4000, step=0.4, goal_bias=0.1):
    nodes, parents = [start], [-1]
    for _ in range(max_iters):
        target = goal if rng.uniform() < goal_bias else \
            rng.uniform(world.sim.lo, world.sim.hi)
        near = int(np.argmin(np.linalg.norm(np.array(nodes) - target, axis=1)))
        direction = target - nodes[near]
        dist = np.linalg.norm(direction)
        q_new = target if dist <= step else nodes[near] + step * direction / dist
        if not world.edge_free(nodes[near], q_new):
            continue
        nodes.append(q_new)
        parents.append(near)
        if np.array_equal(q_new, goal) or (
                np.linalg.norm(q_new - goal) <= step
                and world.edge_free(q_new, goal)):
            nodes.append(goal)
            parents.append(len(nodes) - 2)
            path, i = [], len(nodes) - 1
            while i >= 0:
                path.append(nodes[i])
                i = parents[i]
            return path[::-1]
    return None


def shortcut(path, world, rng, tries=100):
    path = list(path)
    for _ in range(tries):
        if len(path) <= 2:
            break
        i, j = sorted(rng.choice(len(path), size=2, replace=False))
        if j - i >= 2 and world.edge_free(path[i], path[j]):
            del path[i + 1:j]
    return path


# ------------------------------------------------------------------ scenarios

def make_scenario(seed, sigma_t, sigma_r):
    """Sample placement, belief, and configs; plan in the belief.

    Returns None if the RRT fails for this seed.
    """
    rng = np.random.default_rng(seed)
    X = sample_placement(rng)
    X_hat = perturb_placement(X, rng, sigma_t, sigma_r)
    belief_sim = TwoArmSim(X_hat)
    sample_q = lambda: rng.uniform(belief_sim.lo, belief_sim.hi)
    while True:  # random static R1 pose + free start for R2 (in the belief)
        q1, q2_start = sample_q(), sample_q()
        if belief_sim.is_free(q1, q2_start):
            break
    while True:
        q2_goal = sample_q()
        if belief_sim.is_free(q1, q2_goal):
            break
    world = BeliefWorld(belief_sim, q1)
    path = rrt(world, q2_start, q2_goal, rng)
    if path is None:
        return None
    path = shortcut(path, world, rng)
    return {"seed": seed, "X": X, "X_hat": X_hat, "q1": q1, "path": path,
            "sim": TwoArmSim(X)}


def next_scenario(seed, sigma_t, sigma_r, tries=20):
    for s in range(seed, seed + tries):
        scenario = make_scenario(s, sigma_t, sigma_r)
        if scenario is not None:
            err = np.linalg.norm(scenario["X_hat"][:3, 3] - scenario["X"][:3, 3])
            print(f"scenario seed {s}: belief error |dt| = {err:.4f} m, "
                  f"{len(scenario['path'])} waypoints")
            return scenario
    raise SystemExit(f"RRT failed for {tries} seeds starting at {seed}")


# ----------------------------------------------------------------- execution

def execute(sim, q1, path, step=0.02, on_step=None):
    """Play the path on the true scene; stop and report at first contact."""
    qs = densify(path, step)
    for k, q2 in enumerate(qs):
        sim.set_config(q1, q2)
        if on_step is not None:
            on_step()
        if sim.data.ncon:
            return k / max(len(qs) - 1, 1), q2, sim.contacts()
    return 1.0, qs[-1], []


def report(progress, q2, contacts):
    if contacts:
        print(f"\nCOLLISION at {100 * progress:.1f}% of the path:")
        for body_a, body_b, pos, depth in contacts:
            print(f"  {body_a}  <->  {body_b}   at {np.round(pos, 3)}"
                  f"   depth {1e3 * depth:.2f} mm")
        print(f"  q2 = {np.round(q2, 3)}")
    else:
        print("path executed collision-free (belief error was benign here)")


# --------------------------------------------------------------------- viser

def start_viewer(sim):
    """Mirror the model's visual meshes into a viser scene.

    Geom ids stay valid across scenarios: every scenario compiles the same
    two-arm structure, only the base placement differs.
    """
    import viser
    server = viser.ViserServer()
    tints = {"r1/": (120, 145, 190), "r2/": (235, 145, 55)}
    m = sim.model
    handles = []
    for g in range(m.ngeom):
        visual = m.geom_contype[g] == 0 and m.geom_conaffinity[g] == 0
        if not visual or m.geom_type[g] != mujoco.mjtGeom.mjGEOM_MESH:
            continue
        mid = m.geom_dataid[g]
        va, vn = m.mesh_vertadr[mid], m.mesh_vertnum[mid]
        fa, fn = m.mesh_faceadr[mid], m.mesh_facenum[mid]
        body_name = m.body(m.geom_bodyid[g]).name
        handle = server.scene.add_mesh_simple(
            f"/geoms/{g}", m.mesh_vert[va:va + vn], m.mesh_face[fa:fa + fn],
            color=tints[body_name[:3]])
        handles.append((g, handle))
    frames = {name: server.scene.add_frame(f"/{name}", axes_length=0.15,
                                           axes_radius=0.004)
              for name in ("X_true", "X_believed")}
    return {"server": server, "handles": handles, "frames": frames,
            "markers": []}


def sync_viewer(viewer, sim):
    with viewer["server"].atomic():
        for g, handle in viewer["handles"]:
            handle.position = sim.data.geom_xpos[g].copy()
            handle.wxyz = mat_to_quat(sim.data.geom_xmat[g])


def show_scenario(viewer, scenario):
    for name, X in (("X_true", scenario["X"]), ("X_believed", scenario["X_hat"])):
        viewer["frames"][name].position = X[:3, 3]
        viewer["frames"][name].wxyz = mat_to_quat(X[:3, :3])
    for marker in viewer["markers"]:
        marker.remove()
    viewer["markers"].clear()


def mark_contacts(viewer, contacts):
    for _, _, pos, _ in contacts:
        viewer["markers"].append(viewer["server"].scene.add_icosphere(
            f"/contacts/{len(viewer['markers'])}", radius=0.03,
            color=(220, 40, 40), position=pos))


# ---------------------------------------------------------------------- main

def run_interactive(scenario, args, sigma_r):
    viewer = start_viewer(scenario["sim"])
    server = viewer["server"]
    state = {"playing": True, "randomize": False}
    server.gui.add_button("Play").on_click(
        lambda _: state.update(playing=True))
    server.gui.add_button("Pause").on_click(
        lambda _: state.update(playing=False))
    server.gui.add_button("Randomize").on_click(
        lambda _: state.update(randomize=True))
    status = server.gui.add_markdown("")

    qs, k, done = [], 0, False

    def load(scenario):
        nonlocal qs, k, done
        qs, k, done = densify(scenario["path"]), 0, False
        show_scenario(viewer, scenario)
        scenario["sim"].set_config(scenario["q1"], qs[0])
        sync_viewer(viewer, scenario["sim"])
        status.content = (f"**seed {scenario['seed']}** — "
                          f"{len(scenario['path'])} waypoints")

    load(scenario)
    next_seed = scenario["seed"] + 1
    while True:
        if state["randomize"]:
            state["randomize"] = False
            scenario = next_scenario(next_seed, args.sigma_t, sigma_r)
            next_seed = scenario["seed"] + 1
            load(scenario)
            state["playing"] = True
        elif state["playing"] and done:
            load(scenario)  # replay from the start
        elif state["playing"]:
            k += 1
            sim = scenario["sim"]
            sim.set_config(scenario["q1"], qs[k])
            sync_viewer(viewer, sim)
            progress = k / max(len(qs) - 1, 1)
            if sim.data.ncon:
                contacts = sim.contacts()
                report(progress, qs[k], contacts)
                mark_contacts(viewer, contacts)
                state["playing"], done = False, True
                pairs = ", ".join(f"{a} ↔ {b}" for a, b, _, _ in contacts)
                status.content = (f"**seed {scenario['seed']}** — COLLISION at "
                                  f"{100 * progress:.1f}%: {pairs}")
            elif k == len(qs) - 1:
                report(progress, qs[k], [])
                state["playing"], done = False, True
                status.content = (f"**seed {scenario['seed']}** — "
                                  "executed collision-free")
        time.sleep(1.0 / args.rate)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--sigma-t", type=float, default=0.03,
                    help="translation noise stddev on the belief [m]")
    ap.add_argument("--sigma-r-deg", type=float, default=3.0,
                    help="rotation noise stddev on the belief [deg]")
    ap.add_argument("--rate", type=float, default=60.0,
                    help="playback steps per second")
    ap.add_argument("--no-viz", action="store_true")
    args = ap.parse_args()
    sigma_r = np.radians(args.sigma_r_deg)

    scenario = next_scenario(args.seed, args.sigma_t, sigma_r)
    if args.no_viz:
        report(*execute(scenario["sim"], scenario["q1"], scenario["path"]))
        return
    run_interactive(scenario, args, sigma_r)


if __name__ == "__main__":
    main()
