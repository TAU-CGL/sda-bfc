"""Goal-biased RRT with shortcut smoothing in R2's 6-dim joint space."""

import numpy as np


def rrt_plan(world, q_start, q_goal, rng, max_iters=3000, step=0.3,
             goal_bias=0.15):
    q_start = np.asarray(q_start, float)
    q_goal = np.asarray(q_goal, float)
    if not world.config_free(q_start) or not world.config_free(q_goal):
        return None
    if world.edge_free(q_start, q_goal):
        return shortcut(world, [q_start, q_goal], rng)
    nodes = [q_start]
    parents = [-1]
    for _ in range(max_iters):
        target = q_goal if rng.random() < goal_bias \
            else rng.uniform(-np.pi, np.pi, 6)
        arr = np.array(nodes)
        i = int(np.argmin(((arr - target) ** 2).sum(axis=1)))
        delta = target - nodes[i]
        length = np.linalg.norm(delta)
        q_new = target if length <= step else nodes[i] + delta * (step / length)
        if not world.edge_free(nodes[i], q_new):
            continue
        nodes.append(q_new)
        parents.append(i)
        if np.linalg.norm(q_new - q_goal) < step and world.edge_free(q_new, q_goal):
            nodes.append(q_goal)
            parents.append(len(nodes) - 2)
            path = []
            j = len(nodes) - 1
            while j >= 0:
                path.append(nodes[j])
                j = parents[j]
            path.reverse()
            return shortcut(world, path, rng)
    return None


def shortcut(world, path, rng, tries=60):
    path = [np.asarray(p, float) for p in path]
    for _ in range(tries):
        if len(path) <= 2:
            break
        i, j = sorted(rng.integers(0, len(path), 2))
        if j - i < 2:
            continue
        if world.edge_free(path[i], path[j]):
            path = path[:i + 1] + path[j:]
    return path
