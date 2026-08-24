"""Benchmark of all base-offset solvers on the measured touch-pose data.

Each solver runs once per seed; accuracy is measured against the calibrated
ground-truth base transform.  Note the data's own noise floor is ~4-5 mm
(touch residuals scatter +-1-2 mm), so no solver can beat that.
"""

import argparse
import importlib.util
import time
from pathlib import Path

import numpy as np

from sda_bfc import (SolverAdam, SolverAnnealingLP, SolverNewton, SolverSMC,
                     UR5e)

_spec = importlib.util.spec_from_file_location(
    "touch_poses", Path(__file__).resolve().parent.parent / "tests" / "test_touch_poses.py")
touch = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(touch)


def build_problem():
    robot = UR5e()
    radius = robot.get_link_radius(touch.RADII_LINK_INDEX)
    As = [robot.get_cylinder_transform(touch.DH_LINK_INDEX, np.array(q1))
          for q1, _ in touch.TOUCH_POSES]
    Bs = [robot.get_cylinder_transform(touch.DH_LINK_INDEX, np.array(q2))
          for _, q2 in touch.TOUCH_POSES]
    return As, Bs, radius


def rotation_error_deg(X, X_gt):
    R_delta = X[:3, :3].T @ X_gt[:3, :3]
    return np.degrees(np.arccos(np.clip((np.trace(R_delta) - 1.0) / 2.0, -1.0, 1.0)))


def benchmark(seeds):
    As, Bs, r = build_problem()
    X_gt = touch.base_transform()

    solvers = {
        "Newton (multistart)": lambda seed: SolverNewton(As, Bs, r, r)
            .solve_multistart(seed=seed),
        "Annealing-LP": lambda seed: SolverAnnealingLP(As, Bs, r, r, seed=seed)
            .solve(),
        "SMC": lambda seed: SolverSMC(As, Bs, r, r, seed=seed).solve(),
        "Adam (multistart)": lambda seed: SolverAdam(As, Bs, r, r)
            .solve_multistart(seed=seed),
    }
    reference = SolverNewton(As, Bs, r, r)

    header = (f"{'solver':22s} {'time (s)':>14s} {'trans err mm':>16s} "
              f"{'rot err deg':>16s} {'cost':>10s}")
    print(header)
    print("-" * len(header))
    for name, run in solvers.items():
        times, terrs, rerrs, costs = [], [], [], []
        for seed in seeds:
            t0 = time.perf_counter()
            X = run(seed)
            times.append(time.perf_counter() - t0)
            terrs.append(np.linalg.norm(X[:3, 3] - X_gt[:3, 3]) * 1000)
            rerrs.append(rotation_error_deg(X, X_gt))
            costs.append(reference.cost(X))
        print(f"{name:22s} {np.mean(times):6.2f} +-{np.std(times):5.2f} "
              f"{np.mean(terrs):7.2f} max{np.max(terrs):6.2f} "
              f"{np.mean(rerrs):7.3f} max{np.max(rerrs):6.3f} "
              f"{np.max(costs):10.2e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, default=8)
    args = parser.parse_args()
    benchmark(range(args.seeds))
