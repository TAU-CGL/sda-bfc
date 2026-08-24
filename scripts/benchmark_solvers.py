"""Solver benchmark on simulator-generated experiments.

For each random placement (ground truth X known to the benchmark, never to
the solvers), the C++ simulator generates k=20 touching joint-space
configurations.  Every solver then solves with the first k contacts for
k = 1..20, so the curve shows how many touches each solver actually needs --
including the underdetermined regime k < 6 where the touch constraints admit
a continuum of solutions.

Success = translation error < 5 mm and rotation error < 0.5 deg (the data is
noiseless, so successes sit at ~1e-9 and failures are far above threshold).
"""

import argparse
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np

from sda_bfc import (SolverAdam, SolverAnnealingLP, SolverNewton, SolverSMC,
                     UR5e, generate_experiment)

MAX_CONTACTS = 20
SUCCESS_TRANS_M = 5e-3
SUCCESS_ROT_DEG = 0.5

SOLVER_NAMES = ["newton", "annealing", "smc", "adam"]


def solve(name, As, Bs, r):
    if name == "newton":
        return SolverNewton(As, Bs, r, r).solve_multistart()
    if name == "annealing":
        return SolverAnnealingLP(As, Bs, r, r).solve()
    if name == "smc":
        return SolverSMC(As, Bs, r, r).solve()
    if name == "adam":
        return SolverAdam(As, Bs, r, r).solve_multistart(num_starts=1000)
    raise ValueError(name)


def rotation_error_deg(X, X_gt):
    R_delta = X[:3, :3].T @ X_gt[:3, :3]
    return np.degrees(np.arccos(np.clip((np.trace(R_delta) - 1.0) / 2.0, -1.0, 1.0)))


def run_placement(seed):
    robot = UR5e()
    r = robot.get_link_radius(4)
    experiment = generate_experiment(MAX_CONTACTS, seed=seed)
    As = [robot.get_cylinder_transform(3, q) for q in experiment.q_as]
    Bs = [robot.get_cylinder_transform(3, q) for q in experiment.q_bs]
    trans = np.zeros((len(SOLVER_NAMES), MAX_CONTACTS))
    rot = np.zeros((len(SOLVER_NAMES), MAX_CONTACTS))
    times = np.zeros(len(SOLVER_NAMES))
    for si, name in enumerate(SOLVER_NAMES):
        t0 = time.perf_counter()
        for k in range(1, MAX_CONTACTS + 1):
            X = solve(name, As[:k], Bs[:k], r)
            trans[si, k - 1] = np.linalg.norm(X[:3, 3] - experiment.X[:3, 3])
            rot[si, k - 1] = rotation_error_deg(X, experiment.X)
        times[si] = (time.perf_counter() - t0) / MAX_CONTACTS
    return trans, rot, times


def main(num_placements, workers):
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=workers) as pool:
        results = list(pool.map(run_placement, range(num_placements)))
    trans = np.stack([t for t, _, _ in results])   # (placements, solvers, k)
    rot = np.stack([r for _, r, _ in results])
    times = np.stack([t for _, _, t in results]).mean(axis=0)
    success = (trans < SUCCESS_TRANS_M) & (rot < SUCCESS_ROT_DEG)
    np.savez("benchmark_results.npz", trans=trans, rot=rot,
             solvers=SOLVER_NAMES)
    print(f"{num_placements} placements x {MAX_CONTACTS} k-values x "
          f"{len(SOLVER_NAMES)} solvers in {time.time() - t0:.0f}s\n")

    header = "k    " + "".join(f"{name:>12s}" for name in SOLVER_NAMES)
    print("success rate (%):")
    print(header)
    for k in range(1, MAX_CONTACTS + 1):
        row = f"{k:<5d}"
        for si in range(len(SOLVER_NAMES)):
            row += f"{100.0 * success[:, si, k - 1].mean():12.0f}"
        print(row)

    print("\nmedian translation error (mm):")
    print(header)
    for k in range(1, MAX_CONTACTS + 1):
        row = f"{k:<5d}"
        for si in range(len(SOLVER_NAMES)):
            row += f"{1000.0 * np.median(trans[:, si, k - 1]):12.2g}"
        print(row)

    print("\nsummary (best k = smallest k with >= 95% success):")
    print(f"{'solver':<12s} {'best k':>7s} {'succ@bestk':>11s} {'succ@20':>8s} "
          f"{'med mm@20':>10s} {'avg solve s':>12s}")
    for si, name in enumerate(SOLVER_NAMES):
        rates = success[:, si, :].mean(axis=0)
        reliable = np.where(rates >= 0.95)[0]
        best_k = reliable[0] + 1 if len(reliable) else None
        best_label = str(best_k) if best_k else "-"
        succ_best = f"{100 * rates[best_k - 1]:.0f}%" if best_k else "-"
        print(f"{name:<12s} {best_label:>7s} {succ_best:>11s} "
              f"{100 * rates[-1]:7.0f}% {1000 * np.median(trans[:, si, -1]):10.2g} "
              f"{times[si]:12.3f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--placements", type=int, default=100)
    parser.add_argument("--workers", type=int, default=10)
    args = parser.parse_args()
    main(args.placements, args.workers)
