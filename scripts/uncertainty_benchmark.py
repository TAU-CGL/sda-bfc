"""Benchmark: touch generation under uncertainty + SolverNewton calibration.

For each uncertainty scale (base +-1 cm / +-1 deg per axis) and placement:
collect 7 solver-usable touches with the mesh-collision oracle (URDF
collision geometry -- the fair simulation), calibrate the base offset with
the prior-aware SolverNewton, and report:

  * RRT/planning time per attempt and per collected touch
  * touch success rate and >=7-touch completion rate
  * calibration precision (translation / rotation error vs ground truth)
  * certainty improvement factor: initial translation uncertainty
    (half-diagonal of the range box) / final translation error

The --prune flag compares the deep-backoff pruning heuristic.
"""

import argparse
import functools
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from collections import Counter

from planning import MeshOracle, Outcome, TouchSession, calibrate
from planning.belief import PlanningParams  # noqa: E402

BASE_RANGES = (0.01, 0.01, 0.01,
               np.radians(1.0), np.radians(1.0), np.radians(1.0))
NUM_TOUCHES = 11
MAX_ATTEMPTS = 60


@dataclass
class SessionReport:
    scale: float
    prune: bool
    mesh_source: str
    touches: int
    attempts: int
    successes: int
    failures: dict
    plan_seconds: float
    wall_seconds: float
    trans_error_m: float
    rot_error_deg: float


def scaled_ranges(scale):
    return tuple(scale * r for r in BASE_RANGES)


def initial_uncertainty_m(scale):
    return float(np.linalg.norm(np.array(BASE_RANGES[:3]) * scale))


def rotation_error_deg(x, x_true):
    cos = (np.trace(x[:3, :3].T @ x_true[:3, :3]) - 1.0) / 2.0
    return float(np.degrees(np.arccos(np.clip(cos, -1.0, 1.0))))


def calibration_errors(session):
    if len(session.touches) < NUM_TOUCHES:
        return np.nan, np.nan
    x_est = calibrate(session.robot, session.touches,
                      radius=session.oracle.touch_radius)
    if x_est is None:
        return np.nan, np.nan
    trans = float(np.linalg.norm(x_est[:3, 3] - session.x_true[:3, 3]))
    return trans, rotation_error_deg(x_est, session.x_true)


def run_session(job):
    scale, seed, prune, mesh_source = job
    params = PlanningParams(max_backoff=0.2 if prune else None)
    oracle_factory = functools.partial(MeshOracle, mesh_source=mesh_source)
    started = time.perf_counter()
    session = TouchSession(seed=seed, range_tuple=scaled_ranges(scale),
                           oracle_factory=oracle_factory, params=params)
    attempts = session.collect(NUM_TOUCHES, MAX_ATTEMPTS)
    trans_error, rot_error = calibration_errors(session)
    failures = Counter(a.outcome.value for a in attempts
                       if a.outcome is not Outcome.SUCCESS)
    return SessionReport(
        scale=scale, prune=prune, mesh_source=mesh_source,
        touches=len(session.touches),
        attempts=len(attempts),
        successes=sum(a.outcome is Outcome.SUCCESS for a in attempts),
        failures=dict(failures),
        plan_seconds=sum(a.plan_seconds for a in attempts),
        wall_seconds=time.perf_counter() - started,
        trans_error_m=trans_error, rot_error_deg=rot_error)


def summarize(rows, scale, prune, mesh_source):
    group = [r for r in rows if r.scale == scale and r.prune == prune
             and r.mesh_source == mesh_source]
    if not group:
        return None
    complete = np.mean([r.touches >= NUM_TOUCHES for r in group])
    attempts = sum(r.attempts for r in group)
    rate = sum(r.successes for r in group) / max(attempts, 1)
    plan = np.mean([r.plan_seconds / max(r.attempts, 1) for r in group])
    with np.errstate(all="ignore"):
        trans = np.nanmedian([r.trans_error_m for r in group])
        rot = np.nanmedian([r.rot_error_deg for r in group])
    factor = initial_uncertainty_m(scale) / trans if trans > 0 else np.nan
    failures = Counter()
    for r in group:
        failures.update(r.failures)
    top = failures.most_common(1)[0][0] if failures else "-"
    return complete, rate, plan, trans, rot, factor, top


def print_table(rows, scales, prunes, mesh_sources):
    header = (f"{'source':>10s} {'scale':>6s} {'prune':>6s} {'>=k':>5s} "
              f"{'succ/att':>9s} {'plan s/att':>11s} {'trans err':>10s} "
              f"{'rot err':>8s} {'improve x':>10s}")
    print(header)
    print("-" * len(header))
    for mesh_source in mesh_sources:
        for scale in scales:
            for prune in prunes:
                summary = summarize(rows, scale, prune, mesh_source)
                if summary is None:
                    continue
                complete, rate, plan, trans, rot, factor, top = summary
                print(f"{mesh_source:>10s} {scale:6.1f} {str(prune):>6s} "
                      f"{100 * complete:4.0f}% {rate:9.2f} {plan:11.1f} "
                      f"{1000 * trans:8.2f}mm {rot:7.3f} {factor:10.1f}  {top}")


def main(scales, placements, workers, prunes, mesh_sources):
    jobs = [(scale, seed, prune, source) for source in mesh_sources
            for scale in scales for prune in prunes
            for seed in range(placements)]
    with ProcessPoolExecutor(max_workers=workers) as pool:
        rows = list(pool.map(run_session, jobs))
    print(f"{len(jobs)} sessions, {NUM_TOUCHES} touches, "
          f"cap {MAX_ATTEMPTS} attempts\n")
    print_table(rows, scales, prunes, mesh_sources)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--scales", type=float, nargs="+",
                        default=[1.0, 2.0, 4.0, 6.0, 8.0])
    parser.add_argument("--placements", type=int, default=8)
    parser.add_argument("--workers", type=int, default=10)
    parser.add_argument("--prune", action="store_true",
                        help="also run the deep-backoff pruning variant")
    parser.add_argument("--mesh-sources", nargs="+",
                        default=["collision", "visual"])
    args = parser.parse_args()
    main(args.scales, args.placements, args.workers,
         [False, True] if args.prune else [False], args.mesh_sources)
