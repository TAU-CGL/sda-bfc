"""Study: touch generation under placement uncertainty.

For each uncertainty scale (base range 1 cm / 1 deg per axis, scaled), run
several placements; each session tries to collect 7 solver-usable touches
(forearm-forearm, interior, at the TRUE placement) with guarded execution
and RRT planning that only ever see the believed placement.  Reports success
rates, attempt counts, and wall time -- and how far the uncertainty can grow
before the pipeline stops being practical.
"""

import argparse
import sys
import time
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from planning.uncertain_touch import UncertainTouchSession  # noqa: E402

BASE_RANGES = (0.01, 0.01, 0.01,
               np.radians(1.0), np.radians(1.0), np.radians(1.0))
NUM_TOUCHES = 7
MAX_ATTEMPTS = 30


def run_session(args):
    scale, seed = args
    ranges = tuple(scale * r for r in BASE_RANGES)
    t0 = time.perf_counter()
    session = UncertainTouchSession(seed=seed, ranges=ranges)
    attempts = session.collect(NUM_TOUCHES, MAX_ATTEMPTS)
    outcomes = Counter(a["outcome"] for a in attempts)
    return {
        "scale": scale,
        "seed": seed,
        "touches": len(session.touches),
        "attempts": len(attempts),
        "plan_time": sum(a["plan_time"] for a in attempts),
        "wall_time": time.perf_counter() - t0,
        "outcomes": dict(outcomes),
    }


def main(scales, placements, workers):
    jobs = [(scale, seed) for scale in scales for seed in range(placements)]
    with ProcessPoolExecutor(max_workers=workers) as pool:
        results = list(pool.map(run_session, jobs))

    print(f"{'scale':>6s} {'range':>14s} {'>=7 rate':>9s} {'touch/attempt':>14s} "
          f"{'attempts':>9s} {'wall s':>7s} {'plan s':>7s}   outcomes")
    for scale in scales:
        rows = [r for r in results if r["scale"] == scale]
        complete = np.mean([r["touches"] >= NUM_TOUCHES for r in rows])
        touches = sum(r["touches"] for r in rows)
        attempts = sum(r["attempts"] for r in rows)
        outcomes = Counter()
        for r in rows:
            outcomes.update(r["outcomes"])
        fails = {k: v for k, v in outcomes.items() if k != "success"}
        print(f"{scale:6.1f} {scale:5.0f}cm/{scale:.0f}deg "
              f"{100 * complete:8.0f}% {touches / max(attempts, 1):14.2f} "
              f"{attempts / len(rows):9.1f} "
              f"{np.mean([r['wall_time'] for r in rows]):7.1f} "
              f"{np.mean([r['plan_time'] for r in rows]):7.1f}   {dict(fails)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--scales", type=float, nargs="+",
                        default=[1.0, 2.0, 4.0, 6.0, 8.0, 12.0])
    parser.add_argument("--placements", type=int, default=8)
    parser.add_argument("--workers", type=int, default=10)
    args = parser.parse_args()
    main(args.scales, args.placements, args.workers)
