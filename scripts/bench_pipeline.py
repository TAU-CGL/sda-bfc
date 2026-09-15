"""The full-pipeline benchmark: plan in the belief, execute in the truth, calibrate.

Per placement, R2's true pose X is random and the belief X_hat is X plus a uniform offset of
up to +-u cm and +-u deg per axis. Touch attempts run exactly as in experiment_belief_touches
-- sample a touch in the belief, retract to a standoff, plan both transits with AORRTC under
the uncertainty pad, execute in the MuJoCo truth, overshoot the approach until the true
forearms meet -- until --touches true contacts are in. Then every solver runs on the subsets
k = kmin..touches, with encoder noise N(0, sigma^2) on the joints, and is scored against X.
Only the annealing LP solver (MC-SA) is evaluated.

Checkpointing: every finished placement, and within a placement every new touch, is written
to <out stem>/ next to the CSV, so a crashed or killed run continues where it left off when
re-run with the same arguments (a placement in progress resumes from its last touch, with the
RNG state restored). Set RESET = True below to delete all of that and start from scratch.
Each placement draws from its own RNG stream -- (seed, placement, draw) for the true pose and
(seed, placement, draw, uncertainty) for the belief offset and the attempts -- so the true poses
are shared across uncertainty levels and no result depends on what ran before it. A placement
that has fewer than --give-up-touches touches after --give-up-after seconds is abandoned and
its slot gets a new random true pose (draw + 1); abandoned draws are kept in the checkpoint
and counted as redraws / abandoned_s in the summary.

Besides the per-k rows (<out>), a per-placement summary (<out stem>_placements.csv) records
how many touches were collected, whether all --touches were reached, and the wall time spent
collecting and evaluating.

Usage: python scripts/bench_pipeline.py [--placements 10] [--uncertainty 1 2] [--touches 20]
"""
import argparse
import json
import os
import shutil
import time
from pathlib import Path

import numpy as np
import pandas as pd

RESET = False   # True: delete this run's checkpoint directory and run everything from scratch
CHECKPOINT_EVERY = 50   # attempts between checkpoints when no touch has landed

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("SDA_BFC_UR5E_MJCF", str(ROOT / "assets/universal_robots_ur5e/ur5e.xml"))

from sda_bfc import ContactSampler, CylinderPose, SolverAnnealingLP, UR5e  # noqa: E402
from sda_bfc import robot as rb  # noqa: E402
from sda_bfc.config import HOME_Q  # noqa: E402
from sda_bfc.maneuver import _retract_steps, densify  # noqa: E402
from sda_bfc.robot import DualRobot, Robot, set_uncertainty  # noqa: E402
from sda_bfc.rrt import rrt_transit  # noqa: E402
from experiment_belief_touches import REACH, VALID_MARGIN, TruthSim, true_forearm_sd  # noqa: E402

FK = UR5e()
R = FK.get_link_radius(4)
SOLVERS = {"annealing_lp": lambda As, Bs, seed: SolverAnnealingLP(As, Bs, R, R, seed=seed).solve()}


def rotvec(w):
    angle = np.linalg.norm(w)
    if angle < 1e-12:
        return np.eye(3)
    k = w / angle
    K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    return np.eye(3) + np.sin(angle) * K + (1 - np.cos(angle)) * K @ K


def random_placement(rng, dist=(0.55, 0.85), z=0.3, tilt_deg=10.0):
    heading, d = rng.uniform(-np.pi, np.pi), rng.uniform(*dist)
    X = np.eye(4)
    X[:3, :3] = rotvec([0, 0, rng.uniform(-np.pi, np.pi)]) @ rotvec(np.radians(rng.uniform(-tilt_deg, tilt_deg, 2)).tolist() + [0])
    X[:3, 3] = [d * np.cos(heading), d * np.sin(heading), rng.uniform(-z, z)]
    return X


def perturb(X, rng, u):
    dt, dr = rng.uniform(-u * 1e-2, u * 1e-2, 3), rng.uniform(-np.radians(u), np.radians(u), 3)
    X_hat = X.copy()
    X_hat[:3, :3] = rotvec(dr) @ X[:3, :3]
    X_hat[:3, 3] += dt
    return X_hat, dt, dr


def errors(X_sol, X):
    cos = np.clip((np.trace(X_sol[:3, :3].T @ X[:3, :3]) - 1) / 2, -1, 1)
    return 1e3 * np.linalg.norm(X_sol[:3, 3] - X[:3, 3]), np.degrees(np.arccos(cos))


def residual(As, Bs, X):
    return 1e3 * max(abs(CylinderPose.from_se3(np.linalg.inv(A) @ X @ B, R).signed_distance(R))
                     for A, B in zip(As, Bs))


def to_jsonable(o):
    if isinstance(o, dict):
        return {k: to_jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [to_jsonable(v) for v in o]
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, np.generic):
        return o.item()
    return o


def save_json(path, obj):
    """Atomic write: a crash mid-write leaves the previous file intact."""
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(to_jsonable(obj)))
    os.replace(tmp, path)


def load_json(path):
    return json.loads(path.read_text()) if path.exists() else None


def fmt_s(seconds):
    seconds = int(round(seconds))
    if seconds < 60:
        return f"{seconds}s"
    if seconds < 3600:
        return f"{seconds // 60}m{seconds % 60:02d}s"
    return f"{seconds // 3600}h{seconds % 3600 // 60:02d}m"


def collect(X, X_hat, u, want, rng, budget, max_attempts, log=print, state=None, checkpoint=None,
            give_up=(np.inf, 0)):
    """The experiment's attempt loop: returns (touches, stats, seconds spent, gave up?).

    `state` resumes a partially collected placement from a dict written by `checkpoint`,
    which is called after every new touch and every CHECKPOINT_EVERY attempts with the
    touches, stats, RNG state and elapsed time so far. `give_up` = (seconds, touches): stop
    with gave_up=True once more than that many seconds have gone by with fewer touches."""
    pad = np.sqrt(3) * u * 1e-2 + 2.0 * np.sin(0.5 * np.sqrt(3) * np.radians(u)) * REACH
    standoff = max(0.10, pad + 0.05)
    ctx = DualRobot(Robot(name="A"), Robot(X_hat, name="B"))
    sampler, truth = ContactSampler(FK, X_hat, 3, 3), TruthSim(X)

    def padded_blocked(mover, obstacle, q, clearance=0.01):
        set_uncertainty(pad)
        try:
            return ctx._blocked(mover, obstacle, q, clearance)
        finally:
            set_uncertainty(0.0)

    stats = dict(attempts=0, sampler=0, invalid=0, retract=0, plan=0, exec=0, wrong_pair=0, no_touch=0)
    touches, elapsed = [], 0.0
    if state:
        touches = [(np.array(q_a), np.array(q_b)) for q_a, q_b in state["touches"]]
        stats.update(state["stats"])
        rng.bit_generator.state = state["rng_state"]
        elapsed = state["elapsed_s"]
    t0 = time.perf_counter()

    def attempt():
        pose = sampler.sample(int(rng.integers(1 << 31)))
        if pose is None:
            return "sampler"
        q_a, q_b = np.array(pose.q_a), np.array(pose.q_b)
        ctx.b.set_arm(q_b)
        for _ in range(30):
            if not (ctx.gaps(ctx.a, ctx.b, q_a, ignore_forearm_pair=True) < VALID_MARGIN).any():
                break
            q_a[3:], q_b[3:] = rng.uniform(-np.pi, np.pi, 3), rng.uniform(-np.pi, np.pi, 3)
            ctx.b.set_arm(q_b)
        else:
            return "invalid"
        ctx.a.set_arm(q_a)
        back = _retract_steps(ctx, ctx.b, ctx.a, q_b, "gradient", HOME_Q,
                              stop=lambda q, gap: gap >= standoff and not padded_blocked(ctx.b, ctx.a, q))
        if back is None:
            return "retract"
        q_pre = back[-1]
        set_uncertainty(pad)
        ctx.b.set_arm(HOME_Q)
        path_a, _ = rrt_transit(ctx, ctx.a, ctx.b, HOME_Q, q_a, budget=budget)
        ctx.a.set_arm(q_a)
        path_b, _ = rrt_transit(ctx, ctx.b, ctx.a, HOME_Q, q_pre, budget=budget)
        set_uncertainty(0.0)
        if not path_a or not path_b:
            return "plan"
        if any(not truth.clean(q, HOME_Q) for q in densify(path_a)) or \
                any(not truth.clean(q_a, q) for q in densify(path_b)):
            return "exec"
        direction = q_b - q_pre
        step = 0.01 / max(np.linalg.norm(direction), 1e-9)
        t = 0.0
        while t <= 2.5:
            q = q_pre + t * direction
            if not truth.clean(q_a, q, allow_forearms=True):
                return "wrong_pair"
            if true_forearm_sd(FK, X, q_a, q, R) <= 0.0:
                lo, hi = t - step, t
                for _ in range(40):
                    mid = 0.5 * (lo + hi)
                    lo, hi = (lo, mid) if true_forearm_sd(FK, X, q_a, q_pre + mid * direction, R) <= 0 else (mid, hi)
                touches.append((q_a.copy(), q_pre + hi * direction))
                return "touch"
            t += step
        return "no_touch"

    gave_up = False
    while len(touches) < want and stats["attempts"] < max_attempts:
        if elapsed + time.perf_counter() - t0 > give_up[0] and len(touches) < give_up[1]:
            gave_up = True
            break
        stats["attempts"] += 1
        outcome = attempt()
        if outcome == "touch":
            spent = elapsed + time.perf_counter() - t0
            log(f"    touch {len(touches)}/{want} (attempt {stats['attempts']}, {fmt_s(spent)} elapsed, "
                f"ETA {fmt_s(spent / len(touches) * (want - len(touches)))})")
        else:
            stats[outcome] += 1
        if checkpoint and (outcome == "touch" or stats["attempts"] % CHECKPOINT_EVERY == 0):
            checkpoint(dict(touches=touches, stats=stats, rng_state=rng.bit_generator.state,
                            elapsed_s=elapsed + time.perf_counter() - t0))
    return touches, stats, elapsed + time.perf_counter() - t0, gave_up


def evaluate(touches, X, solvers, noise, rng, seed, kmin, kstep, log=print, **tags):
    q = np.asarray(touches, float) + rng.normal(0, noise, (len(touches), 2, 6))
    As = [FK.get_cylinder_transform(3, a) for a, _ in q]
    Bs = [FK.get_cylinder_transform(3, b) for _, b in q]
    rows = []
    for k in range(kmin, len(q) + 1, kstep):
        for name in solvers:
            t0 = time.perf_counter()
            X_sol = np.asarray(SOLVERS[name](As[:k], Bs[:k], seed))
            dt = time.perf_counter() - t0
            t_err, r_err = errors(X_sol, X)
            rows.append(dict(tags, k=k, solver=name, t_err_mm=t_err, r_err_deg=r_err,
                             residual_mm=residual(As[:k], Bs[:k], X_sol), time_s=dt))
            log(f"    {name:13s} k={k:2d}  {t_err:9.3f} mm  {r_err:8.4f} deg  res {rows[-1]['residual_mm']:.3f} mm  {dt:5.2f} s")
    return rows


def run_placement(a, u, i, ckpt):
    """One (uncertainty, placement) cell: returns (record, loaded from checkpoint?)."""
    done, partial = ckpt / f"u{u:g}_p{i}.json", ckpt / f"u{u:g}_p{i}.partial.json"
    rec = load_json(done)
    if rec is not None:
        s = rec["summary"]
        print(f"uncertainty {u:g}, placement {i}: done earlier ({s['n_touches']}/{s['want']} touches "
              f"in {s['collect_s']:.0f} s, {s['redraws']} redraws), loaded from {done.name}", flush=True)
        return rec, True

    state = load_json(partial)
    resumes, draw, abandoned = 0, 0, []
    if state:
        resumes, draw, abandoned = state["resumes"] + 1, state["draw"], state["abandoned"]
    while True:   # draws for this slot: redraw the true pose if the current one gives up
        X = random_placement(np.random.default_rng([a.seed, i, draw]))
        rng = np.random.default_rng([a.seed, i, draw, int(round(u * 1e6))])
        X_hat, dt, dr = perturb(X, rng, u)
        off_mm, off_deg = 1e3 * np.linalg.norm(dt), np.degrees(np.linalg.norm(dr))
        print(f"uncertainty {u:g}, placement {i}" + (f" (redraw {draw})" if draw else "") +
              f": belief off by {off_mm:.1f} mm, {off_deg:.2f} deg", flush=True)
        if state:
            print(f"  resuming from {partial.name}: {len(state['touches'])} touches, "
                  f"{state['stats']['attempts']} attempts, {state['elapsed_s']:.0f} s so far", flush=True)
        touches, stats, collect_s, gave_up = collect(
            X, X_hat, u, a.touches, rng, a.budget, a.max_attempts, state=state,
            give_up=(a.give_up_after, a.give_up_touches),
            checkpoint=lambda snap: save_json(partial, dict(snap, resumes=resumes, draw=draw, abandoned=abandoned)))
        if not gave_up:
            break
        print(f"  gave up: only {len(touches)}/{a.touches} touches after {fmt_s(collect_s)}; "
              + ", ".join(f"{k}={v}" for k, v in stats.items() if v) + "; drawing a new placement", flush=True)
        abandoned.append(dict(draw=draw, n_touches=len(touches), collect_s=collect_s, **stats))
        draw, state = draw + 1, None
        save_json(partial, dict(touches=[], stats={}, rng_state=None, elapsed_s=0.0,
                                resumes=resumes, draw=draw, abandoned=abandoned))
    success, evaluated = len(touches) >= a.touches, len(touches) >= a.kmin
    print(f"  {len(touches)}/{a.touches} touches in {collect_s:.0f} s"
          + ("" if success else "  ** INCOMPLETE **" if evaluated else "  ** TOO FEW TO EVALUATE **")
          + "; " + ", ".join(f"{k}={v}" for k, v in stats.items() if v), flush=True)

    tags = dict(uncertainty=u, placement=i, n_touches=len(touches), success=success, collect_s=collect_s, **stats)
    t0 = time.perf_counter()
    rows = evaluate(touches, X, list(SOLVERS), a.noise, rng, a.seed, a.kmin, a.kstep, **tags) if evaluated else []
    summary = dict(tags, want=a.touches, evaluated=evaluated, evaluate_s=time.perf_counter() - t0,
                   belief_off_mm=off_mm, belief_off_deg=off_deg, resumes=resumes, draw=draw,
                   redraws=len(abandoned), abandoned_s=sum(x["collect_s"] for x in abandoned))
    rec = dict(X=X, X_hat=X_hat, dt=dt, dr=dr, touches=touches, rows=rows, summary=summary, abandoned=abandoned)
    save_json(done, rec)
    partial.unlink(missing_ok=True)
    return rec, False


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--placements", type=int, default=100)
    ap.add_argument("--uncertainty", type=float, nargs="+", default=[1.0, 2.0, 3.0, 4.0, 5.0],
                    help="belief offset bounds, +-u cm and +-u deg per axis; one run per value")
    ap.add_argument("--touches", type=int, default=20)
    ap.add_argument("--kmin", type=int, default=6)
    ap.add_argument("--kstep", type=int, default=1)
    ap.add_argument("--noise", type=float, default=0.00085, help="encoder noise std [rad]")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--budget", type=float, default=3.0, help="AORRTC budget per transit [s]")
    ap.add_argument("--max-attempts", type=int, default=8000)
    ap.add_argument("--give-up-after", type=float, default=300.0,
                    help="seconds after which a placement with fewer than --give-up-touches touches is "
                         "abandoned and a new random placement is drawn for its slot")
    ap.add_argument("--give-up-touches", type=int, default=5)
    ap.add_argument("--out", type=Path, default=None)
    a = ap.parse_args()
    rb.set_walls(False), rb.set_stands(False), rb.set_obstacles(False)   # the truth is the bare two-arm MJCF

    out = a.out or ROOT / "results" / f"bench_pipeline_seed{a.seed}_noise{a.noise:g}.csv"
    ckpt = out.with_suffix("")
    if RESET and ckpt.exists():
        print(f"RESET: deleting {ckpt}", flush=True)
        shutil.rmtree(ckpt)
    ckpt.mkdir(parents=True, exist_ok=True)
    config = {k: getattr(a, k) for k in ("seed", "touches", "kmin", "kstep", "noise", "budget", "max_attempts")}
    previous = load_json(ckpt / "config.json")
    if previous is not None and previous != config:
        raise SystemExit(f"{ckpt} holds checkpoints of a run with different settings {previous}; "
                         f"set RESET = True to discard them or pass a different --out")
    save_json(ckpt / "config.json", config)
    cells = [(u, i) for u in a.uncertainty for i in range(a.placements)]
    n_done = sum((ckpt / f"u{u:g}_p{i}.json").exists() for u, i in cells)
    if n_done:
        print(f"resuming: {n_done}/{len(cells)} placements already in {ckpt}", flush=True)

    recs, t_start, ran = [], time.perf_counter(), 0
    for n, (u, i) in enumerate(cells, 1):
        rec, loaded = run_placement(a, u, i, ckpt)
        recs.append(rec)
        ran += not loaded
        elapsed = time.perf_counter() - t_start
        left = len(cells) - n
        eta = f"ETA {fmt_s(elapsed / ran * left)}" if ran and left else ("done" if not left else "ETA unknown")
        print(f"  [{n}/{len(cells)} placements, {fmt_s(elapsed)} elapsed, "
              f"{fmt_s(elapsed / ran) if ran else '-'} per placement, {eta}]", flush=True)
    df = pd.DataFrame([row for rec in recs for row in rec["rows"]])
    summary = pd.DataFrame([rec["summary"] for rec in recs])
    df.to_csv(out, index=False)
    summary_out = out.with_name(out.stem + "_placements.csv")
    summary.to_csv(summary_out, index=False)

    pd.set_option("display.width", 200), pd.set_option("display.max_columns", 20)
    print(f"\nper uncertainty: success = all {a.touches} touches collected; evaluated = at least kmin={a.kmin}")
    print(summary.groupby("uncertainty").agg(
        placements=("placement", "count"), success_rate=("success", "mean"), evaluated=("evaluated", "sum"),
        touches_mean=("n_touches", "mean"), touches_min=("n_touches", "min"),
        attempts_mean=("attempts", "mean"), collect_s_median=("collect_s", "median"),
        collect_s_max=("collect_s", "max"), evaluate_s_median=("evaluate_s", "median"),
        redraws=("redraws", "sum"), abandoned_s=("abandoned_s", "sum")).round(3))
    incomplete = summary[~summary.success]
    if len(incomplete):
        print("\nincomplete placements:")
        print(incomplete[["uncertainty", "placement", "n_touches", "want", "attempts", "collect_s"]].to_string(index=False))
    if len(df):
        df["fail"] = (df.t_err_mm > 10) | (df.r_err_deg > 1)
        for col in ("t_err_mm", "r_err_deg", "fail", "time_s"):
            print(f"\n{'median ' if col != 'fail' else 'rate of '}{col}")
            print(df.pivot_table(index="uncertainty", columns="k", values=col,
                                 aggfunc="median" if col != "fail" else "mean").round(4))
    print(f"\nwrote {out} and {summary_out}")


if __name__ == "__main__":
    main()
