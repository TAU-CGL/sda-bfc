"""The end-to-end belief experiment: plan at X_hat, execute at X, calibrate.

X is the true placement of arm B (BASE_OFFSET).  X_hat is a noisy belief
(uniform +-1 cm / +-1 deg per axis by default).  Per touch attempt, entirely
in the BELIEF: sample a forearm-forearm touch, retract to a 10 cm standoff,
plan A's move and B's transit with AORRTC under the uncertainty pad u
(derived from the noise ranges).  Then execute in the TRUTH (MuJoCo scene at
X, analytic cylinders at X): transits must be collision-free (the pad's
guarantee), and the approach overshoots past the believed touch until the
TRUE forearm gap crosses zero -- watching MuJoCo for any wrong-pair contact.

A success is a true forearm-forearm tangency.  After N successes, the Newton
solver recovers X from the touches alone and is compared against the truth.

Usage: python scripts/experiment_belief_touches.py [--touches 11] [--seed 0]
"""

import argparse
import time

import mujoco
import numpy as np

from sda_bfc import ContactSampler, CylinderPose, SolverNewton, UR5e
from sda_bfc import robot as rb
from sda_bfc.collision import _build_model
from sda_bfc.config import HOME_Q, MJCF_PATH
from sda_bfc.maneuver import _retract_steps, densify
from sda_bfc.robot import DualRobot, Robot, offset_matrix, set_uncertainty
from sda_bfc.rrt import rrt_transit

REACH = 1.0                                # max point distance from B's base
VALID_MARGIN = 0.005                        # non-forearm clearance at the touch


class TruthSim:
    """MuJoCo scene at the TRUE placement: the wrong-pair contact monitor."""

    def __init__(self, X):
        self.model = _build_model(X, str(MJCF_PATH), 0.0)
        self.data = mujoco.MjData(self.model)
        self.qadr = {
            p: np.array([self.model.joint(j).qposadr[0]
                         for j in range(self.model.njnt)
                         if self.model.joint(j).name.startswith(p)])
            for p in ("r1/", "r2/")}
        self.forearm_pair = frozenset(
            self.model.body(f"{p}forearm_link").id for p in ("r1/", "r2/"))

    def contact_pairs(self, q_a, q_b):
        self.data.qpos[self.qadr["r1/"]] = q_a
        self.data.qpos[self.qadr["r2/"]] = q_b
        mujoco.mj_kinematics(self.model, self.data)
        mujoco.mj_collision(self.model, self.data)
        return {frozenset((self.model.geom_bodyid[c.geom1],
                           self.model.geom_bodyid[c.geom2]))
                for c in self.data.contact[:self.data.ncon]}

    def clean(self, q_a, q_b, allow_forearms=False):
        pairs = self.contact_pairs(q_a, q_b)
        if allow_forearms:
            pairs.discard(self.forearm_pair)
        return not pairs


def true_forearm_sd(fk, X, q_a, q_b, r):
    A = fk.get_cylinder_transform(3, q_a)
    B = X @ fk.get_cylinder_transform(3, q_b)
    return CylinderPose.from_se3(A, r).signed_distance(CylinderPose.from_se3(B, r))


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--touches", type=int, default=11)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--noise-t-cm", type=float, default=1.0,
                    help="per-axis translation noise bound [cm]")
    ap.add_argument("--noise-r-deg", type=float, default=1.0,
                    help="per-axis rotation noise bound [deg]")
    ap.add_argument("--budget", type=float, default=3.0)
    ap.add_argument("--max-attempts", type=int, default=800)
    args = ap.parse_args()
    noise_t, noise_r = args.noise_t_cm * 1e-2, np.radians(args.noise_r_deg)
    rng = np.random.default_rng(args.seed)
    fk = UR5e()
    r = fk.get_link_radius(4)

    X = offset_matrix()                                    # ground truth
    delta = np.concatenate([rng.uniform(-noise_t, noise_t, 3),
                            rng.uniform(-noise_r, noise_r, 3)])
    x_hat = rb.BASE_OFFSET + delta                         # the belief
    X_hat = offset_matrix(x_hat)
    u = float(np.linalg.norm(delta[:3])
              + 2.0 * np.sin(0.5 * np.linalg.norm(delta[3:])) * REACH)
    u = max(u, np.sqrt(3) * noise_t
            + 2.0 * np.sin(0.5 * np.sqrt(3) * noise_r) * REACH)  # range bound
    standoff = max(0.10, u + 0.05)  # the padded planner needs room at q_pre
    print(f"belief error: |dt| = {np.linalg.norm(delta[:3])*1e3:.1f} mm, "
          f"|dr| = {np.degrees(np.linalg.norm(delta[3:])):.2f} deg; "
          f"uncertainty pad u = {u*1e3:.1f} mm")

    # the simulated truth is the bare two-arm MJCF scene: no bench, stands or
    # floor exist there, so the belief must not plan against them either
    rb.set_walls(False)
    rb.set_stands(False)
    rb.set_obstacles(False)

    ctx = DualRobot(Robot(name="A"), Robot(X_hat, name="B"))  # belief world
    sampler = ContactSampler(fk, X_hat, 3, 3)
    truth = TruthSim(X)

    def padded_blocked(mover, obstacle, q, clearance=0.01):
        set_uncertainty(u)
        try:
            return ctx._blocked(mover, obstacle, q, clearance)
        finally:
            set_uncertainty(0.0)

    stats = {"sampler": 0, "invalid": 0, "retract": 0, "plan": 0,
             "exec": 0, "wrong-pair": 0, "no-touch": 0}
    touches, attempt, t0 = [], 0, time.perf_counter()
    while len(touches) < args.touches and attempt < args.max_attempts:
        attempt += 1
        pose = sampler.sample(attempt)
        if pose is None:
            stats["sampler"] += 1
            continue
        q_a, q_b = np.array(pose.q_a), np.array(pose.q_b)
        ctx.b.set_arm(q_b)
        ok = False
        for _ in range(30):  # wrists cannot move the contact: re-roll them
            if not (ctx.gaps(ctx.a, ctx.b, q_a, ignore_forearm_pair=True)
                    < VALID_MARGIN).any():
                ok = True
                break
            q_a[3:] = rng.uniform(-np.pi, np.pi, 3)
            q_b[3:] = rng.uniform(-np.pi, np.pi, 3)
            ctx.b.set_arm(q_b)
        if not ok:
            stats["invalid"] += 1
            continue

        # retract B from the believed touch until the standoff pose is safe
        # to hand to the padded planner (the touch itself lives inside the
        # unsafety region -- the pad only governs the free-space legs)
        ctx.a.set_arm(q_a)
        back = _retract_steps(
            ctx, ctx.b, ctx.a, q_b, "gradient", HOME_Q,
            stop=lambda q, gap: gap >= standoff
            and not padded_blocked(ctx.b, ctx.a, q))
        if back is None:
            stats["retract"] += 1
            continue
        q_pre = back[-1]

        # plan both arms' moves in the belief, under the pad
        set_uncertainty(u)
        ctx.b.set_arm(HOME_Q)
        path_a, info_a = rrt_transit(ctx, ctx.a, ctx.b, HOME_Q, q_a,
                                     budget=args.budget)
        ctx.a.set_arm(q_a)
        path_b, info_b = rrt_transit(ctx, ctx.b, ctx.a, HOME_Q, q_pre,
                                     budget=args.budget)
        set_uncertainty(0.0)
        if not path_a or not path_b:
            stats["plan"] += 1
            continue

        # execute in the TRUTH: transits must be clean (the pad's promise)
        if any(not truth.clean(q, HOME_Q) for q in densify(path_a)) or \
                any(not truth.clean(q_a, q) for q in densify(path_b)):
            stats["exec"] += 1
            continue

        # approach: overshoot past the believed touch until the TRUE gap
        # closes, watching for any wrong-pair contact
        direction = q_b - q_pre
        step = 0.01 / max(np.linalg.norm(direction), 1e-9)
        t, sd_prev, outcome = 0.0, None, None
        while t <= 2.5:
            q = q_pre + t * direction
            if not truth.clean(q_a, q, allow_forearms=True):
                outcome = "wrong-pair"
                break
            sd = true_forearm_sd(fk, X, q_a, q, r)
            if sd <= 0.0:
                lo, hi = t - step, t
                for _ in range(40):  # bisect the tangency
                    mid = 0.5 * (lo + hi)
                    if true_forearm_sd(fk, X, q_a, q_pre + mid * direction, r) <= 0:
                        hi = mid
                    else:
                        lo = mid
                touches.append((q_a.copy(), q_pre + hi * direction))
                outcome = "touch"
                break
            sd_prev, t = sd, t + step
        if outcome == "touch":
            print(f"  touch {len(touches):2d}/{args.touches} "
                  f"(attempt {attempt}, plans {info_a['method']}/{info_b['method']})")
        else:
            stats[outcome or "no-touch"] += 1
    set_uncertainty(0.0)

    n = len(touches)
    print(f"\n{n} touches from {attempt} attempts "
          f"({time.perf_counter() - t0:.0f} s); failures: "
          + ", ".join(f"{k}={v}" for k, v in stats.items() if v))
    if n < 3:
        raise SystemExit("not enough touches to calibrate")

    As = [fk.get_cylinder_transform(3, qa) for qa, _ in touches]
    Bs = [fk.get_cylinder_transform(3, qb) for _, qb in touches]
    t1 = time.perf_counter()
    X_sol = SolverNewton(As, Bs, r, r).solve_multistart()
    t_solve = time.perf_counter() - t1
    et = np.linalg.norm(X_sol[:3, 3] - X[:3, 3])
    cos = np.clip((np.trace(X_sol[:3, :3].T @ X[:3, :3]) - 1) / 2, -1, 1)
    er = np.degrees(np.arccos(cos))
    e0 = np.linalg.norm(X_hat[:3, 3] - X[:3, 3])
    print(f"\nSolverNewton ({n} touches, {t_solve:.2f} s):")
    print(f"  belief error before: {e0*1e3:8.3f} mm")
    print(f"  translation error:   {et*1e3:11.6f} mm   "
          f"(improvement x{e0/max(et, 1e-12):.0f})")
    print(f"  rotation error:      {er:11.7f} deg")


if __name__ == "__main__":
    main()
