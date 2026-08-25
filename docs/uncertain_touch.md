# Touch Generation Under Placement Uncertainty

*Notes from the sda-bfc simulation study — generating link-link touches
between two UR5e arms when the second arm's base placement is only known up
to an uncertainty range.*

## The problem

Two UR5e arms: R1 at a known base (the origin), R2 at an unknown true
placement **X_gt**. The robot only has a belief **X_initial**, off from the
truth by up to a range box in (x, y, z, roll, pitch, yaw). We want the arms
to deliberately touch forearm-to-forearm — the touches feed a calibration
solver that recovers the true placement — but every motion must be planned
with the belief alone, and nothing may pass through anything.

**Ground rule**: X_gt exists only inside the simulator's collision oracle
(the stand-in for real force sensing). Every planning computation sees
X_initial and the range, nothing more.

## How collision works — two separate mechanisms

- **Ground truth ("physics")**: each arm is 6 capsules (segment + radius,
  calibrated against the real UR5e, including the 138 mm lateral offset of
  the upper-arm tube). Motions execute in small joint steps; each step
  checks all 36 cross-arm capsule pairs with exact segment-segment distances
  against R2-at-X_gt. The first pair reaching zero clearance stops the
  motion, bisected onto the contact surface — the simulated "feel" of a
  guarded real arm.
- **Belief side (the planner's model)**: R1's capsules are *expanded* for
  the uncertainty — sample the range box (translation corners × three
  samples per rotation axis, applied inversely), transform each capsule's
  bounding-box vertices, take the convex hull, pad slightly for the rotation
  arcs. A plan that avoids the expanded hulls is collision-free for *every*
  placement in the box. R2's capsules at X_initial are tested against the
  hulls with a conservative facet-plane check.

## One touch attempt

1. Generate a touching candidate (qA, qB) with a joint-space Newton solver,
   pretending the placement is X_initial.
2. Filter cheaply in the belief, before any motion: R1's path to qA must
   not sweep the expanded believed R2-home; R2's home must be outside the
   expanded R1; a *pre-touch* config must exist (back one of qB's first
   three joints off until it clears the expansion); an RRT path from home
   to pre-touch must exist.
3. Move R1 home → qA, guarded. Unexpected contact → retreat, fail.
4. Execute R2's RRT path, guarded. Premature contact → retreat along the
   executed trail, fail.
5. **Guarded approach**: drive the backed-off joint through qB plus
   overshoot until contact is *felt*.
6. Success only if the felt pair is forearm-forearm **and** interior to
   both cylinders — i.e., a touch the calibration solver can use exactly.
   Otherwise: near-miss, retreat, retry.

## Results (8 placements per scale, 7 touches required, 30-attempt cap)

| Uncertainty (±cm / ±deg per axis) | Placements reaching 7 touches | Touches per attempt | Avg attempts | Avg sim time |
|---|---|---|---|---|
| 1 | 100% | 0.84 | 8.4 | 95 s |
| 2 | 100% | 0.75 | 9.4 | 86 s |
| 4 | 100% | 0.51 | 13.8 | 106 s |
| 6 | 88% | 0.32 | 20.8 | 95 s |
| 8 | 75% | 0.21 | 26.5 | 49 s |
| 12 | 0% | 0.02 | cap | — |

## Insights

- **Realistic uncertainty is comfortable.** At ±1–2 cm/deg, ~0.8 usable
  touches per attempt: 7 poses in ~9 attempts, ~1.5 minutes of simulation
  per placement (nearly all planning; execution is milliseconds).
- **The practical ceiling is ±6–8 cm/deg.** 88% of placements still finish
  at 6, 75% at 8, zero at 12.
- **The failure mode shifts with scale, and that's the interesting part.**
  Small box: failures are near-misses — right link pair but end-cap
  contact, or an adjacent link touched first; the belief was slightly wrong
  about *where along the forearm* contact lands. Large box: the dominant
  failure is "no viable candidate" — the expanded obstacles swallow the
  workspace and the planner cannot even find a pre-touch config. The
  ceiling is set by conservative planning, not by execution luck.
- **Belief-side filtering matters enormously.** Rejecting doomed candidates
  before moving (R1 sweeping the believed R2-home, unreachable pre-touch
  configs) cut wasted physical attempts from ~40% to nearly zero.
- **Interior contacts are non-negotiable.** A capsule end-cap touch
  satisfies "we felt contact" but violates the solver's infinite-line touch
  model and would bias calibration; the pipeline rejects them.
- **The touches collected are exact constraints at the true placement**, so
  they feed straight into the calibration solvers — the natural loop
  (touch → calibrate → shrink uncertainty → touch again) is the next step.

## Reproducing

```bash
python3 scripts/uncertainty_study.py                 # the table above
python3 visualization/uncertain_touch_scene.py       # interactive scene
```

The viewer shows the true arm (meshes), the robot's *belief ghost*
(translucent capsules at X_initial — exactly the planner's model), the
expanded collision hulls, the executed forearm trace, and an animation of
the whole guarded attempt, including retreats on failure.
