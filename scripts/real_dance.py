"""Blind touch dance on two real UR5e arms via ur_rtde.

Blue arm (origin frame) : 192.168.56.101
Orange arm (unknown X)  : 192.168.57.101

Protocol mirrors blind_touch_dance.py: guarded joint-space segments, one arm
moving at a time.  While one arm moves, a ContactSentinel monitors the OTHER
(static) arm's joint velocities and motor currents at ~250 Hz; any deviation
beyond the calibrated noise band triggers an immediate stopJ of the mover.
The most-distal responding joint of the static arm attributes the contact to
a link: torques from a force on cylinder k propagate only to joints 0..k-1,
so attributed link = most distal responding joint + 1.  A touch is recorded
for calibration only when the static arm attributes the contact to its
forearm (link 3).

Bring-up order (do NOT skip):
  1. python3 real_dance.py --test-sentinel   arms far apart, nothing moves;
     tap each link of each arm by hand and check detection + attribution.
  2. python3 real_dance.py --dry-run         connects, reads states, checks
     reachability of every posture with moveJ disabled.
  3. python3 real_dance.py --execute         runs the dance.  Keep a hand on
     the e-stop.  Speeds are capped at APPROACH_SPEED (5 cm/s at the tip).

The script assumes both robots are e-series, in Remote Control mode, with
matching payload/TCP configuration and clear space below both home columns.
"""

import argparse
import sys
import threading
import time

import numpy as np

try:
    from rtde_control import RTDEControlInterface
    from rtde_receive import RTDEReceiveInterface
except ImportError:
    print("ur_rtde not installed: pip install ur_rtde")
    sys.exit(1)

import blind_touch_dance as dance

BLUE_IP = "192.168.56.101"
ORANGE_IP = "192.168.57.101"

MONITOR_PERIOD = 0.004
BASELINE_SECONDS = 2.0
CURRENT_SIGMA_GAIN = 6.0
CURRENT_FLOOR_A = 0.15
VELOCITY_FLOOR_RAD_S = 0.01
DEBOUNCE_SAMPLES = 3

TRAVEL_SPEED = 0.4
TRAVEL_ACCEL = 0.8
APPROACH_SPEED = 0.1
APPROACH_ACCEL = 0.5
STOP_DECEL = 3.0
SETTLE_SECONDS = 0.5


class ContactSentinel:
    """Watches a static arm's joint velocities and currents for a bump."""

    def __init__(self, receive, name):
        self.receive = receive
        self.name = name
        self.mean = np.zeros(6)
        self.std = np.ones(6)
        self.triggered = threading.Event()
        self.affected_joints = None
        self._stop = threading.Event()
        self._thread = None

    def calibrate(self, seconds=BASELINE_SECONDS):
        samples = []
        t_end = time.time() + seconds
        while time.time() < t_end:
            samples.append(self.receive.getActualCurrent())
            time.sleep(MONITOR_PERIOD)
        samples = np.array(samples)
        self.mean = samples.mean(axis=0)
        self.std = samples.std(axis=0)
        print(f"[{self.name}] baseline current mean {np.round(self.mean, 3)} "
              f"std {np.round(self.std, 3)}")

    def _threshold(self):
        return np.maximum(CURRENT_SIGMA_GAIN * self.std, CURRENT_FLOOR_A)

    def _loop(self):
        threshold = self._threshold()
        strikes = np.zeros(6, int)
        while not self._stop.is_set():
            current = np.abs(np.array(self.receive.getActualCurrent()) - self.mean)
            velocity = np.abs(self.receive.getActualQd())
            hit = (current > threshold) | (velocity > VELOCITY_FLOOR_RAD_S)
            strikes = np.where(hit, strikes + 1, 0)
            if np.any(strikes >= DEBOUNCE_SAMPLES):
                self.affected_joints = np.where(strikes >= DEBOUNCE_SAMPLES)[0]
                self.triggered.set()
                return
            time.sleep(MONITOR_PERIOD)

    def arm(self):
        self.triggered.clear()
        self.affected_joints = None
        self._stop.clear()
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def disarm(self):
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=1.0)

    def attributed_link(self):
        if self.affected_joints is None or len(self.affected_joints) == 0:
            return None
        return int(self.affected_joints.max()) + 1


class RealArm:
    def __init__(self, ip, name, execute):
        self.name = name
        self.execute = execute
        self.control = RTDEControlInterface(ip) if execute else None
        self.receive = RTDEReceiveInterface(ip)
        self.sentinel = ContactSentinel(self.receive, name)

    def q(self):
        return np.array(self.receive.getActualQ())

    def stop(self):
        if self.control is not None:
            self.control.stopJ(STOP_DECEL)

    def move_guarded(self, target, other, speed=APPROACH_SPEED,
                     accel=APPROACH_ACCEL):
        """Async moveJ toward target while the OTHER arm's sentinel watches.

        Returns (reached, attributed_link_of_static_arm, contact_q)."""
        if not self.execute:
            print(f"[dry-run] {self.name} moveJ -> {np.round(target, 3)}")
            return True, None, None
        other.sentinel.calibrate(0.5)
        other.sentinel.arm()
        self.control.moveJ(list(target), speed, accel, True)
        try:
            while True:
                if other.sentinel.triggered.is_set():
                    self.stop()
                    time.sleep(SETTLE_SECONDS)
                    link = other.sentinel.attributed_link()
                    print(f"[{self.name}] contact felt by {other.name}, "
                          f"joints {other.sentinel.affected_joints}, "
                          f"attributed link {link}")
                    return False, link, self.q()
                if self.control.getAsyncOperationProgress() < 0:
                    return True, None, None
                time.sleep(MONITOR_PERIOD)
        finally:
            other.sentinel.disarm()

    def close(self):
        if self.control is not None:
            self.control.stopScript()
        self.receive.disconnect()


def retreat(arm, other, waypoint):
    reached, _, _ = arm.move_guarded(waypoint, other, TRAVEL_SPEED, TRAVEL_ACCEL)
    if not reached:
        raise RuntimeError(f"{arm.name}: contact during retreat -- "
                           "workspace inconsistent, stopping for safety")


def perform_real_dance(blue, orange):
    touches = []

    def sweep_pass(theta0, rung, tilt, sign, qA):
        target = dance.orange_config(
            theta0 + sign * (2.0 * np.pi - dance.SWEEP_MARGIN), rung, tilt)
        reached, link, qB = orange.move_guarded(target, blue)
        if not reached and link == dance.DH_LINK and qB is not None:
            touches.append((qA.copy(), qB.copy()))
            print(f"  touch #{len(touches)} recorded")
        retreat(orange, blue, dance.orange_config(theta0, rung, tilt))
        return not reached

    blue_home = dance.fold_config(0.0)
    orange_home = dance.fold_config(0.0)
    retreat(blue, orange, blue_home)
    retreat(orange, blue, orange_home)

    for direction in dance.A_DIRECTIONS:
        retreat(blue, orange, dance.fold_config(direction))
        for blue_tilt in dance.BLUE_TILTS:
            reached, _, _ = blue.move_guarded(
                dance.blue_config(direction, blue_tilt), orange)
            if not reached:
                retreat(blue, orange, dance.fold_config(direction))
                break
            qA = dance.blue_config(direction, blue_tilt)
            found_any = False
            theta0 = orange.q()[0]
            for rung in dance.ORANGE_RUNGS:
                reached, _, _ = orange.move_guarded(
                    dance.orange_config(theta0, rung), blue)
                if not reached:
                    retreat(orange, blue, dance.fold_config(theta0))
                    continue
                tilts = [0.0]
                for tilt in tilts:
                    hit_cw = sweep_pass(theta0, rung, tilt, +1, qA)
                    hit_ccw = sweep_pass(theta0, rung, tilt, -1, qA) if hit_cw else False
                    if (hit_cw or hit_ccw) and tilt == 0.0:
                        tilts.extend(dance.ORANGE_TILTS)
                        found_any = True
                    if len(touches) >= dance.TARGET_TOUCHES:
                        break
                retreat(orange, blue, dance.orange_config(theta0, rung))
                if len(touches) >= dance.TARGET_TOUCHES:
                    break
            retreat(orange, blue, dance.fold_config(theta0))
            retreat(blue, orange, dance.fold_config(direction))
            if not found_any or len(touches) >= dance.TARGET_TOUCHES:
                break
        if len(touches) >= dance.TARGET_TOUCHES:
            break
    return touches


def test_sentinel(blue, orange):
    print("Tap each link by hand; Ctrl-C to finish.")
    for arm in [blue, orange]:
        arm.sentinel.calibrate()
    try:
        while True:
            for arm in [blue, orange]:
                arm.sentinel.arm()
            while not (blue.sentinel.triggered.is_set()
                       or orange.sentinel.triggered.is_set()):
                time.sleep(MONITOR_PERIOD)
            for arm in [blue, orange]:
                arm.sentinel.disarm()
                if arm.sentinel.triggered.is_set():
                    print(f"[{arm.name}] joints {arm.sentinel.affected_joints} "
                          f"-> attributed link {arm.sentinel.attributed_link()}")
            time.sleep(1.0)
    except KeyboardInterrupt:
        pass


def main():
    parser = argparse.ArgumentParser()
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--test-sentinel", action="store_true")
    mode.add_argument("--dry-run", action="store_true")
    mode.add_argument("--execute", action="store_true")
    args = parser.parse_args()

    if args.execute:
        answer = input("Arms will MOVE and intentionally touch. Clear the cell, "
                       "hold the e-stop, type YES to continue: ")
        if answer.strip() != "YES":
            sys.exit(0)

    blue = RealArm(BLUE_IP, "blue", args.execute)
    orange = RealArm(ORANGE_IP, "orange", args.execute)
    try:
        if args.test_sentinel:
            test_sentinel(blue, orange)
            return
        touches = perform_real_dance(blue, orange)
        print(f"\n{len(touches)} touches collected")
        if len(touches) >= dance.MIN_TOUCHES:
            np.savez("real_dance_touches.npz",
                     qA=np.array([t[0] for t in touches]),
                     qB=np.array([t[1] for t in touches]))
            X = dance.calibrate(touches)
            np.set_printoptions(precision=6, suppress=True)
            print("estimated base offset X (orange base in blue frame):\n", X)
        else:
            print("not enough touches for calibration")
    finally:
        for arm in [blue, orange]:
            try:
                arm.stop()
                arm.close()
            except Exception:
                pass


if __name__ == "__main__":
    main()
