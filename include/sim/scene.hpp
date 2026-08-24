#pragma once

#include <algorithm>
#include <limits>
#include <vector>

#include "arm_capsules.hpp"

namespace sda_bfc {

    struct PairClearance {
        int i, j;
        double clearance;
    };

    // The "candle" home posture: whole arm folded into a vertical column.
    inline JointConfig foldConfig(double yaw) {
        JointConfig q;
        q << yaw, -M_PI_2, 0.0, 0.0, 0.0, 0.0;
        return q;
    }

    // Arm A at the identity, arm B at base transform X.
    class TwoArmScene {
    public:
        TwoArmScene(const ForwardKinematics<6>& fk, const SE3& X) :
            model_(fk), X_(X) {}

        const SE3& baseOffset() const { return X_; }
        const ArmCapsuleModel& model() const { return model_; }

        std::vector<PairClearance> selfClearances(const JointConfig& q) const {
            auto caps = model_.capsules(q);
            std::vector<PairClearance> out;
            for (int i = 0; i < kNumCapsuleLinks; i++) {
                for (int j = i + 2; j < kNumCapsuleLinks; j++) {
                    out.push_back({i, j, capsuleClearance(caps[i], caps[j])});
                }
            }
            return out;
        }

        std::vector<PairClearance> crossClearances(const JointConfig& qA,
                                                   const JointConfig& qB) const {
            auto capsA = model_.capsules(qA);
            auto capsB = model_.capsules(qB, X_);
            std::vector<PairClearance> out;
            for (int i = 0; i < kNumCapsuleLinks; i++) {
                for (int j = 0; j < kNumCapsuleLinks; j++) {
                    out.push_back({i, j, capsuleClearance(capsA[i], capsB[j])});
                }
            }
            return out;
        }

        PairClearance firstCrossContact(const JointConfig& qA,
                                        const JointConfig& qB) const {
            PairClearance best{-1, -1, std::numeric_limits<double>::infinity()};
            for (const PairClearance& pc : crossClearances(qA, qB)) {
                if (pc.clearance < best.clearance) best = pc;
            }
            return best;
        }

        double minSelfClearance(const JointConfig& q) const {
            double best = std::numeric_limits<double>::infinity();
            for (const PairClearance& pc : selfClearances(q)) {
                best = std::min(best, pc.clearance);
            }
            return best;
        }

        SegmentClosest closestOnPair(const JointConfig& qA, const JointConfig& qB,
                                     int i, int j) const {
            Capsule a = model_.capsules(qA)[i];
            Capsule b = model_.capsules(qB, X_)[j];
            return segmentClosest(a.a, a.b, b.a, b.b);
        }

        // Side-to-side touch: closest points strictly interior to both
        // segments, so segment contact coincides with the solvers'
        // infinite-line contact model.
        bool contactInterior(const JointConfig& qA, const JointConfig& qB,
                             int i, int j, double margin = 0.03) const {
            SegmentClosest sc = closestOnPair(qA, qB, i, j);
            return margin < sc.s && sc.s < 1.0 - margin
                && margin < sc.t && sc.t < 1.0 - margin;
        }

        bool collisionFreeAtHome(double yawA = 0.0, double yawB = 0.0) const {
            JointConfig qA = foldConfig(yawA), qB = foldConfig(yawB);
            if (minSelfClearance(qA) <= 0.0 || minSelfClearance(qB) <= 0.0) {
                return false;
            }
            for (const PairClearance& pc : crossClearances(qA, qB)) {
                if (pc.clearance <= 0.0) return false;
            }
            return true;
        }

        // No collision anywhere, except pair (ti, tj) which must be within
        // touchTol of exact contact and interior.
        bool validContact(const JointConfig& qA, const JointConfig& qB,
                          int ti, int tj, double touchTol = 1e-6,
                          double othersMargin = 5e-3,
                          double interiorMargin = 0.03) const {
            if (minSelfClearance(qA) <= 0.0 || minSelfClearance(qB) <= 0.0) {
                return false;
            }
            for (const PairClearance& pc : crossClearances(qA, qB)) {
                if (pc.i == ti && pc.j == tj) {
                    if (std::abs(pc.clearance) > touchTol) return false;
                } else if (pc.clearance <= othersMargin) {
                    return false;
                }
            }
            return contactInterior(qA, qB, ti, tj, interiorMargin);
        }

    private:
        ArmCapsuleModel model_;
        SE3 X_;
    };

}
