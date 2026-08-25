#pragma once

#include <vector>

#include "../sim/arm_capsules.hpp"
#include "../sim/gjk.hpp"

namespace sda_bfc {

    // The base column (links 0, 1) stands on the floor by construction;
    // workcell halfspaces constrain only the movable links.
    inline constexpr int kFirstWorkcellLink = 2;

    class PlanningWorld {
    public:
        virtual ~PlanningWorld() = default;
        virtual bool isFree(const JointConfig& q) const = 0;

        bool edgeFree(const JointConfig& q0, const JointConfig& q1,
                      double step = 0.05) const {
            double span = (q1 - q0).cwiseAbs().maxCoeff();
            int n = std::max(1, (int)std::ceil(span / step));
            for (int k = 0; k <= n; k++) {
                if (!isFree(q0 + (q1 - q0) * ((double)k / n))) return false;
            }
            return true;
        }
    };

    // R2 plans at its believed base against the uncertainty-expanded capsules
    // of the static arm plus workcell halfspaces.  For the approach corridor,
    // the dynamic arm's distal assembly (forearm + wrists) may be exempted
    // against exactly one obstacle -- the touch target's expanded hull --
    // while every other pair stays protected.
    class BeliefWorld : public PlanningWorld {
    public:
        BeliefWorld(const ForwardKinematics<6>& fk, const SE3& xBelief,
                    const JointConfig& qStatic, const UncertaintyRanges& ranges)
            : model_(fk),
              R_(xBelief.block<3,3>(0, 0)), t_(xBelief.block<3,1>(0, 3)) {
            for (const Capsule& c : model_.capsules(qStatic)) {
                obstacles_.push_back(expandCapsule(c, ranges));
            }
        }

        void addHalfspace(const Halfspace& halfspace) {
            halfspaces_.push_back(halfspace);
        }

        bool isFree(const JointConfig& q) const override {
            return freeExcept(q, 0, -1, kNumCapsuleLinks);
        }

        // Pairs (m, s) with allowedStaticLo <= s <= allowedStaticHi and
        // m >= allowedMovingMin are exempt from the obstacle check.
        bool freeExcept(const JointConfig& q, int allowedStaticLo,
                        int allowedStaticHi, int allowedMovingMin) const {
            auto capsules = model_.capsules(q);
            if (selfCollides(capsules)) return false;
            std::array<Capsule, kNumCapsuleLinks> world;
            for (int i = 0; i < kNumCapsuleLinks; i++) {
                world[i] = {R_ * capsules[i].a + t_, R_ * capsules[i].b + t_,
                            capsules[i].r};
                if (i >= kFirstWorkcellLink) {
                    for (const Halfspace& halfspace : halfspaces_) {
                        if (!halfspace.admits(world[i])) return false;
                    }
                }
            }
            for (int m = 0; m < kNumCapsuleLinks; m++) {
                for (int s = 0; s < (int)obstacles_.size(); s++) {
                    if (s >= allowedStaticLo && s <= allowedStaticHi
                        && m >= allowedMovingMin) continue;
                    if (cloudIntersectsCapsule(obstacles_[s], world[m].a,
                                               world[m].b, world[m].r)) {
                        return false;
                    }
                }
            }
            return true;
        }

        bool corridorEdgeFree(const JointConfig& q0, const JointConfig& q1,
                              int allowedStaticLo, int allowedStaticHi,
                              int allowedMovingMin, double step = 0.02) const {
            double span = (q1 - q0).cwiseAbs().maxCoeff();
            int n = std::max(1, (int)std::ceil(span / step));
            for (int k = 0; k <= n; k++) {
                if (!freeExcept(q0 + (q1 - q0) * ((double)k / n),
                                allowedStaticLo, allowedStaticHi,
                                allowedMovingMin)) {
                    return false;
                }
            }
            return true;
        }

    private:
        static bool selfCollides(
                const std::array<Capsule, kNumCapsuleLinks>& capsules) {
            for (int i = 0; i < kNumCapsuleLinks; i++) {
                for (int j = i + 2; j < kNumCapsuleLinks; j++) {
                    if (capsuleClearance(capsules[i], capsules[j]) <= 0.0) {
                        return true;
                    }
                }
            }
            return false;
        }

        ArmCapsuleModel model_;
        Eigen::Matrix3d R_;
        R3 t_;
        std::vector<PointCloud> obstacles_;
        std::vector<Halfspace> halfspaces_;
    };

}
