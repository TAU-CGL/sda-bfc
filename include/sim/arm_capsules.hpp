#pragma once

#include <array>
#include <utility>

#include "../fk/fk.hpp"
#include "capsule.hpp"

namespace sda_bfc {

    using JointConfig = ForwardKinematics<6>::JointSpaceConfig;

    inline constexpr int kNumCapsuleLinks = 6;
    inline constexpr int kForearmLink = 3;       // DH link index
    inline constexpr int kForearmRadiusIndex = 4; // radii index (DH 3 <-> radii 4)
    inline constexpr double kHousingLength = 0.12;
    // DH link i -> radii index: i for the base links, i+1 from the forearm on.
    inline constexpr std::array<int, kNumCapsuleLinks> kRadiusIndex = {0, 1, 2, 4, 5, 6};

    class ArmCapsuleModel {
    public:
        explicit ArmCapsuleModel(const ForwardKinematics<6>& fk) : fk_(fk) {
            for (int i = 0; i < kNumCapsuleLinks; i++) {
                double aIn = i > 0 ? fk_.getDhA(i - 1) : 0.0;
                double dIn = i > 0 ? fk_.getDhD(i - 1) : fk_.getDhD(0);
                if (std::abs(aIn) > 1e-6) {
                    extents_[i] = {std::min(0.0, -aIn), std::max(0.0, -aIn)};
                } else {
                    double length = std::abs(dIn) > 1e-6 ? std::abs(dIn) : kHousingLength;
                    extents_[i] = {-length / 2.0, length / 2.0};
                }
                radii_[i] = fk_.getLinkRadius(kRadiusIndex[i]);
            }
        }

        std::array<Capsule, kNumCapsuleLinks> capsules(
                const JointConfig& q, const SE3& base = SE3::Identity()) const {
            std::array<Capsule, kNumCapsuleLinks> out;
            for (int i = 0; i < kNumCapsuleLinks; i++) {
                SE3 T = base * fk_.getCylinderTransform(i, q);
                R3 p = T.block<3,1>(0, 3), u = T.block<3,1>(0, 2);
                out[i] = {p + extents_[i].first * u, p + extents_[i].second * u,
                          radii_[i]};
            }
            return out;
        }

        double linkRadius(int i) const { return radii_[i]; }
        std::pair<double, double> extents(int i) const { return extents_[i]; }
        const ForwardKinematics<6>& fk() const { return fk_; }

    private:
        ForwardKinematics<6> fk_;
        std::array<std::pair<double, double>, kNumCapsuleLinks> extents_;
        std::array<double, kNumCapsuleLinks> radii_;
    };

}
