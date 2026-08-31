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
    // Lateral offset of each link's physical tube from its DH frame line,
    // applied along the frame z-axis (measured on the real UR5e; the upper
    // arm tube sits 138 mm off its DH line, the forearm 7 mm).
    inline constexpr std::array<double, kNumCapsuleLinks> kLinkZOffsets =
        {0.0, 0.0, 0.138, 0.007, 0.0, 0.0};

    // The last capsule covers the wrist_3 body AND the Robotiq 2F-85 bolted to it. The
    // gripper used to be modelled by nothing at all, so a sampled contact could put a
    // fingertip straight through the other arm; downstream that showed up as most
    // generated contacts being rejected on a gripper pair by the consuming project.
    //
    // AXIS, and this is why it is not just a longer extent. Every other capsule runs
    // along its link's TUBE, which Z_TO_X puts on the cylinder frame's z. The gripper
    // hangs off the flange along the TOOL axis, which is that same frame's -x --
    // perpendicular to the tube. Lengthening the tube extent grows the capsule sideways
    // and covers none of the gripper.
    //
    // EXTENT AND RADIUS, measured along the tool axis from the joint-5 cylinder frame:
    // the wrist_3 body spans 0.0406..0.1126 at r 0.0360, and the gripper 0.0759..0.2543
    // at r 0.0627 -- the latter fitted to all 26,337 vertices of the 2F-85's collision
    // meshes at the fingers-CLOSED envelope, the only state the cell runs it in. Closing
    // swings the fingertips forward, so closed is narrower but 14 mm longer than open;
    // the two numbers have to come from the same state or the tips sit outside.
    //
    // One capsule covers both, shaft from the wrist_3 body's centre to the gripper
    // shaft's end. The SHAFT is not the extent -- the caps reach r past each endpoint,
    // so a shaft spanning the whole body would make the capsule 2r too long. Verified:
    // every gripper vertex is within 0.06268 of this shaft, and the whole wrist_3
    // capsule within 0.0540.
    inline constexpr int kToolLink = kNumCapsuleLinks - 1;
    inline constexpr double kToolShaftNear = 0.0766;   // wrist_3 body centre
    inline constexpr double kToolShaftFar = 0.1916;    // gripper shaft end
    inline constexpr double kToolRadius = 0.0627;      // 2F-85 closed, at the knuckles

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
            extents_[kToolLink] = {kToolShaftNear, kToolShaftFar};
            radii_[kToolLink] = kToolRadius;
        }

        std::array<Capsule, kNumCapsuleLinks> capsules(
                const JointConfig& q, const SE3& base = SE3::Identity()) const {
            std::array<Capsule, kNumCapsuleLinks> out;
            for (int i = 0; i < kNumCapsuleLinks; i++) {
                SE3 T = base * fk_.getCylinderTransform(i, q, kLinkZOffsets[i]);
                R3 p = T.block<3,1>(0, 3);
                // the tool link runs along the tool axis, every other along its own tube
                R3 u = i == kToolLink ? R3(-T.block<3,1>(0, 0)) : R3(T.block<3,1>(0, 2));
                out[i] = {p + extents_[i].first * u, p + extents_[i].second * u,
                          radii_[i]};
            }
            return out;
        }

        double linkRadius(int i) const { return radii_[i]; }
        double linkZOffset(int i) const { return kLinkZOffsets[i]; }
        std::pair<double, double> extents(int i) const { return extents_[i]; }
        const ForwardKinematics<6>& fk() const { return fk_; }

    private:
        ForwardKinematics<6> fk_;
        std::array<std::pair<double, double>, kNumCapsuleLinks> extents_;
        std::array<double, kNumCapsuleLinks> radii_;
    };

}
