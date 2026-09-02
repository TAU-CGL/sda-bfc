#pragma once

#include "fk.hpp"

namespace sda_bfc {
    class UR5e : public ForwardKinematics<6> {
    public:
        UR5e() : ForwardKinematics<6>(
            DHParameter(0.1625, 0.0, 0.0, 0.1333, 0.0997, 0.0996),
            DHParameter(0.0, -0.425, -0.3922, 0.0, 0.0, 0.0),
            DHParameter(M_PI_2, 0.0, 0.0, M_PI_2, -M_PI_2, 0.0),
            LinkRadii(0.0755, 0.0601, 0.0601, 0.0578, 0.235 / (2.0 * M_PI), 0.0393, 0.0376),
            // Measured on the real UR5e: the upper arm tube sits 138 mm off
            // its DH line, the forearm 7 mm.
            LinkZOffsets(0.0, 0.0, 0.138, 0.007, 0.0, 0.0),
            // Axis index -> linkRadii index: i for the base links, i + 1 from
            // the forearm on.
            RadiusIndexMap{0, 1, 2, 4, 5, 6}
        ){}
    };
}