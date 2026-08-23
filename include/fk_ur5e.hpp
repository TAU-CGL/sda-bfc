#pragma once

#include "fk.hpp"

namespace sda_bfc {
    class UR5e : public ForwardKinematics<6> {
    public:
        UR5e() : ForwardKinematics<6>(
            DHParameter(0.1625, 0.0, 0.0, 0.1333, 0.0997, 0.0996),
            DHParameter(0.0, -0.425, -0.3922, 0.0, 0.0, 0.0),
            DHParameter(M_PI_2, 0.0, 0.0, M_PI_2, -M_PI_2, 0.0),
            LinkRadii(0.0755, 0.0601, 0.0601, 0.0578, 0.235 / (2.0 * M_PI), 0.0393, 0.0376)
        ){}
    };
}