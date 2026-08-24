#pragma once

#include <random>
#include <stdexcept>

#include "scene.hpp"

namespace sda_bfc {

    struct PlacementParams {
        double minDistance = 0.5, maxDistance = 0.78;
        double maxTiltDeg = 2.0;
        double maxAbsZ = 0.05;
        int maxAttempts = 100;
    };

    // Random base offset of R2 relative to R1: planar distance within reach,
    // free bearing and yaw, small roll/pitch and height offset.
    inline SE3 samplePlacement(std::mt19937& gen, const PlacementParams& params = {}) {
        std::uniform_real_distribution<double> distanceDist(params.minDistance,
                                                            params.maxDistance);
        std::uniform_real_distribution<double> angleDist(0.0, 2.0 * M_PI);
        std::uniform_real_distribution<double> tiltDist(
            -params.maxTiltDeg * M_PI / 180.0, params.maxTiltDeg * M_PI / 180.0);
        std::uniform_real_distribution<double> zDist(-params.maxAbsZ, params.maxAbsZ);
        double distance = distanceDist(gen);
        double bearing = angleDist(gen);
        double yaw = angleDist(gen);
        double roll = tiltDist(gen);
        double pitch = tiltDist(gen);
        double z = zDist(gen);
        Eigen::Matrix3d Rz, Ry, Rx;
        Rz << std::cos(yaw), -std::sin(yaw), 0, std::sin(yaw), std::cos(yaw), 0, 0, 0, 1;
        Ry << std::cos(pitch), 0, std::sin(pitch), 0, 1, 0, -std::sin(pitch), 0, std::cos(pitch);
        Rx << 1, 0, 0, 0, std::cos(roll), -std::sin(roll), 0, std::sin(roll), std::cos(roll);
        SE3 X = SE3::Identity();
        X.block<3,3>(0,0) = Rz * Ry * Rx;
        X.block<3,1>(0,3) = R3(distance * std::cos(bearing),
                               distance * std::sin(bearing), z);
        return X;
    }

    inline SE3 sampleValidPlacement(std::mt19937& gen,
                                    const ForwardKinematics<6>& fk,
                                    const PlacementParams& params = {}) {
        for (int attempt = 0; attempt < params.maxAttempts; attempt++) {
            SE3 X = samplePlacement(gen, params);
            if (TwoArmScene(fk, X).collisionFreeAtHome()) return X;
        }
        throw std::runtime_error("sampleValidPlacement: no collision-free placement found");
    }

}
