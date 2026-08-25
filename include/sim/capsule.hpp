#pragma once

#include <algorithm>
#include <array>
#include <vector>

#include "../geometry.hpp"

namespace sda_bfc {

    struct Capsule {
        R3 a, b;
        double r;
    };

    struct SegmentClosest {
        double distance;
        double s, t;
    };

    inline SegmentClosest segmentClosest(const R3& p1, const R3& q1,
                                         const R3& p2, const R3& q2) {
        R3 d1 = q1 - p1, d2 = q2 - p2, r = p1 - p2;
        double a = d1.dot(d1), e = d2.dot(d2);
        double b = d1.dot(d2), c = d1.dot(r), f = d2.dot(r);
        double s, t;
        if (a <= 1e-14 && e <= 1e-14) {
            s = t = 0.0;
        } else if (a <= 1e-14) {
            s = 0.0;
            t = std::clamp(f / e, 0.0, 1.0);
        } else if (e <= 1e-14) {
            t = 0.0;
            s = std::clamp(-c / a, 0.0, 1.0);
        } else {
            double denom = a * e - b * b;
            s = denom > 1e-14
                ? std::clamp((b * f - c * e) / denom, 0.0, 1.0) : 0.0;
            t = (b * s + f) / e;
            if (t < 0.0) {
                t = 0.0;
                s = std::clamp(-c / a, 0.0, 1.0);
            } else if (t > 1.0) {
                t = 1.0;
                s = std::clamp((b - c) / a, 0.0, 1.0);
            }
        }
        double distance = ((p1 + s * d1) - (p2 + t * d2)).norm();
        return {distance, s, t};
    }

    inline double segmentDistance(const R3& p1, const R3& q1,
                                  const R3& p2, const R3& q2) {
        return segmentClosest(p1, q1, p2, q2).distance;
    }

    inline double capsuleClearance(const Capsule& c1, const Capsule& c2) {
        return segmentDistance(c1.a, c1.b, c2.a, c2.b) - (c1.r + c2.r);
    }

    // A capsule is admitted while it stays on the free side:
    // normal . p + r <= offset at both endpoints.
    struct Halfspace {
        R3 normal;
        double offset;

        bool admits(const Capsule& c) const {
            return normal.dot(c.a) + c.r <= offset
                && normal.dot(c.b) + c.r <= offset;
        }
    };

    // ------------------------------------------------------------------
    // Uncertainty expansion: the placement of the other arm is uncertain
    // within a +-range box in (x, y, z, roll, pitch, yaw); the static arm's
    // capsules are expanded so that planning against them with the nominal
    // placement is safe for every placement in the range.  Rotations pivot
    // about the static arm's base (the frame origin of the capsules).
    // ------------------------------------------------------------------

    struct UncertaintyRanges {
        double x = 0.0, y = 0.0, z = 0.0;       // meters
        double roll = 0.0, pitch = 0.0, yaw = 0.0;  // radians
    };

    inline std::array<R3, 8> capsuleObbVertices(const Capsule& c,
                                                double extraRadius = 0.0) {
        R3 axis = c.b - c.a;
        double length = axis.norm();
        R3 u = length > 1e-12 ? R3(axis / length) : R3(R3::UnitZ());
        R3 ex = u.cross(R3::UnitZ());
        if (ex.norm() < 1e-9) ex = u.cross(R3::UnitX());
        ex.normalize();
        R3 ey = u.cross(ex).normalized();
        double r = c.r + extraRadius;
        std::array<R3, 8> out;
        int k = 0;
        for (double h : {-r, length + r}) {
            for (double sx : {-1.0, 1.0}) {
                for (double sy : {-1.0, 1.0}) {
                    out[k++] = c.a + h * u + sx * r * ex + sy * r * ey;
                }
            }
        }
        return out;
    }

    // Inverse transforms dT^-1 sampling the range box, with
    // dT = trans(t) . Rz(yaw) . Ry(pitch) . Rx(roll).  Translation axes at
    // {-max, +max} (exact: translation of a convex set is linear); rotation
    // axes at {-max, 0, +max} (the hull of extreme rotations alone is the
    // CHORD of the swept arc and cuts inside it -- the midpoint sample plus
    // the sagitta padding below restore an outer bound).
    inline std::vector<std::pair<Eigen::Matrix3d, R3>> rangeSampleTransforms(
            const UncertaintyRanges& ranges) {
        std::vector<std::pair<Eigen::Matrix3d, R3>> out;
        for (double tx : {-1.0, 1.0}) {
        for (double ty : {-1.0, 1.0}) {
        for (double tz : {-1.0, 1.0}) {
            for (double sr : {-1.0, 0.0, 1.0}) {
            for (double sp : {-1.0, 0.0, 1.0}) {
            for (double sy : {-1.0, 0.0, 1.0}) {
                double roll = sr * ranges.roll, pitch = sp * ranges.pitch,
                       yaw = sy * ranges.yaw;
                Eigen::Matrix3d Rz, Ry, Rx;
                Rz << std::cos(yaw), -std::sin(yaw), 0,
                      std::sin(yaw), std::cos(yaw), 0, 0, 0, 1;
                Ry << std::cos(pitch), 0, std::sin(pitch), 0, 1, 0,
                      -std::sin(pitch), 0, std::cos(pitch);
                Rx << 1, 0, 0, 0, std::cos(roll), -std::sin(roll),
                      0, std::sin(roll), std::cos(roll);
                Eigen::Matrix3d R = Rz * Ry * Rx;
                R3 t(tx * ranges.x, ty * ranges.y, tz * ranges.z);
                out.emplace_back(R.transpose(), -(R.transpose() * t));
            }
            }
            }
        }
        }
        }
        return out;
    }

    // Outer bound on the arc-vs-hull deviation between rotation samples: the
    // worst point sits rMax from the pivot and at most (roll+pitch+yaw)/2
    // away from the nearest sampled rotation.
    inline double sagittaPadding(const Capsule& c, const UncertaintyRanges& ranges) {
        double rMax = 0.0;
        for (const R3& v : capsuleObbVertices(c)) {
            rMax = std::max(rMax, v.norm());
        }
        double thetaGap = 0.5 * (std::abs(ranges.roll) + std::abs(ranges.pitch)
                                 + std::abs(ranges.yaw));
        return rMax * (1.0 - std::cos(thetaGap));
    }

    // Point cloud whose convex hull outer-bounds the capsule under every
    // placement in the uncertainty range (applied as dT^-1).
    inline Eigen::Matrix<double, Eigen::Dynamic, 3> expandCapsule(
            const Capsule& c, const UncertaintyRanges& ranges) {
        auto obb = capsuleObbVertices(c, sagittaPadding(c, ranges));
        auto transforms = rangeSampleTransforms(ranges);
        Eigen::Matrix<double, Eigen::Dynamic, 3> points(
            8 * (long)transforms.size(), 3);
        long row = 0;
        for (const auto& [Rinv, tinv] : transforms) {
            for (const R3& v : obb) {
                points.row(row++) = (Rinv * v + tinv).transpose();
            }
        }
        return points;
    }

}
