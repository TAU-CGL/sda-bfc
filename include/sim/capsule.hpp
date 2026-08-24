#pragma once

#include <algorithm>

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

}
