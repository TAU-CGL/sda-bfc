#pragma once

#include "capsule.hpp"

namespace sda_bfc {

    using PointCloud = Eigen::Matrix<double, Eigen::Dynamic, 3>;

    inline R3 supportOfCloud(const PointCloud& points, const R3& direction) {
        Eigen::Index best;
        (points * direction).maxCoeff(&best);
        return points.row(best);
    }

    inline R3 supportOfSegment(const R3& a, const R3& b, const R3& direction) {
        return a.dot(direction) >= b.dot(direction) ? a : b;
    }

    namespace detail {

        inline bool updateSimplex(R3* s, int& n, R3& d) {
            R3 A = s[0], AO = -A;
            if (n == 2) {
                R3 B = s[1], AB = B - A;
                if (AB.dot(AO) > 0.0) {
                    d = AB.cross(AO).cross(AB);
                } else {
                    n = 1;
                    d = AO;
                }
                return false;
            }
            if (n == 3) {
                R3 B = s[1], C = s[2];
                R3 AB = B - A, AC = C - A, ABC = AB.cross(AC);
                if (ABC.cross(AC).dot(AO) > 0.0) {
                    if (AC.dot(AO) > 0.0) {
                        s[1] = C; n = 2;
                        d = AC.cross(AO).cross(AC);
                    } else if (AB.dot(AO) > 0.0) {
                        n = 2;
                        d = AB.cross(AO).cross(AB);
                    } else {
                        n = 1; d = AO;
                    }
                } else if (AB.cross(ABC).dot(AO) > 0.0) {
                    if (AB.dot(AO) > 0.0) {
                        n = 2;
                        d = AB.cross(AO).cross(AB);
                    } else {
                        n = 1; d = AO;
                    }
                } else if (ABC.dot(AO) > 0.0) {
                    d = ABC;
                } else {
                    s[1] = C; s[2] = B;
                    d = -ABC;
                }
                return false;
            }
            // Tetrahedron: test the three faces containing the newest vertex.
            R3 B = s[1], C = s[2], D = s[3];
            R3 AB = B - A, AC = C - A, AD = D - A;
            R3 ABC = AB.cross(AC), ACD = AC.cross(AD), ADB = AD.cross(AB);
            if (ABC.dot(AO) > 0.0) {
                s[3] = s[2]; n = 3;
                return updateSimplex(s, n, d);
            }
            if (ACD.dot(AO) > 0.0) {
                s[1] = C; s[2] = D; n = 3;
                return updateSimplex(s, n, d);
            }
            if (ADB.dot(AO) > 0.0) {
                s[1] = D; s[2] = B; n = 3;
                return updateSimplex(s, n, d);
            }
            return true;
        }

    }

    // True iff distance(conv(points), segment [a, b]) <= r: boolean GJK on
    // the Minkowski difference (conv(points) + ball_r) - segment.
    inline bool cloudIntersectsCapsule(const PointCloud& points,
                                       const R3& a, const R3& b, double r) {
        auto support = [&](const R3& d) {
            return R3(supportOfCloud(points, d) + r * d.normalized()
                      - supportOfSegment(a, b, -d));
        };
        R3 simplex[4];
        simplex[0] = support(R3::UnitX());
        int n = 1;
        R3 d = -simplex[0];
        for (int iter = 0; iter < 64; iter++) {
            if (d.squaredNorm() < 1e-18) return true;
            R3 point = support(d);
            if (point.dot(d) < 1e-12) return false;
            for (int i = n; i > 0; i--) simplex[i] = simplex[i - 1];
            simplex[0] = point;
            n++;
            if (detail::updateSimplex(simplex, n, d)) return true;
        }
        return true;  // no convergence: report collision (conservative)
    }

}
