#pragma once

#define _USE_MATH_DEFINES
#include <cmath>
#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

#include <Eigen/Dense>

namespace sda_bfc {

    using R3 = Eigen::Vector3d;
    using S2 = Eigen::Vector3d; // Name is different for readability purposes
    using SE3 = Eigen::Matrix4d;

    struct CylinderPose {
        R3 p; // position
        S2 u; // orientation (direction of the line axis)
        double r; // radius

        CylinderPose(R3 p, S2 u, double r) : p(p), u(u), r(r) {}

        static CylinderPose fromSE3(SE3 T, double r) {
            R3 p = T.block<3,1>(0, 3);
            S2 u = T.block<3,1>(0, 2);
            return CylinderPose(p, u, r);
        }

        SE3 toSE3() const {
            R3 a = u.cross(R3::UnitZ());
            if (a.norm() < 1e-9) {
                a = u.cross(R3::UnitX());
            }
            a.normalize();
            R3 b = u.cross(a).normalized();
            if (a.cross(b).dot(u) < 0.0) {
                R3 tmp = a;
                a = b;
                b = tmp;
            }
            SE3 T = SE3::Identity();
            T.block<3,1>(0,0) = a;
            T.block<3,1>(0,1) = b;
            T.block<3,1>(0,2) = u;
            T.block<3,1>(0,3) = p;
            return T;
        }

        static double signedDistance(const CylinderPose& q1, const CylinderPose& q2) {
            R3 dp = q2.p - q1.p;
            R3 n = q1.u.cross(q2.u);
            double nNorm = n.norm();
            double lineDistance;
            if (nNorm < 1e-9) {
                lineDistance = (dp - dp.dot(q1.u) * q1.u).norm();
            } else {
                lineDistance = std::abs(dp.dot(n)) / nNorm;
            }
            return lineDistance - (q1.r + q2.r);
        }

        double signedDistance(double otherR) const { // From an upright cylinder through origin whose radius is `otherR`
            double denom = std::sqrt(u[0] * u[0] + u[1] * u[1]);
            double dist;
            if (denom < 1e-9) {
                dist = std::sqrt(p[0] * p[0] + p[1] * p[1]);
            } else {
                dist = std::abs(p[0] * u[1] - p[1] * u[0]) / denom;
            }
            return dist - (r + otherR);
        }

        double implicitTouchCondition(double otherR) const { // From an upright cylinder through origin whose radius is `otherR`
            double denom = u[0] * u[0] + u[1] * u[1];
            double dist = p[0] * u[1] - p[1] * u[0];
            dist = dist * dist - denom * (r + otherR) * (r + otherR);
            return dist;
        }
    };

}