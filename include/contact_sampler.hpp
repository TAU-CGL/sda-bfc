#pragma once

#include <algorithm>
#include <optional>
#include <random>
#include <stdexcept>

#include "fk/fk.hpp"

namespace sda_bfc {

    using JointConfig = Eigen::Vector<double, 6>;

    struct ContactParams {
        int maxIterations = 100;
        double tolerance = 1e-14;
        double damping = 1e-9;
        int maxBacktracks = 20;
        double minDenom = 0.1;
        double jointRange = M_PI;
        int maxRestarts = 2000;
        double clearanceTol = 1e-6;
        double interiorMargin = 0.02;
    };

    struct ContactPose {
        JointConfig qA, qB;
    };

    struct SegmentClosest {
        double distance, s, t;  // s, t normalized to [0, 1] on each segment
    };

    inline SegmentClosest segmentClosest(const R3& p1, const R3& q1,
                                         const R3& p2, const R3& q2) {
        R3 d1 = q1 - p1, d2 = q2 - p2, r = p1 - p2;
        double a = d1.squaredNorm(), e = d2.squaredNorm(), f = d2.dot(r);
        double c = d1.dot(r), b = d1.dot(d2);
        double denom = a * e - b * b;
        double s = (denom > 1e-12) ? std::clamp((b * f - c * e) / denom, 0.0, 1.0) : 0.0;
        double t = (e > 1e-12) ? std::clamp((b * s + f) / e, 0.0, 1.0) : 0.0;
        s = (a > 1e-12) ? std::clamp((b * t - c) / a, 0.0, 1.0) : 0.0;
        return {(p1 + s * d1 - p2 - t * d2).norm(), s, t};
    }

    // Samples touching configurations q = (qA, qB) for two arms of the same
    // model: arm A based at the identity, arm B based at X.  The touching
    // cylinders are chosen by axis index (getCylinderTransform indexing, with
    // the physical zOffset and radius taken from the FK model).  Damped Gauss-Newton
    // drives the implicit touch polynomial of the two cylinder axes to zero
    // over the joints each cylinder depends on (0..idx-1); the remaining
    // distal joints are sampled uniformly since they cannot move the contact.
    // Only the touching cylinders are considered here: full-geometry validity
    // (self collisions, other cross-arm pairs) is checked Python-side by
    // sda_bfc.collision against the MJCF collision model.
    class ContactSampler {
    public:
        using Theta = Eigen::VectorXd;

        ContactSampler(const ForwardKinematics<6>& fk, const SE3& X,
                         int idxA, int idxB, ContactParams params = {}) :
            fk_(fk), X_(X), idxA_(idxA), idxB_(idxB), params_(params),
            aA_(fk.getDhA(idxA - 1)), aB_(fk.getDhA(idxB - 1)),
            s_(fk.getCylinderRadius(idxA) + fk.getCylinderRadius(idxB)) {
            if (std::abs(aA_) < 1e-9 || std::abs(aB_) < 1e-9) {
                throw std::invalid_argument(
                    "contact link has zero length (dhA == 0)");
            }
        }

        double residual(const Theta& theta) const {
            return evaluate(theta, nullptr);
        }

        double residualWithJacobian(const Theta& theta, Theta& J) const {
            return evaluate(theta, &J);
        }

        std::optional<ContactPose> sample(std::mt19937& gen) const {
            std::uniform_real_distribution<double> joint(-params_.jointRange,
                                                         params_.jointRange);
            for (int restart = 0; restart < params_.maxRestarts; restart++) {
                Theta theta(idxA_ + idxB_);
                for (int i = 0; i < theta.size(); i++) theta[i] = joint(gen);
                if (!solveNewton(theta)) continue;
                JointConfig qA = JointConfig::Zero(), qB = JointConfig::Zero();
                qA.head(idxA_) = theta.head(idxA_);
                qB.head(idxB_) = theta.tail(idxB_);
                if (!touchOnSegments(qA, qB)) continue;
                for (int i = idxA_; i < 6; i++) qA[i] = joint(gen);
                for (int i = idxB_; i < 6; i++) qB[i] = joint(gen);
                return ContactPose{qA, qB};
            }
            return std::nullopt;
        }

    private:
        // The polynomial vanishes wherever the infinite axes are at distance
        // rA + rB; accept only solutions whose touch point lies on the finite
        // cylinders, at least interiorMargin away from either end, and whose
        // axes are far enough from parallel for the contact to be stable.
        bool touchOnSegments(const JointConfig& qA, const JointConfig& qB) const {
            SE3 A = fk_.getCylinderTransform(idxA_, qA, fk_.getLinkZOffset(idxA_));
            SE3 B = X_ * fk_.getCylinderTransform(idxB_, qB, fk_.getLinkZOffset(idxB_));
            SE3 T = A.inverse() * B;
            R3 u = T.block<3,1>(0,2);
            if (u[0] * u[0] + u[1] * u[1] < params_.minDenom) return false;
            R3 pA = A.block<3,1>(0,3), uA = A.block<3,1>(0,2);
            R3 pB = B.block<3,1>(0,3), uB = B.block<3,1>(0,2);
            SegmentClosest cp = segmentClosest(pA, pA - aA_ * uA,
                                               pB, pB - aB_ * uB);
            if (std::abs(cp.distance - s_) > params_.clearanceTol) return false;
            double lenA = std::abs(aA_), lenB = std::abs(aB_);
            return std::min(cp.s, 1.0 - cp.s) * lenA >= params_.interiorMargin
                && std::min(cp.t, 1.0 - cp.t) * lenB >= params_.interiorMargin;
        }

        // f = c^2 - (ux^2 + uy^2) s^2 with (p, u) from T_rel = A^{-1} X B;
        // J via joint twists: for a frame G rigidly downstream of joint k,
        // dG/dq_k rotates about the axis (o_k, w_k) of that joint.
        double evaluate(const Theta& theta, Theta* J) const {
            JointConfig qA = JointConfig::Zero(), qB = JointConfig::Zero();
            qA.head(idxA_) = theta.head(idxA_);
            qB.head(idxB_) = theta.tail(idxB_);
            SE3 A = fk_.getCylinderTransform(idxA_, qA, fk_.getLinkZOffset(idxA_));
            SE3 B = fk_.getCylinderTransform(idxB_, qB, fk_.getLinkZOffset(idxB_));
            Eigen::Matrix3d RA = A.block<3,3>(0,0);
            R3 pA = A.block<3,1>(0,3);
            SE3 Y = X_ * B;
            R3 pY = Y.block<3,1>(0,3), uY = Y.block<3,1>(0,2);
            R3 p = RA.transpose() * (pY - pA);
            R3 u = RA.transpose() * uY;
            double c = p[0] * u[1] - p[1] * u[0];
            double f = c * c - (u[0] * u[0] + u[1] * u[1]) * s_ * s_;
            if (J) {
                J->resize(idxA_ + idxB_);
                R3 gp = 2.0 * c * R3(u[1], -u[0], 0.0);
                R3 gu = 2.0 * c * R3(-p[1], p[0], 0.0)
                    - 2.0 * s_ * s_ * R3(u[0], u[1], 0.0);
                Eigen::Matrix3d RM = RA.transpose() * X_.block<3,3>(0,0);
                R3 pB = B.block<3,1>(0,3), uB = B.block<3,1>(0,2);
                for (int k = 0; k < idxA_; k++) {
                    SE3 FA = fk_.getJointFrame(k, qA);
                    R3 wA = FA.block<3,1>(0,2), oA = FA.block<3,1>(0,3);
                    R3 dp = -(RA.transpose() * wA.cross(pY - oA));
                    R3 du = -(RA.transpose() * wA.cross(uY));
                    (*J)[k] = gp.dot(dp) + gu.dot(du);
                }
                for (int k = 0; k < idxB_; k++) {
                    SE3 FB = fk_.getJointFrame(k, qB);
                    R3 wB = FB.block<3,1>(0,2), oB = FB.block<3,1>(0,3);
                    R3 dp = RM * wB.cross(pB - oB);
                    R3 du = RM * wB.cross(uB);
                    (*J)[idxA_ + k] = gp.dot(dp) + gu.dot(du);
                }
            }
            return f;
        }

        bool solveNewton(Theta& theta) const {
            for (int iter = 0; iter < params_.maxIterations; iter++) {
                Theta J;
                double f = evaluate(theta, &J);
                if (std::abs(f) < params_.tolerance) return true;
                double jtj = J.squaredNorm();
                double mu = params_.damping * std::max(jtj, 1e-12);
                Theta delta = -f * J / (jtj + mu);
                double scale = 1.0;
                bool improved = false;
                for (int bt = 0; bt < params_.maxBacktracks; bt++) {
                    Theta candidate = theta + scale * delta;
                    if (std::abs(evaluate(candidate, nullptr)) < std::abs(f)) {
                        theta = candidate;
                        improved = true;
                        break;
                    }
                    scale *= 0.5;
                }
                if (!improved) return false;
            }
            Theta J;
            return std::abs(evaluate(theta, &J)) < params_.tolerance;
        }

        ForwardKinematics<6> fk_;
        SE3 X_;
        int idxA_, idxB_;
        ContactParams params_;
        double aA_, aB_, s_;
    };

}
