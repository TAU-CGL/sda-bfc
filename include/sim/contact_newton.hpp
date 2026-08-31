#pragma once

#include <optional>
#include <random>

#include "scene.hpp"

namespace sda_bfc {

    struct ContactParams {
        int maxIterations = 100;
        double tolerance = 1e-14;
        double damping = 1e-9;
        int maxBacktracks = 20;
        double minDenom = 0.1;
        double jointRange = M_PI;
        int maxProximalRestarts = 2000;
        int maxDistalAttempts = 100;
        double clearanceTol = 1e-6;
        double othersMargin = 5e-3;
        double interiorMargin = 0.35;
        int touchLink = kForearmLink;
    };

    struct ContactPose {
        JointConfig qA, qB;
    };

    // Solves the touch condition for the proximal joints theta = (qA[0:3],
    // qB[0:3]) with damped Gauss-Newton on the same implicit polynomial the
    // solvers use, then samples the distal joints uniformly until the full
    // configuration is collision-free with a single interior forearm contact.
    class ContactGenerator {
    public:
        using Theta = Eigen::Vector<double, 6>;

        ContactGenerator(const ForwardKinematics<6>& fk, const SE3& X,
                         ContactParams params = {}) :
            scene_(fk, X), X_(X), params_(params) {
            s_ = 2.0 * scene_.model().linkRadius(params_.touchLink);
        }

        double residual(const Theta& theta) const {
            return evaluate(theta, nullptr);
        }

        double residualWithJacobian(const Theta& theta, Theta& J) const {
            return evaluate(theta, &J);
        }

        // Workcell constraints: exact for the static arm (known base),
        // margined by the caller for the dynamic arm (uncertain base).
        void addStaticHalfspace(const Halfspace& h) { staticHalfspaces_.push_back(h); }
        void addDynamicHalfspace(const Halfspace& h) { dynamicHalfspaces_.push_back(h); }

        std::optional<ContactPose> generate(std::mt19937& gen) const {
            std::uniform_real_distribution<double> joint(-params_.jointRange,
                                                         params_.jointRange);
            const int link = params_.touchLink;
            for (int restart = 0; restart < params_.maxProximalRestarts; restart++) {
                Theta theta;
                for (int i = 0; i < 6; i++) theta[i] = joint(gen);
                if (!solveProximal(theta)) continue;
                JointConfig qA = padConfig(theta.head<3>());
                JointConfig qB = padConfig(theta.tail<3>());
                if (relativeDenom(qA, qB) < params_.minDenom) continue;
                Capsule capA = scene_.model().capsules(qA)[link];
                Capsule capB = scene_.model().capsules(qB, X_)[link];
                if (std::abs(capsuleClearance(capA, capB)) > params_.clearanceTol) {
                    continue;  // touch on the infinite line but off the segments
                }
                if (!scene_.contactInterior(qA, qB, link, link,
                                            params_.interiorMargin)) {
                    continue;
                }
                // Distal joints cannot move links 0..link, so any violation
                // among those pairs is unfixable: gate before the distal loop.
                if (!proximalClear(qA, qB)) continue;
                if (!admitted(qA, qB, params_.touchLink)) continue;
                for (int attempt = 0; attempt < params_.maxDistalAttempts; attempt++) {
                    for (int i = 3; i < 6; i++) {
                        qA[i] = joint(gen);
                        qB[i] = joint(gen);
                    }
                    if (admitted(qA, qB, kNumCapsuleLinks - 1)
                            && scene_.validContact(qA, qB, link, link,
                                                   params_.clearanceTol,
                                                   params_.othersMargin,
                                                   params_.interiorMargin)) {
                        return ContactPose{qA, qB};
                    }
                }
            }
            return std::nullopt;
        }

        const TwoArmScene& scene() const { return scene_; }

    private:
        // Workcell admission for links kFirstWorkcellLink..lastLink (the base
        // column stands on the floor by construction).
        bool admitted(const JointConfig& qA, const JointConfig& qB,
                      int lastLink) const {
            auto capsA = scene_.model().capsules(qA);
            auto capsB = scene_.model().capsules(qB, X_);
            for (int i = 2; i <= lastLink; i++) {
                for (const Halfspace& h : staticHalfspaces_) {
                    if (!h.admits(capsA[i])) return false;
                }
                for (const Halfspace& h : dynamicHalfspaces_) {
                    if (!h.admits(capsB[i])) return false;
                }
            }
            return true;
        }

        bool proximalClear(const JointConfig& qA, const JointConfig& qB) const {
            const int link = params_.touchLink;
            auto capsA = scene_.model().capsules(qA);
            auto capsB = scene_.model().capsules(qB, X_);
            for (int i = 0; i <= link; i++) {
                for (int j = 0; j <= link; j++) {
                    if (i == link && j == link) continue;
                    if (capsuleClearance(capsA[i], capsB[j]) <= params_.othersMargin) {
                        return false;
                    }
                }
            }
            for (int i = 0; i <= link; i++) {
                for (int j = i + 2; j <= link; j++) {
                    if (capsuleClearance(capsA[i], capsA[j]) <= 0.0) return false;
                    if (capsuleClearance(capsB[i], capsB[j]) <= 0.0) return false;
                }
            }
            return true;
        }

        static JointConfig padConfig(const Eigen::Vector3d& proximal) {
            JointConfig q = JointConfig::Zero();
            q.head<3>() = proximal;
            return q;
        }

        double relativeDenom(const JointConfig& qA, const JointConfig& qB) const {
            const auto& fk = scene_.model().fk();
            SE3 T = fk.getCylinderTransform(params_.touchLink, qA).inverse()
                * X_ * fk.getCylinderTransform(params_.touchLink, qB);
            R3 u = T.block<3,1>(0, 2);
            return u[0] * u[0] + u[1] * u[1];
        }

        // f = c^2 - (ux^2 + uy^2) s^2 with (p, u) from T_rel = A^{-1} X B;
        // J via joint twists: for a frame G rigidly downstream of joint k,
        // dG/dq_k rotates about the axis (o_k, w_k) of that joint.
        double evaluate(const Theta& theta, Theta* J) const {
            const auto& fk = scene_.model().fk();
            const int link = params_.touchLink;
            JointConfig qA = padConfig(theta.head<3>());
            JointConfig qB = padConfig(theta.tail<3>());
            SE3 A = fk.getCylinderTransform(link, qA);
            SE3 B = fk.getCylinderTransform(link, qB);
            Eigen::Matrix3d RA = A.block<3,3>(0,0);
            R3 pA = A.block<3,1>(0,3);
            SE3 Y = X_ * B;
            R3 pY = Y.block<3,1>(0,3), uY = Y.block<3,1>(0,2);
            R3 p = RA.transpose() * (pY - pA);
            R3 u = RA.transpose() * uY;
            double c = p[0] * u[1] - p[1] * u[0];
            double f = c * c - (u[0] * u[0] + u[1] * u[1]) * s_ * s_;
            if (J) {
                R3 gp = 2.0 * c * R3(u[1], -u[0], 0.0);
                R3 gu = 2.0 * c * R3(-p[1], p[0], 0.0)
                    - 2.0 * s_ * s_ * R3(u[0], u[1], 0.0);
                Eigen::Matrix3d RM = RA.transpose() * X_.block<3,3>(0,0);
                R3 pB = B.block<3,1>(0,3), uB = B.block<3,1>(0,2);
                for (int k = 0; k < 3; k++) {
                    SE3 FA = fk.getJointFrame(k, qA);
                    R3 wA = FA.block<3,1>(0,2), oA = FA.block<3,1>(0,3);
                    R3 dp = -(RA.transpose() * wA.cross(pY - oA));
                    R3 du = -(RA.transpose() * wA.cross(uY));
                    (*J)[k] = gp.dot(dp) + gu.dot(du);
                }
                for (int k = 0; k < 3; k++) {
                    SE3 FB = fk.getJointFrame(k, qB);
                    R3 wB = FB.block<3,1>(0,2), oB = FB.block<3,1>(0,3);
                    R3 dp = RM * wB.cross(pB - oB);
                    R3 du = RM * wB.cross(uB);
                    (*J)[3 + k] = gp.dot(dp) + gu.dot(du);
                }
            }
            return f;
        }

        bool solveProximal(Theta& theta) const {
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

        TwoArmScene scene_;
        SE3 X_;
        double s_;
        ContactParams params_;
        std::vector<Halfspace> staticHalfspaces_, dynamicHalfspaces_;
    };

}
