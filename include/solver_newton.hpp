#pragma once

#include <random>

#include "solver.hpp"

namespace sda_bfc {

    using Vector6d = Eigen::Vector<double, 6>;
    using Matrix6d = Eigen::Matrix<double, 6, 6>;

    inline Eigen::Matrix3d skew(const R3& w) {
        Eigen::Matrix3d S;
        S <<
            0.0, -w[2], w[1],
            w[2], 0.0, -w[0],
            -w[1], w[0], 0.0;
        return S;
    }

    inline SE3 se3Exp(const Vector6d& delta) {
        R3 v = delta.head<3>(), w = delta.tail<3>();
        double theta = w.norm();
        Eigen::Matrix3d W = skew(w), W2 = W * W;
        double a, b, c;
        if (theta < 1e-9) {
            a = 1.0; b = 0.5; c = 1.0 / 6.0;
        } else {
            a = std::sin(theta) / theta;
            b = (1.0 - std::cos(theta)) / (theta * theta);
            c = (theta - std::sin(theta)) / (theta * theta * theta);
        }
        SE3 T = SE3::Identity();
        T.block<3,3>(0,0) = Eigen::Matrix3d::Identity() + a * W + b * W2;
        T.block<3,1>(0,3) = (Eigen::Matrix3d::Identity() + b * W + c * W2) * v;
        return T;
    }

    class SolverNewton : public Solver {
    public:
        using Solver::Solver;

        // J = d f_t / d delta for the multiplicative update X exp(delta^).
        double residualWithJacobian(const SE3& X, size_t t, Vector6d& J) const {
            SE3 M = Ainvs[t] * X;
            SE3 T = M * Bs[t];
            R3 p = T.block<3,1>(0,3), u = T.block<3,1>(0,2);
            double c = p[0] * u[1] - p[1] * u[0];
            double f = c * c - (u[0] * u[0] + u[1] * u[1]) * s * s;
            R3 gp = 2.0 * c * R3(u[1], -u[0], 0.0);
            R3 gu = 2.0 * c * R3(-p[1], p[0], 0.0) - 2.0 * s * s * R3(u[0], u[1], 0.0);
            Eigen::Matrix3d RmT = M.block<3,3>(0,0).transpose();
            R3 gpLocal = RmT * gp, guLocal = RmT * gu;
            R3 tb = Bs[t].block<3,1>(0,3), bz = Bs[t].block<3,1>(0,2);
            J.head<3>() = gpLocal;
            J.tail<3>() = tb.cross(gpLocal) + bz.cross(guLocal);
            return f;
        }

        SE3 solve(SE3 X, int maxIterations = 100, double tolerance = 1e-14) const override {
            for (int iter = 0; iter < maxIterations; iter++) {
                Matrix6d H = Matrix6d::Zero();
                Vector6d g = Vector6d::Zero();
                for (size_t t = 0; t < Bs.size(); t++) {
                    Vector6d J;
                    double f = residualWithJacobian(X, t, J);
                    H += J * J.transpose();
                    g += J * f;
                }
                H += 1e-12 * H.trace() * Matrix6d::Identity();
                Vector6d delta = -H.ldlt().solve(g);
                X = X * se3Exp(delta);
                if (delta.norm() < tolerance) break;
            }
            return X;
        }

        // The cost landscape is quartic with many spurious local minima; deterministic
        // seeded restarts + best-cost selection find the global basin reliably.
        SE3 solveMultistart(int numStarts = 2000, double translationRange = 1.2,
                            int maxIterations = 150, unsigned seed = 0) const {
            std::mt19937 gen(seed);
            std::uniform_real_distribution<double> translation(-translationRange, translationRange);
            std::uniform_real_distribution<double> angle(0.0, M_PI);
            std::normal_distribution<double> gauss;
            SE3 best = solve(SE3::Identity(), maxIterations);
            double bestCost = cost(best);
            for (int i = 0; i < numStarts; i++) {
                R3 w(gauss(gen), gauss(gen), gauss(gen));
                w.normalize();
                w *= angle(gen);
                Vector6d d;
                d << translation(gen), translation(gen), translation(gen), w[0], w[1], w[2];
                SE3 X = solve(se3Exp(d), maxIterations);
                double c = cost(X);
                if (c < bestCost) {
                    bestCost = c;
                    best = X;
                }
            }
            return best;
        }
    };

}
