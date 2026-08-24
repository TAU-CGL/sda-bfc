#pragma once

#include "solver_newton.hpp"

namespace sda_bfc {

    // First-order alternative to Gauss-Newton: Adam on the SE(3) manifold.
    // Moments live in the 6-dim exponential coordinates of the current
    // iterate (no parallel transport, as is standard practice); the update
    // is multiplicative, X <- X exp(delta^), so iterates stay on SE(3).
    // Inherits solveMultistart from SolverNewton, which dispatches to this
    // solve() through the virtual call.
    class SolverAdam : public SolverNewton {
    public:
        SolverAdam(const std::vector<SE3>& As, const std::vector<SE3>& Bs,
                   double radiusA, double radiusB,
                   double learningRate = 0.01, double beta1 = 0.9,
                   double beta2 = 0.999, double epsilon = 1e-8) :
            SolverNewton(As, Bs, radiusA, radiusB),
            learningRate(learningRate), beta1(beta1), beta2(beta2),
            epsilon(epsilon) {}

        SE3 solve(SE3 X, int maxIterations = 2000,
                  double tolerance = 1e-12) const override {
            Vector6d m = Vector6d::Zero(), v = Vector6d::Zero();
            for (int iter = 1; iter <= maxIterations; iter++) {
                Vector6d grad = Vector6d::Zero();
                for (size_t t = 0; t < Bs.size(); t++) {
                    Vector6d J;
                    double f = residualWithJacobian(X, t, J);
                    grad += 2.0 * f * J;
                }
                if (grad.norm() < tolerance) break;
                m = beta1 * m + (1.0 - beta1) * grad;
                v = beta2 * v + (1.0 - beta2) * grad.cwiseProduct(grad);
                Vector6d mHat = m / (1.0 - std::pow(beta1, iter));
                Vector6d vHat = v / (1.0 - std::pow(beta2, iter));
                Vector6d delta = -learningRate *
                    mHat.cwiseQuotient((vHat.cwiseSqrt().array() + epsilon).matrix());
                X = X * se3Exp(delta);
                if (delta.norm() < tolerance) break;
            }
            return X;
        }

    private:
        double learningRate, beta1, beta2, epsilon;
    };

}
