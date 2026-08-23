#pragma once

#include <vector>

#include "geometry.hpp"

namespace sda_bfc {

    class Solver {
    public:
        Solver(const std::vector<SE3>& As, const std::vector<SE3>& Bs,
               double radiusA, double radiusB) : Bs(Bs), s(radiusA + radiusB) {
            Ainvs.reserve(As.size());
            for (const SE3& A : As) Ainvs.push_back(A.inverse());
        }

        virtual ~Solver() = default;

        virtual SE3 solve(SE3 x0, int maxIterations = 100, double tolerance = 1e-14) const = 0;

        // f_t(X) = c^2 - (ux^2 + uy^2) s^2 with (p, u) taken from T = A_t^{-1} X B_t,
        // c = px uy - py ux (the implicit touch condition).
        double residual(const SE3& X, size_t t) const {
            SE3 T = Ainvs[t] * X * Bs[t];
            R3 p = T.block<3,1>(0,3), u = T.block<3,1>(0,2);
            double c = p[0] * u[1] - p[1] * u[0];
            return c * c - (u[0] * u[0] + u[1] * u[1]) * s * s;
        }

        double cost(const SE3& X) const {
            double total = 0.0;
            for (size_t t = 0; t < Bs.size(); t++) {
                double f = residual(X, t);
                total += f * f;
            }
            return total;
        }

    protected:
        std::vector<SE3> Ainvs, Bs;
        double s;
    };

}
