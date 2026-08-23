#pragma once

#include <algorithm>
#include <random>
#include <vector>

#include "solver.hpp"

namespace sda_bfc {

    // Minimize c^T x subject to A x <= b, x >= 0 (dense two-phase simplex,
    // Bland's rule). Returns false if infeasible.
    inline bool simplexLP(const Eigen::MatrixXd& A, const Eigen::VectorXd& b,
                          const Eigen::VectorXd& c, Eigen::VectorXd& x) {
        const int m = (int)A.rows(), n = (int)A.cols();
        int nArt = 0;
        std::vector<int> artOfRow(m, -1);
        for (int i = 0; i < m; i++) if (b[i] < 0.0) artOfRow[i] = nArt++;
        const int total = n + m + nArt;
        Eigen::MatrixXd T = Eigen::MatrixXd::Zero(m + 1, total + 1);
        std::vector<int> basis(m);
        for (int i = 0; i < m; i++) {
            double sign = b[i] < 0.0 ? -1.0 : 1.0;
            T.block(i, 0, 1, n) = sign * A.row(i);
            T(i, n + i) = sign;
            T(i, total) = sign * b[i];
            if (artOfRow[i] >= 0) {
                T(i, n + m + artOfRow[i]) = 1.0;
                basis[i] = n + m + artOfRow[i];
            } else {
                basis[i] = n + i;
            }
        }
        auto pivot = [&](int row, int col) {
            T.row(row) /= T(row, col);
            for (int i = 0; i <= m; i++)
                if (i != row && std::abs(T(i, col)) > 1e-14)
                    T.row(i) -= T(i, col) * T.row(row);
            basis[row] = col;
        };
        auto iterate = [&](int numCols) {
            while (true) {
                int col = -1;
                for (int j = 0; j < numCols; j++)
                    if (T(m, j) < -1e-11) { col = j; break; }
                if (col < 0) return;
                int row = -1;
                double bestRatio = 0.0;
                for (int i = 0; i < m; i++) {
                    if (T(i, col) > 1e-11) {
                        double ratio = T(i, total) / T(i, col);
                        if (row < 0 || ratio < bestRatio - 1e-14 ||
                            (ratio < bestRatio + 1e-14 && basis[i] < basis[row])) {
                            row = i;
                            bestRatio = ratio;
                        }
                    }
                }
                if (row < 0) return;
                pivot(row, col);
            }
        };
        if (nArt > 0) {
            for (int i = 0; i < m; i++)
                if (artOfRow[i] >= 0) T.row(m) -= T.row(i);
            iterate(n + m);
            if (T(m, total) < -1e-8) return false;
            for (int i = 0; i < m; i++) {
                if (basis[i] >= n + m) {
                    for (int j = 0; j < n + m; j++)
                        if (std::abs(T(i, j)) > 1e-11) { pivot(i, j); break; }
                }
            }
        }
        T.row(m).setZero();
        T.block(m, 0, 1, n) = c.transpose();
        for (int i = 0; i < m; i++)
            if (basis[i] < n) T.row(m) -= c[basis[i]] * T.row(i);
        iterate(n + m);
        x = Eigen::VectorXd::Zero(n);
        for (int i = 0; i < m; i++)
            if (basis[i] < n) x[basis[i]] = T(i, total);
        return true;
    }

    class SolverAnnealingLP : public Solver {
    public:
        SolverAnnealingLP(const std::vector<SE3>& As, const std::vector<SE3>& Bs,
                          double radiusA, double radiusB,
                          int nInitial = 5000, int nElites = 40, int nPerElite = 40,
                          double eps = 2e-3, unsigned seed = 0,
                          std::vector<double> sigmaSchedule = {0.3, 0.15, 0.075, 0.03, 0.01}) :
            Solver(As, Bs, radiusA, radiusB),
            nInitial(nInitial), nElites(nElites), nPerElite(nPerElite),
            eps(eps), seed(seed), sigmaSchedule(std::move(sigmaSchedule)) {}

        // dist_t(v) = c0 + c1 vx + c2 vy + c3 vz for orientation Rm; the
        // expressions are sympy-derived (CSE) and transplanted verbatim.
        void coefficients(const Eigen::Matrix3d& Rm, size_t t,
                          double& c0, double& c1, double& c2, double& c3) const {
            const SE3& Ai = Ainvs[t];
            const SE3& B = Bs[t];
            const double Atinv_1 = Ai(0,0), Atinv_2 = Ai(0,1), Atinv_3 = Ai(0,2), Atinv_4 = Ai(0,3);
            const double Atinv_5 = Ai(1,0), Atinv_6 = Ai(1,1), Atinv_7 = Ai(1,2), Atinv_8 = Ai(1,3);
            const double Atinv_9 = Ai(2,0), Atinv_10 = Ai(2,1), Atinv_11 = Ai(2,2), Atinv_12 = Ai(2,3);
            const double Bt_1 = B(0,0), Bt_2 = B(0,1), Bt_3 = B(0,2), Bt_4 = B(0,3);
            const double Bt_5 = B(1,0), Bt_6 = B(1,1), Bt_7 = B(1,2), Bt_8 = B(1,3);
            const double Bt_9 = B(2,0), Bt_10 = B(2,1), Bt_11 = B(2,2), Bt_12 = B(2,3);
            const double R1 = Rm(0,0), R2 = Rm(0,1), R3 = Rm(0,2);
            const double R4 = Rm(1,0), R5 = Rm(1,1), R6 = Rm(1,2);
            const double R7 = Rm(2,0), R8 = Rm(2,1), R9 = Rm(2,2);
            (void)Atinv_4; (void)Atinv_8; (void)Atinv_12;
            (void)Bt_1; (void)Bt_2; (void)Bt_5; (void)Bt_6; (void)Bt_9; (void)Bt_10;
            const double x0 = Atinv_5*R3;
            const double x1 = Bt_3*R1;
            const double x2 = Atinv_2*R6;
            const double x3 = Bt_11*x2;
            const double x4 = 2*Atinv_1;
            const double x5 = x3*x4;
            const double x6 = Bt_3*R4;
            const double x7 = Atinv_1*R3;
            const double x8 = Bt_11*x7;
            const double x9 = 2*Atinv_2;
            const double x10 = x8*x9;
            const double x11 = Bt_7*R2;
            const double x12 = Bt_7*R5;
            const double x13 = Atinv_2*x4;
            const double x14 = x1*x12;
            const double x15 = x11*x6;
            const double x16 = Atinv_3*R9;
            const double x17 = Bt_11*x16;
            const double x18 = x17*x4;
            const double x19 = Atinv_3*x8;
            const double x20 = Bt_3*R7;
            const double x21 = 2*x20;
            const double x22 = Bt_7*R8;
            const double x23 = 2*x22;
            const double x24 = Atinv_3*x4;
            const double x25 = x1*x22;
            const double x26 = x11*x20;
            const double x27 = x17*x9;
            const double x28 = Atinv_3*x3;
            const double x29 = Atinv_3*x9;
            const double x30 = x22*x6;
            const double x31 = x12*x20;
            const double x32 = Atinv_6*R6;
            const double x33 = Bt_11*x32;
            const double x34 = 2*Atinv_5;
            const double x35 = x33*x34;
            const double x36 = Bt_11*x0;
            const double x37 = 2*Atinv_6;
            const double x38 = x36*x37;
            const double x39 = Atinv_6*x34;
            const double x40 = Atinv_7*R9;
            const double x41 = Bt_11*x40;
            const double x42 = x34*x41;
            const double x43 = Atinv_7*x36;
            const double x44 = Atinv_7*x34;
            const double x45 = x37*x41;
            const double x46 = Atinv_7*x33;
            const double x47 = Atinv_7*x37;
            const double x48 = pow(Bt_11, 2);
            const double x49 = 2*x48;
            const double x50 = x49*x7;
            const double x51 = pow(Bt_3, 2);
            const double x52 = R1*x51;
            const double x53 = R4*x52;
            const double x54 = pow(Bt_7, 2);
            const double x55 = R2*x54;
            const double x56 = R5*x55;
            const double x57 = R7*x52;
            const double x58 = R8*x55;
            const double x59 = 2*Bt_11;
            const double x60 = R3*x59;
            const double x61 = pow(Atinv_1, 2);
            const double x62 = x1*x61;
            const double x63 = 2*x11;
            const double x64 = R4*R7*x51;
            const double x65 = R5*R8*x54;
            const double x66 = R6*x59;
            const double x67 = pow(Atinv_2, 2);
            const double x68 = x6*x67;
            const double x69 = 2*x12;
            const double x70 = R9*x59;
            const double x71 = pow(Atinv_3, 2);
            const double x72 = x20*x71;
            const double x73 = x0*x49;
            const double x74 = pow(Atinv_5, 2);
            const double x75 = x60*x74;
            const double x76 = pow(Atinv_6, 2);
            const double x77 = x66*x76;
            const double x78 = pow(Atinv_7, 2);
            const double x79 = x70*x78;
            const double x80 = pow(R3, 2)*x48;
            const double x81 = pow(R1, 2)*x51;
            const double x82 = pow(R2, 2)*x54;
            const double x83 = pow(R6, 2)*x48;
            const double x84 = pow(R4, 2)*x51;
            const double x85 = pow(R5, 2)*x54;
            const double x86 = pow(R9, 2)*x48;
            const double x87 = pow(R7, 2)*x51;
            const double x88 = pow(R8, 2)*x54;
            const double x89 = pow(x1*x18 + x1*x35 + x1*x42 + x1*x5 + x1*x63*x74 + x1*x75 + x10*x12 + x10*x6 + x11*x18 + x11*x35 + x11*x42 + x11*x5 + x11*x60*x61 + x11*x75 + x12*x27 + x12*x38 + x12*x45 + x12*x66*x67 + x12*x77 + x13*x14 + x13*x15 + x13*x53 + x13*x56 + x14*x39 + x15*x39 + x16*x2*x49 + x16*x50 + x19*x21 + x19*x23 + x2*x50 + x20*x23*x78 + x20*x79 + x21*x28 + x21*x43 + x21*x46 + x22*x70*x71 + x22*x79 + x23*x28 + x23*x43 + x23*x46 + x23*x72 + x24*x25 + x24*x26 + x24*x57 + x24*x58 + x25*x44 + x26*x44 + x27*x6 + x29*x30 + x29*x31 + x29*x64 + x29*x65 + x30*x47 + x31*x47 + x32*x40*x49 + x32*x73 + x38*x6 + x39*x53 + x39*x56 + x40*x73 + x44*x57 + x44*x58 + x45*x6 + x47*x64 + x47*x65 + x6*x69*x76 + x6*x77 + x60*x62 + x61*x80 + x61*x81 + x61*x82 + x62*x63 + x66*x68 + x67*x83 + x67*x84 + x67*x85 + x68*x69 + x70*x72 + x71*x86 + x71*x87 + x71*x88 + x74*x80 + x74*x81 + x74*x82 + x76*x83 + x76*x84 + x76*x85 + x78*x86 + x78*x87 + x78*x88, -1.0/2.0);
            const double x90 = Atinv_4*x89;
            const double x91 = Bt_11*x90;
            const double x92 = Atinv_5*x90;
            const double x93 = Atinv_6*x90;
            const double x94 = Atinv_7*x90;
            const double x95 = Atinv_8*x89;
            const double x96 = Bt_11*x95;
            const double x97 = Atinv_1*x95;
            const double x98 = Atinv_2*x95;
            const double x99 = Atinv_3*x95;
            const double x100 = Atinv_1*x89;
            const double x101 = x100*x33;
            const double x102 = Bt_4*R1;
            const double x103 = Bt_8*R2;
            const double x104 = Atinv_6*x89;
            const double x105 = Bt_12*x7;
            const double x106 = x104*x105;
            const double x107 = Atinv_6*x100;
            const double x108 = x107*x6;
            const double x109 = x107*x12;
            const double x110 = x100*x41;
            const double x111 = Atinv_7*x89;
            const double x112 = x105*x111;
            const double x113 = Atinv_7*x100;
            const double x114 = x113*x20;
            const double x115 = x113*x22;
            const double x116 = Atinv_2*x89;
            const double x117 = x116*x36;
            const double x118 = Bt_4*R4;
            const double x119 = Bt_8*R5;
            const double x120 = Atinv_5*x89;
            const double x121 = Bt_12*x2;
            const double x122 = x120*x121;
            const double x123 = Atinv_2*x120;
            const double x124 = x116*x41;
            const double x125 = x111*x121;
            const double x126 = Atinv_7*x116;
            const double x127 = Atinv_3*x89;
            const double x128 = x127*x36;
            const double x129 = Bt_4*R7;
            const double x130 = Bt_8*R8;
            const double x131 = Bt_12*x16;
            const double x132 = x120*x131;
            const double x133 = Atinv_3*x120;
            const double x134 = x127*x33;
            const double x135 = x104*x131;
            const double x136 = Atinv_3*x104;
            const double x137 = x104*x8;
            const double x138 = Bt_12*x100;
            const double x139 = x138*x32;
            const double x140 = x1*x107;
            const double x141 = x107*x11;
            const double x142 = x111*x8;
            const double x143 = x138*x40;
            const double x144 = x1*x113;
            const double x145 = x11*x113;
            const double x146 = x120*x3;
            const double x147 = Bt_12*x116;
            const double x148 = x0*x147;
            const double x149 = x123*x6;
            const double x150 = x12*x123;
            const double x151 = x111*x3;
            const double x152 = x147*x40;
            const double x153 = x126*x6;
            const double x154 = x12*x126;
            const double x155 = x120*x17;
            const double x156 = Bt_12*x127;
            const double x157 = x0*x156;
            const double x158 = x133*x20;
            const double x159 = x133*x22;
            const double x160 = x104*x17;
            const double x161 = x156*x32;
            const double x162 = x136*x20;
            const double x163 = x136*x22;
            c0 = x0*x91 + x1*x119*x123 + x1*x122 + x1*x130*x133 + x1*x132 - x1*x139 - x1*x143 + x1*x92 - x1*x97 + x101*x102 + x101*x103 + x102*x109 + x102*x110 + x102*x115 - x102*x146 - x102*x150 - x102*x155 - x102*x159 + x103*x108 + x103*x110 + x103*x114 - x103*x146 - x103*x149 - x103*x155 - x103*x158 + x106*x12 + x106*x6 + x11*x118*x123 + x11*x122 + x11*x129*x133 + x11*x132 - x11*x139 - x11*x143 + x11*x92 - x11*x97 + x112*x20 + x112*x22 + x117*x118 + x117*x119 + x118*x124 + x118*x126*x22 - x118*x137 - x118*x141 - x118*x160 - x118*x163 + x119*x124 + x119*x126*x20 - x119*x137 - x119*x140 - x119*x160 - x119*x162 + x12*x129*x136 + x12*x135 - x12*x148 - x12*x152 + x12*x93 - x12*x98 + x125*x20 + x125*x22 + x128*x129 + x128*x130 + x129*x134 - x129*x142 - x129*x145 - x129*x151 - x129*x154 + x130*x134 + x130*x136*x6 - x130*x142 - x130*x144 - x130*x151 - x130*x153 + x135*x6 - x148*x6 - x152*x6 - x157*x20 - x157*x22 - x16*x96 - x161*x20 - x161*x22 - x2*x96 + x20*x94 - x20*x99 + x22*x94 - x22*x99 + x32*x91 + x40*x91 + x6*x93 - x6*x98 - x7*x96;
            c1 = x101 + x108 + x109 + x110 + x114 + x115 - x146 - x149 - x150 - x155 - x158 - x159;
            c2 = Atinv_2*Atinv_5*Bt_11*R3*x89 + Atinv_2*Atinv_5*Bt_3*R1*x89 + Atinv_2*Atinv_5*Bt_7*R2*x89 + Atinv_2*Atinv_7*Bt_11*R9*x89 + Atinv_2*Atinv_7*Bt_3*R7*x89 + Atinv_2*Atinv_7*Bt_7*R8*x89 - x137 - x140 - x141 - x160 - x162 - x163;
            c3 = Atinv_3*Atinv_5*Bt_11*R3*x89 + Atinv_3*Atinv_5*Bt_3*R1*x89 + Atinv_3*Atinv_5*Bt_7*R2*x89 + Atinv_3*Atinv_6*Bt_11*R6*x89 + Atinv_3*Atinv_6*Bt_3*R4*x89 + Atinv_3*Atinv_6*Bt_7*R5*x89 - x142 - x144 - x145 - x151 - x153 - x154;
        }

        std::pair<R3, double> solveTranslation(
                const Eigen::VectorXd& c0, const Eigen::MatrixXd& C,
                const std::vector<R3>& vInits) const {
            auto decomposition = C.completeOrthogonalDecomposition();
            R3 bestV = R3::Zero();
            double bestScore = std::numeric_limits<double>::infinity();
            for (const R3& v0 : vInits) {
                R3 v = v0;
                Eigen::VectorXd signs;
                for (int i = 0; i < 30; i++) {
                    Eigen::VectorXd newSigns =
                        ((c0 + C * v).array() >= 0.0).select(
                            Eigen::VectorXd::Ones(c0.size()), -Eigen::VectorXd::Ones(c0.size()));
                    if (signs.size() > 0 && newSigns == signs) break;
                    signs = newSigns;
                    v = decomposition.solve(signs * s - c0);
                }
                double score = ((c0 + C * v).array().abs() - s).abs().maxCoeff();
                if (score < bestScore) {
                    bestV = v;
                    bestScore = score;
                }
            }
            return {bestV, bestScore};
        }

        std::pair<R3, double> chebyshevLP(const Eigen::VectorXd& c0,
                                          const Eigen::MatrixXd& C, const R3& v) const {
            const int k = (int)c0.size();
            Eigen::VectorXd signs = ((c0 + C * v).array() >= 0.0).select(
                Eigen::VectorXd::Ones(k), -Eigen::VectorXd::Ones(k));
            Eigen::MatrixXd sC = signs.asDiagonal() * C;
            // variables: [v+ (3), v- (3), m (1)], v = v+ - v-
            Eigen::MatrixXd A(2 * k, 7);
            A.block(0, 0, k, 3) = sC;
            A.block(0, 3, k, 3) = -sC;
            A.block(k, 0, k, 3) = -sC;
            A.block(k, 3, k, 3) = sC;
            A.col(6).setConstant(-1.0);
            Eigen::VectorXd b(2 * k);
            b.head(k) = Eigen::VectorXd::Constant(k, s) - signs.cwiseProduct(c0);
            b.tail(k) = signs.cwiseProduct(c0) - Eigen::VectorXd::Constant(k, s);
            Eigen::VectorXd cost = Eigen::VectorXd::Zero(7);
            cost[6] = 1.0;
            Eigen::VectorXd x;
            if (!simplexLP(A, b, cost, x))
                return {v, std::numeric_limits<double>::infinity()};
            return {x.head<3>() - x.segment<3>(3), x[6]};
        }

        std::pair<R3, double> evaluateOrientation(
                const Eigen::Quaterniond& q, const std::vector<R3>& vInits) const {
            Eigen::Matrix3d Rm = q.toRotationMatrix();
            const int k = (int)Bs.size();
            Eigen::VectorXd c0(k);
            Eigen::MatrixXd C(k, 3);
            for (int t = 0; t < k; t++)
                coefficients(Rm, t, c0[t], C(t, 0), C(t, 1), C(t, 2));
            auto [v, score] = solveTranslation(c0, C, vInits);
            if (score < 10.0 * eps)
                return chebyshevLP(c0, C, v);
            return {v, score};
        }

        static SE3 weightedAveragePose(const std::vector<Eigen::Quaterniond>& quats,
                                       const std::vector<R3>& translations,
                                       const std::vector<double>& scores, double tau) {
            Eigen::VectorXd weights(scores.size());
            for (size_t i = 0; i < scores.size(); i++)
                weights[i] = std::exp(-(scores[i] / tau) * (scores[i] / tau));
            Eigen::Matrix4d M = Eigen::Matrix4d::Zero();
            Eigen::Vector4d q0 = quats[0].coeffs();
            R3 vAvg = R3::Zero();
            for (size_t i = 0; i < quats.size(); i++) {
                Eigen::Vector4d qi = quats[i].coeffs();
                if (qi.dot(q0) < 0.0) qi = -qi;
                M += weights[i] * qi * qi.transpose();
                vAvg += weights[i] * translations[i];
            }
            Eigen::Vector4d qAvg = Eigen::SelfAdjointEigenSolver<Eigen::Matrix4d>(M)
                .eigenvectors().col(3);
            SE3 X = SE3::Identity();
            X.block<3,3>(0,0) = Eigen::Quaterniond(qAvg).normalized().toRotationMatrix();
            X.block<3,1>(0,3) = vAvg / weights.sum();
            return X;
        }

        SE3 solve(SE3, int = 100, double = 1e-14) const override {
            std::mt19937 gen(seed);
            std::normal_distribution<double> gauss;
            std::uniform_real_distribution<double> box(-2.0, 2.0);
            std::vector<R3> vInits = {R3::Zero()};
            for (int i = 0; i < 4; i++)
                vInits.push_back(R3(box(gen), box(gen), box(gen)));

            struct Sample { double score; Eigen::Quaterniond q; R3 v; };
            std::vector<Sample> samples;
            for (int i = 0; i < nInitial; i++) {
                Eigen::Quaterniond q(gauss(gen), gauss(gen), gauss(gen), gauss(gen));
                q.normalize();
                auto [v, score] = evaluateOrientation(q, vInits);
                samples.push_back({score, q, v});
            }

            auto byScore = [](const Sample& a, const Sample& b) { return a.score < b.score; };
            for (double sigma : sigmaSchedule) {
                std::sort(samples.begin(), samples.end(), byScore);
                size_t numElites = std::min((size_t)nElites, samples.size());
                for (size_t e = 0; e < numElites; e++) {
                    Sample elite = samples[e];
                    for (int i = 0; i < nPerElite; i++) {
                        R3 w(gauss(gen), gauss(gen), gauss(gen));
                        w *= sigma;
                        double angle = w.norm();
                        Eigen::Quaterniond step = angle < 1e-12
                            ? Eigen::Quaterniond::Identity()
                            : Eigen::Quaterniond(Eigen::AngleAxisd(angle, w / angle));
                        Eigen::Quaterniond qNew = elite.q * step;
                        std::vector<R3> warmInits = {elite.v};
                        warmInits.insert(warmInits.end(), vInits.begin(), vInits.end());
                        auto [vNew, scoreNew] = evaluateOrientation(qNew, warmInits);
                        samples.push_back({scoreNew, qNew, vNew});
                    }
                }
            }

            std::sort(samples.begin(), samples.end(), byScore);
            const Sample& best = samples[0];
            std::vector<Sample> feasible;
            for (const Sample& sample : samples)
                if (sample.score <= eps) feasible.push_back(sample);
            if (feasible.empty()) feasible.push_back(best);

            std::vector<Eigen::Quaterniond> qs;
            std::vector<R3> vs;
            std::vector<double> scores;
            for (const Sample& sample : feasible) {
                if (best.q.angularDistance(sample.q) < 0.2 &&
                    (sample.v - best.v).norm() < 0.1) {
                    qs.push_back(sample.q);
                    vs.push_back(sample.v);
                    scores.push_back(sample.score);
                }
            }
            return weightedAveragePose(qs, vs, scores, eps);
        }

    private:
        int nInitial, nElites, nPerElite;
        double eps;
        unsigned seed;
        std::vector<double> sigmaSchedule;
    };

}
