#pragma once

#include "solver_annealing_lp.hpp"

namespace sda_bfc {

    // Sequential-Monte-Carlo variant of the annealing-LP solver: instead of a
    // hard elite cutoff, particles are softly reweighted (ESS-adaptive
    // temperature), systematically resampled, and jittered with the shrinking
    // sigma schedule; a fresh uniform fraction guards against mode collapse.
    class SolverSMC : public SolverAnnealingLP {
    public:
        SolverSMC(const std::vector<SE3>& As, const std::vector<SE3>& Bs,
                  double radiusA, double radiusB,
                  int numParticles = 1000, double freshFraction = 0.1,
                  double essTarget = 0.02, double eps = 2e-3, unsigned seed = 0,
                  std::vector<double> sigmaSchedule = {0.3, 0.2, 0.15, 0.1, 0.075, 0.05, 0.03, 0.02, 0.01}) :
            SolverAnnealingLP(As, Bs, radiusA, radiusB, numParticles, 0, 0,
                              eps, seed, std::move(sigmaSchedule)),
            numParticles(numParticles), freshFraction(freshFraction), essTarget(essTarget) {}

        SE3 solve(SE3, int = 100, double = 1e-14) const override {
            std::mt19937 gen(seed);
            std::normal_distribution<double> gauss;
            std::uniform_real_distribution<double> box(-2.0, 2.0);
            std::uniform_real_distribution<double> unif(0.0, 1.0);
            std::vector<R3> vInits = {R3::Zero()};
            for (int i = 0; i < 4; i++)
                vInits.push_back(R3(box(gen), box(gen), box(gen)));

            struct Particle { double score; Eigen::Quaterniond q; R3 v; };
            auto uniformQuat = [&]() {
                Eigen::Quaterniond q(gauss(gen), gauss(gen), gauss(gen), gauss(gen));
                q.normalize();
                return q;
            };
            auto evaluate = [&](const Eigen::Quaterniond& q, const std::vector<R3>& inits) {
                auto [v, score] = evaluateOrientation(q, inits);
                return Particle{score, q, v};
            };

            std::vector<Particle> particles;
            particles.reserve(numParticles);
            for (int i = 0; i < numParticles; i++)
                particles.push_back(evaluate(uniformQuat(), vInits));

            for (double sigma : sigmaSchedule) {
                Eigen::ArrayXd scoresSq(numParticles);
                for (int i = 0; i < numParticles; i++)
                    scoresSq[i] = particles[i].score * particles[i].score;
                double minSq = scoresSq.minCoeff();
                // Weights w_i = exp((min^2 - s_i^2)/tau^2), tau bisected so the
                // effective sample size hits essTarget * N.
                auto weightsAt = [&](double tau) {
                    return ((minSq - scoresSq) / (tau * tau)).exp();
                };
                auto essAt = [&](double tau) {
                    Eigen::ArrayXd w = weightsAt(tau);
                    double sw = w.sum();
                    return sw * sw / (w.square().sum() + 1e-300);
                };
                double lo = 1e-6, hi = 10.0;
                for (int it = 0; it < 80; it++) {
                    double mid = std::sqrt(lo * hi);
                    (essAt(mid) < essTarget * numParticles ? lo : hi) = mid;
                }
                Eigen::ArrayXd weights = weightsAt(std::sqrt(lo * hi));
                weights /= weights.sum();

                int numFresh = (int)(freshFraction * numParticles);
                int numResampled = numParticles - numFresh;
                std::vector<Particle> next;
                next.reserve(numParticles);
                double u0 = unif(gen) / numResampled;
                double cumulative = weights[0];
                int idx = 0;
                for (int i = 0; i < numResampled; i++) {
                    double target = u0 + (double)i / numResampled;
                    while (cumulative < target && idx + 1 < numParticles)
                        cumulative += weights[++idx];
                    const Particle& parent = particles[idx];
                    R3 w(gauss(gen), gauss(gen), gauss(gen));
                    w *= sigma;
                    double angle = w.norm();
                    Eigen::Quaterniond step = angle < 1e-12
                        ? Eigen::Quaterniond::Identity()
                        : Eigen::Quaterniond(Eigen::AngleAxisd(angle, w / angle));
                    std::vector<R3> warmInits = {parent.v};
                    warmInits.insert(warmInits.end(), vInits.begin(), vInits.end());
                    Particle child = evaluate(parent.q * step, warmInits);
                    // Elitist accept: resampling must not lose the incumbent.
                    next.push_back(child.score < parent.score ? child : parent);
                }
                for (int i = 0; i < numFresh; i++)
                    next.push_back(evaluate(uniformQuat(), vInits));
                particles = std::move(next);
            }

            std::sort(particles.begin(), particles.end(),
                      [](const Particle& a, const Particle& b) { return a.score < b.score; });
            const Particle& best = particles[0];
            std::vector<Particle> feasible;
            for (const Particle& particle : particles)
                if (particle.score <= eps) feasible.push_back(particle);
            if (feasible.empty()) feasible.push_back(best);

            std::vector<Eigen::Quaterniond> qs;
            std::vector<R3> vs;
            std::vector<double> scores;
            for (const Particle& particle : feasible) {
                if (best.q.angularDistance(particle.q) < 0.2 &&
                    (particle.v - best.v).norm() < 0.1) {
                    qs.push_back(particle.q);
                    vs.push_back(particle.v);
                    scores.push_back(particle.score);
                }
            }
            return weightedAveragePose(qs, vs, scores, eps);
        }

    private:
        int numParticles;
        double freshFraction, essTarget;
    };

}
