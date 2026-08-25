#pragma once

#include <algorithm>
#include <optional>
#include <random>

#include "world.hpp"

namespace sda_bfc {

    using JointPath = std::vector<JointConfig>;

    class Planner {
    public:
        virtual ~Planner() = default;
        virtual std::optional<JointPath> plan(const PlanningWorld& world,
                                              const JointConfig& start,
                                              const JointConfig& goal,
                                              unsigned seed) const = 0;
    };

    class RRTPlanner : public Planner {
    public:
        RRTPlanner(int maxIterations = 3000, double stepSize = 0.3,
                   double goalBias = 0.15, int shortcutTries = 60,
                   double jointRange = M_PI) :
            maxIterations_(maxIterations), stepSize_(stepSize),
            goalBias_(goalBias), shortcutTries_(shortcutTries),
            jointRange_(jointRange) {}

        std::optional<JointPath> plan(const PlanningWorld& world,
                                      const JointConfig& start,
                                      const JointConfig& goal,
                                      unsigned seed) const override {
            if (!world.isFree(start) || !world.isFree(goal)) return std::nullopt;
            std::mt19937 gen(seed);
            if (world.edgeFree(start, goal)) {
                return shortcut(world, {start, goal}, gen);
            }
            std::vector<JointConfig> nodes{start};
            std::vector<int> parents{-1};
            std::uniform_real_distribution<double> joint(-jointRange_, jointRange_);
            std::uniform_real_distribution<double> unit(0.0, 1.0);
            for (int iter = 0; iter < maxIterations_; iter++) {
                JointConfig target;
                if (unit(gen) < goalBias_) {
                    target = goal;
                } else {
                    for (int i = 0; i < 6; i++) target[i] = joint(gen);
                }
                int nearest = nearestNode(nodes, target);
                JointConfig q = steer(nodes[nearest], target);
                if (!world.edgeFree(nodes[nearest], q)) continue;
                nodes.push_back(q);
                parents.push_back(nearest);
                if ((q - goal).norm() < stepSize_ && world.edgeFree(q, goal)) {
                    nodes.push_back(goal);
                    parents.push_back((int)nodes.size() - 2);
                    return shortcut(world, backtrack(nodes, parents), gen);
                }
            }
            return std::nullopt;
        }

    private:
        static int nearestNode(const std::vector<JointConfig>& nodes,
                               const JointConfig& target) {
            int best = 0;
            double bestDistance = (nodes[0] - target).squaredNorm();
            for (int i = 1; i < (int)nodes.size(); i++) {
                double distance = (nodes[i] - target).squaredNorm();
                if (distance < bestDistance) {
                    best = i;
                    bestDistance = distance;
                }
            }
            return best;
        }

        JointConfig steer(const JointConfig& from, const JointConfig& to) const {
            JointConfig delta = to - from;
            double length = delta.norm();
            return length <= stepSize_ ? to
                : JointConfig(from + delta * (stepSize_ / length));
        }

        static JointPath backtrack(const std::vector<JointConfig>& nodes,
                                   const std::vector<int>& parents) {
            JointPath path;
            for (int i = (int)nodes.size() - 1; i >= 0; i = parents[i]) {
                path.push_back(nodes[i]);
            }
            std::reverse(path.begin(), path.end());
            return path;
        }

        JointPath shortcut(const PlanningWorld& world, JointPath path,
                           std::mt19937& gen) const {
            std::uniform_int_distribution<int> index(0, 1 << 20);
            for (int attempt = 0; attempt < shortcutTries_; attempt++) {
                if (path.size() <= 2) break;
                int i = index(gen) % path.size();
                int j = index(gen) % path.size();
                if (i > j) std::swap(i, j);
                if (j - i < 2) continue;
                if (world.edgeFree(path[i], path[j])) {
                    path.erase(path.begin() + i + 1, path.begin() + j);
                }
            }
            return path;
        }

        int maxIterations_;
        double stepSize_, goalBias_;
        int shortcutTries_;
        double jointRange_;
    };

}
