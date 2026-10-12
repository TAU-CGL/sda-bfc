#ifndef DABFC_DUAL_ARM_SIMULATOR_H_
#define DABFC_DUAL_ARM_SIMULATOR_H_

#include <array>
#include <limits>
#include <memory>
#include <optional>
#include <random>
#include <vector>

#include "dabfc/collision.h"
#include "dabfc/robot_arm.h"
#include "dabfc/se3.h"

namespace dabfc {

using State = std::array<Eigen::VectorXd, 2>; // (q1, q2): the joint values of both arms

struct SimulationCollisionReport {
    // A link touching a link of an arm (otherArm) or an obstacle (none; otherLink is its index)
    struct Contact {
        int arm;
        int link;
        std::optional<int> otherArm;
        int otherLink;
        Eigen::Vector3d point; // world frame
    };
    State q; // the configuration in contact
    std::vector<Contact> contacts;
};

class DualArmSimulator {
public:
    // The same model may serve as both arms; throws std::invalid_argument on a null arm
    DualArmSimulator(std::shared_ptr<const RobotArm> arm1, const SE3& base1,
                     std::shared_ptr<const RobotArm> arm2, const SE3& base2);

    std::shared_ptr<const RobotArm> arm(int i) const { return m_arms.at(i); }
    const SE3& base(int i) const { return m_bases.at(i); }
    void setBase(int i, const SE3& base) { m_bases.at(i) = base; }

    const State& state() const { return m_q; }
    void setState(const State& q);

    void addObstacle(std::shared_ptr<const collision::Shape> shape, const SE3& pose);

    // A random pose for arm 2's base near arm 1's, in its frame: x and y within 0.75 m, z within 0.15 m,
    // any yaw, tilted by at most 3 degrees
    SE3 randomPlacementOfArm2(std::mt19937& rng) const;

    // Everything touching at q (empty: collision free). Throws std::invalid_argument if the size
    // of a q[i] is not arm(i)->dof()
    std::vector<SimulationCollisionReport::Contact> contactsAt(const State& q) const;
    bool isCollisionFreeAt(const State& q) const { return contactsAt(q).empty(); }
    bool isCollisionFreeNow() const { return isCollisionFreeAt(m_q); }

    // Moves arm i linearly in joint space towards target, by at most distance (the largest joint
    // change, rad), checking collisions every resolution() rad. On contact it stops at the last
    // collision-free configuration and reports the one in contact
    std::optional<SimulationCollisionReport>
    moveJ(int i, const Eigen::VectorXd& target,
          double distance = std::numeric_limits<double>::infinity());
    double resolution() const { return m_resolution; }
    void setResolution(double resolution) { m_resolution = resolution; }

private:
    struct Obstacle {
        std::shared_ptr<const collision::Shape> shape;
        SE3 pose;
    };

    std::array<std::shared_ptr<const RobotArm>, 2> m_arms;
    std::array<SE3, 2> m_bases;
    State m_q;
    std::vector<Obstacle> m_obstacles;
    double m_resolution = 0.005; // rad, about 0.3 degrees
};

} // namespace dabfc

#endif // DABFC_DUAL_ARM_SIMULATOR_H_
