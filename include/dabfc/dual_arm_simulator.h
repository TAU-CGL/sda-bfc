#ifndef DABFC_DUAL_ARM_SIMULATOR_H_
#define DABFC_DUAL_ARM_SIMULATOR_H_

#include <array>
#include <memory>
#include <vector>

#include "dabfc/collision.h"
#include "dabfc/robot_arm.h"
#include "dabfc/se3.h"

namespace dabfc {

class DualArmSimulator {
public:
    using State = std::array<Eigen::VectorXd, 2>; // (q1, q2)

    // The same model may serve as both arms; throws std::invalid_argument on a null arm
    DualArmSimulator(std::shared_ptr<const RobotArm> arm1, const SE3& base1,
                     std::shared_ptr<const RobotArm> arm2, const SE3& base2);

    std::shared_ptr<const RobotArm> arm(int i) const { return m_arms.at(i); }
    const SE3& base(int i) const { return m_bases.at(i); }
    void setBase(int i, const SE3& base) { m_bases.at(i) = base; }

    const State& state() const { return m_q; }
    void setState(const State& q);

    void addObstacle(std::shared_ptr<const collision::Shape> shape, const SE3& pose);

    // No self-collision, no contact between the arms, no contact with an obstacle.
    // Throws std::invalid_argument if the size of a q[i] is not arm(i)->dof()
    bool isCollisionFreeAt(const State& q) const;
    bool isCollisionFreeNow() const { return isCollisionFreeAt(m_q); }

private:
    struct Obstacle {
        std::shared_ptr<const collision::Shape> shape;
        SE3 pose;
    };

    std::array<std::shared_ptr<const RobotArm>, 2> m_arms;
    std::array<SE3, 2> m_bases;
    State m_q;
    std::vector<Obstacle> m_obstacles;
};

} // namespace dabfc

#endif // DABFC_DUAL_ARM_SIMULATOR_H_
