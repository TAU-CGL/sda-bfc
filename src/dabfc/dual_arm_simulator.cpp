#include "dabfc/dual_arm_simulator.h"

#include <stdexcept>

namespace dabfc {

namespace {

std::shared_ptr<const RobotArm> notNull(std::shared_ptr<const RobotArm> arm)
{
    if (!arm) throw std::invalid_argument("DualArmSimulator: null arm");
    return arm;
}

} // namespace

DualArmSimulator::DualArmSimulator(std::shared_ptr<const RobotArm> arm1, const SE3& base1,
                                   std::shared_ptr<const RobotArm> arm2, const SE3& base2)
    : m_arms{notNull(std::move(arm1)), notNull(std::move(arm2))}, m_bases{base1, base2},
      m_q{m_arms[0]->home(), m_arms[1]->home()}
{
}

void DualArmSimulator::setState(const State& q)
{
    for (int a = 0; a < 2; ++a)
        if (q[a].size() != m_arms[a]->dof())
            throw std::invalid_argument("DualArmSimulator::setState: wrong number of joint values");
    m_q = q;
}

void DualArmSimulator::addObstacle(std::shared_ptr<const collision::Shape> shape, const SE3& pose)
{
    if (!shape) throw std::invalid_argument("DualArmSimulator::addObstacle: null shape");
    m_obstacles.push_back({std::move(shape), pose});
}

bool DualArmSimulator::isCollisionFreeAt(const State& q) const
{
    std::array<std::vector<SE3>, 2> poses;
    for (int a = 0; a < 2; ++a) {
        poses[a] = m_arms[a]->meshPoses(q[a], m_bases[a]);
        if (!m_arms[a]->isSelfCollisionFree(poses[a])) return false;
        for (const Obstacle& obstacle : m_obstacles)
            if (m_arms[a]->touches(poses[a], *obstacle.shape, obstacle.pose)) return false;
    }
    return !m_arms[0]->touches(poses[0], *m_arms[1], poses[1]);
}

} // namespace dabfc
