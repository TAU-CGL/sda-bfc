#include "dabfc/dual_arm_simulator.h"

#include <algorithm>
#include <cmath>
#include <numbers>
#include <stdexcept>

namespace dabfc {

namespace {

constexpr double pi = std::numbers::pi;
constexpr double maxBaseXY = 0.75;           // m
constexpr double maxBaseZ = 0.15;            // m
constexpr double maxBaseTilt = 3 * pi / 180; // rad

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

SE3 DualArmSimulator::randomPlacementOfArm2(std::mt19937& rng) const
{
    std::uniform_real_distribution<double> xy(-maxBaseXY, maxBaseXY);
    std::uniform_real_distribution<double> z(-maxBaseZ, maxBaseZ);
    std::uniform_real_distribution<double> angle(-pi, pi);
    std::uniform_real_distribution<double> tilt(0, maxBaseTilt);
    const Eigen::Vector3d xyz(xy(rng), xy(rng), z(rng));
    const double tiltDirection = angle(rng);
    const Eigen::Vector3d tiltAxis(std::cos(tiltDirection), std::sin(tiltDirection), 0);
    const Eigen::Quaterniond rotation = Eigen::AngleAxisd(angle(rng), Eigen::Vector3d::UnitZ()) *
                                        Eigen::AngleAxisd(tilt(rng), tiltAxis);
    return m_bases[0] * SE3(xyz, rotation);
}

std::vector<SimulationCollisionReport::Contact> DualArmSimulator::contactsAt(const State& q) const
{
    std::vector<SimulationCollisionReport::Contact> contacts;
    std::array<std::vector<SE3>, 2> poses;
    for (int a = 0; a < 2; ++a) {
        poses[a] = m_arms[a]->meshPoses(q[a], m_bases[a]);
        for (const RobotArm::LinkContact& c : m_arms[a]->selfContacts(poses[a]))
            contacts.push_back({.arm = a,
                                .link = c.link,
                                .otherArm = a,
                                .otherLink = *c.otherLink,
                                .point = c.point});
        for (int o = 0; o < std::ssize(m_obstacles); ++o)
            for (const RobotArm::LinkContact& c :
                 m_arms[a]->contacts(poses[a], *m_obstacles[o].shape, m_obstacles[o].pose))
                contacts.push_back({.arm = a, .link = c.link, .otherLink = o, .point = c.point});
    }
    for (const RobotArm::LinkContact& c : m_arms[0]->contacts(poses[0], *m_arms[1], poses[1]))
        contacts.push_back(
            {.arm = 0, .link = c.link, .otherArm = 1, .otherLink = *c.otherLink, .point = c.point});
    return contacts;
}

std::optional<SimulationCollisionReport>
DualArmSimulator::moveJ(int i, const Eigen::VectorXd& target, double distance)
{
    const Eigen::VectorXd start = m_q.at(i);
    if (target.size() != start.size())
        throw std::invalid_argument("DualArmSimulator::moveJ: wrong number of joint values");
    const double length = (target - start).lpNorm<Eigen::Infinity>();
    const double travel = std::min(distance, length);
    const int steps = static_cast<int>(std::ceil(travel / m_resolution));
    State q = m_q;
    for (int k = 1; k <= steps; ++k) {
        const double t = std::min(k * m_resolution, travel) / length; // 1 exactly on arrival
        q[i] = (1 - t) * start + t * target;
        std::vector<SimulationCollisionReport::Contact> contacts = contactsAt(q);
        if (!contacts.empty()) return SimulationCollisionReport{q, std::move(contacts)};
        m_q = q;
    }
    return std::nullopt;
}

} // namespace dabfc
