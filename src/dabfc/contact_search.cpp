#include "dabfc/contact_search.h"

#include <algorithm>
#include <cmath>
#include <numbers>
#include <string>

namespace dabfc {

namespace {

// Every link frame at or above the base plane: nothing hangs below the robot's mounting
bool upright(const RobotArm& arm, const Eigen::VectorXd& q)
{
    return std::ranges::all_of(arm.fk(q),
                               [](const SE3& frame) { return frame.translation().z() >= 0; });
}

bool armsTouch(const SimulationCollisionReport& report)
{
    return std::ranges::any_of(report.contacts, [](const SimulationCollisionReport::Contact& c) {
        return c.otherArm && *c.otherArm != c.arm;
    });
}

} // namespace

ContactSearch::ContactSearch(DualArmSimulator& sim, int contactCount, unsigned seed)
    : m_sim(sim), m_contactCount(contactCount), m_rng(seed)
{
    m_sim.setState({m_sim.arm(0)->home(), m_sim.arm(1)->home()});
}

bool ContactSearch::step(double distance)
{
    if (std::ssize(m_contacts) >= m_contactCount) return false;
    if (m_forward.empty() && m_back.empty()) plan();
    const bool forward = !m_forward.empty();
    std::vector<Move>& moves = forward ? m_forward : m_back;
    const Move move = moves.back();
    const auto report = m_sim.moveJ(move.arm, move.to, distance);
    if (report && armsTouch(*report)) m_contacts.push_back(*report);
    if (!report && m_sim.state()[move.arm] != move.to) return true; // still on its way
    moves.pop_back();                                               // arrived, or blocked
    if (!forward) return true;
    m_back.push_back({.arm = move.arm, .from = move.to, .to = move.from});
    if (report) m_forward.clear();
    return true;
}

std::string ContactSearch::status() const
{
    if (std::ssize(m_contacts) >= m_contactCount) return "done";
    if (!m_forward.empty()) {
        const char* const moves[] = {"arm 1 turning its base", "arm 2 to a random configuration",
                                     "arm 1 to a random configuration"};
        return moves[m_forward.size() - 1];
    }
    if (!m_back.empty()) return "arm " + std::to_string(m_back.back().arm + 1) + " going home";
    return "starting a cycle";
}

void ContactSearch::plan()
{
    ++m_cycles;
    const State& q = m_sim.state();
    const Eigen::VectorXd q1 = randomConfiguration(0);
    Eigen::VectorXd turned = q1; // a full turn of the base joint, staying within its range
    turned[0] -= std::copysign(2 * std::numbers::pi, q1[0]);
    m_forward = {{.arm = 0, .from = q1, .to = turned},
                 {.arm = 1, .from = q[1], .to = randomConfiguration(1)},
                 {.arm = 0, .from = q[0], .to = q1}};
}

Eigen::VectorXd ContactSearch::randomConfiguration(int arm)
{
    const std::shared_ptr<const RobotArm> model = m_sim.arm(arm);
    const Eigen::VectorXd lower = model->lowerLimits();
    const Eigen::VectorXd upper = model->upperLimits();
    Eigen::VectorXd q(lower.size());
    for (int tries = 0; tries < 1000; ++tries) {
        for (int k = 0; k < q.size(); ++k)
            q[k] = std::uniform_real_distribution<double>(lower[k], upper[k])(m_rng);
        if (upright(*model, q)) break;
    }
    return q;
}

} // namespace dabfc
