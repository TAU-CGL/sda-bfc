#ifndef DABFC_CONTACT_SEARCH_H_
#define DABFC_CONTACT_SEARCH_H_

#include <random>
#include <string>
#include <vector>

#include <Eigen/Core>

#include "dabfc/dual_arm_simulator.h"
#include "dabfc/motion.h"

namespace dabfc {

// Finds configurations where the two arms touch. Each cycle: arm 2 moves to a random upright
// (no link below its base), far-reaching configuration along the shortest joint-space path, then
// arm 1 does the same and turns its base joint a full circle; after a contact, or after the turn, the moves are
// retraced in reverse order, back home along the collision-free segments just travelled.
// Contacts between the arms are kept; a self-contact only ends the cycle
class ContactSearch : public Motion {
public:
    ContactSearch(DualArmSimulator& sim, int contactCount, unsigned seed); // homes both arms

    bool step(double distance) override;
    std::string status() const; // what the arms are doing, for a display
    const std::vector<SimulationCollisionReport>& contacts() const { return m_contacts; }
    int cycles() const { return m_cycles; }

private:
    struct Move {
        int arm;
        Eigen::VectorXd from;
        Eigen::VectorXd to;
    };

    void plan(); // a new cycle
    Eigen::VectorXd randomConfiguration(int arm, const Eigen::VectorXd& from); // reached by the short arcs

    DualArmSimulator& m_sim;
    int m_contactCount;
    std::mt19937 m_rng;
    std::vector<Move> m_forward; // the cycle's remaining moves, next one last
    std::vector<Move> m_back;    // the way home, next move last
    std::vector<SimulationCollisionReport> m_contacts;
    int m_cycles = 0;
};

} // namespace dabfc

#endif // DABFC_CONTACT_SEARCH_H_
