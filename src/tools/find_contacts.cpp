// Offline contact search between two UR5e arms, as fast as possible: arm 2 gets a random base
// near arm 1 (DualArmSimulator::randomBase2), then a ContactSearch runs until it has found the
// requested number of contact configurations, or gives up after the given number of cycles.
// Usage: find_contacts [contacts = 50] [seed = 0] [cycles = 1000]
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <memory>
#include <random>

#include "dabfc/contact_search.h"
#include "dabfc/dual_arm_simulator.h"
#include "dabfc/ur5e.h"

using namespace dabfc;

namespace {

void print(const Eigen::VectorXd& q)
{
    for (double v : q) std::printf("%.6f ", v);
}

} // namespace

int main(int argc, char** argv)
{
    const int count = argc > 1 ? std::atoi(argv[1]) : 50;
    const unsigned seed = argc > 2 ? static_cast<unsigned>(std::atoi(argv[2])) : 0;
    const int maxCycles = argc > 3 ? std::atoi(argv[3]) : 1000;
    const std::shared_ptr<const RobotArm> ur5e = std::make_shared<UR5e>(MESH_DIR);
    DualArmSimulator sim(ur5e, SE3{}, ur5e, SE3{});
    std::mt19937 rng(seed);
    for (int tries = 0; tries < 100; ++tries) {
        sim.setBase(1, sim.randomBase2(rng));
        if (sim.isCollisionFreeNow()) break;
    }
    const Eigen::Vector3d p = sim.base(1).translation();
    const Eigen::Quaterniond r = sim.base(1).rotation();
    std::fprintf(stderr, "arm 2 is %.2f m away\n", p.norm());

    const auto start = std::chrono::steady_clock::now();
    ContactSearch search(sim, count, seed);
    int reachedSweep = 0; // cycles in which both arms got to their random configurations
    while (search.step(std::numeric_limits<double>::infinity()) && search.cycles() <= maxCycles)
        reachedSweep += search.status() == "arm 1 turning its base";
    const double seconds =
        std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();

    std::printf("# base2: x y z qw qx qy qz = %.6f %.6f %.6f %.6f %.6f %.6f %.6f\n", p.x(), p.y(),
                p.z(), r.w(), r.x(), r.y(), r.z());
    std::printf("# columns: q1[%d] q2[%d] link1 link2 x y z (world point)\n", ur5e->dof(),
                ur5e->dof());
    for (const SimulationCollisionReport& report : search.contacts())
        for (const SimulationCollisionReport::Contact& c : report.contacts) {
            if (!c.otherArm || *c.otherArm == c.arm) continue;
            print(report.q[0]);
            print(report.q[1]);
            std::printf("%s %s %.6f %.6f %.6f\n", ur5e->links()[c.link].name.c_str(),
                        ur5e->links()[c.otherLink].name.c_str(), c.point.x(), c.point.y(),
                        c.point.z());
        }
    std::fprintf(stderr, "%zu contacts in %d cycles (%d reached the sweep), %.2f s\n",
                 search.contacts().size(), search.cycles(), reachedSweep, seconds);
}
