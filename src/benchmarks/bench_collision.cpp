// Cost of a DualArmSimulator collision check with the arms apart and interleaved, and of moveJ
// (a full turn of arm 1's base joint)
#include <chrono>
#include <cstdio>
#include <memory>

#include "dabfc/dual_arm_simulator.h"
#include "dabfc/ur5e.h"

using namespace dabfc;

template <typename F> double seconds(F f)
{
    const auto start = std::chrono::steady_clock::now();
    f();
    return std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
}

int main()
{
    const std::shared_ptr<const RobotArm> ur5e = std::make_shared<UR5e>(MESH_DIR);
    DualArmSimulator sim(ur5e, SE3{}, ur5e, SE3({-0.6, 0, 0}));
    constexpr int checks = 1000;
    const double apart = seconds([&] {
        for (int i = 0; i < checks; ++i) sim.isCollisionFreeNow();
    });
    Eigen::VectorXd turned = sim.state()[0];
    turned[0] += 2 * 3.14159265358979;
    const double steps = turned[0] / sim.resolution();
    const double turn = seconds([&] { sim.moveJ(0, turned); });

    sim.setBase(1, SE3({-1.2, 0, 0}, {0, 0, 3.14159265358979})); // facing each other
    State reaching = sim.state();
    reaching[0] << 0, -1.2, 1.2, -1.5, 0, 0, 0; // both stretched towards the middle
    reaching[1] = reaching[0];
    sim.setState(reaching);
    const bool free = sim.isCollisionFreeNow();
    const double near = seconds([&] {
        for (int i = 0; i < checks; ++i) sim.isCollisionFreeNow();
    });
    std::printf("check, arms apart: %.1f us\ncheck, arms interleaved (%s): %.1f us\n"
                "moveJ full turn: %.3f s (%.1f us per step)\n",
                1e6 * apart / checks, free ? "free" : "in contact", 1e6 * near / checks, turn,
                1e6 * turn / steps);
}
