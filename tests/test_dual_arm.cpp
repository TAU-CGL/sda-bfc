#include "dabfc/contact_search.h"
#include "dabfc/dual_arm_simulator.h"
#include "dabfc/ur5e.h"

#include <algorithm>
#include <limits>

#include <gtest/gtest.h>

namespace dabfc {
namespace {

bool armsTouch(const SimulationCollisionReport& report)
{
    return std::ranges::any_of(report.contacts, [](const SimulationCollisionReport::Contact& c) {
        return c.otherArm && *c.otherArm != c.arm;
    });
}

// Standard UR5e tool0 position at q = 0
TEST(UR5eTest, ZeroPoseFk)
{
    UR5e arm(MESH_DIR);
    Eigen::Vector3d tool = arm.fk(Eigen::VectorXd::Zero(7))[arm.toolLink()].translation();
    EXPECT_LT((tool - Eigen::Vector3d(-0.8172, -0.2329, 0.0628)).norm(), 1e-9);
}

TEST(DualArmSimulatorTest, Collisions)
{
    const std::shared_ptr<const RobotArm> ur5e = std::make_shared<UR5e>(MESH_DIR);
    DualArmSimulator sim(ur5e, SE3{}, ur5e, SE3({-0.6, 0, 0}));
    State home = sim.state(), q = home;
    EXPECT_TRUE(sim.isCollisionFreeNow());
    for (int k = 0; k <= 8; ++k) { // closing the gripper is never a self-collision
        q[0][6] = 0.1 * k;
        EXPECT_TRUE(sim.isCollisionFreeAt(q));
    }
    q[0] = Eigen::VectorXd::Zero(7); // arm 1 stretched along -x, through arm 2
    EXPECT_FALSE(sim.isCollisionFreeAt(q));
    q[0] = home[0];
    q[0][2] = 3.1; // elbow folded onto the upper arm
    EXPECT_FALSE(sim.isCollisionFreeAt(q));
    EXPECT_EQ(sim.state()[0], home[0]); // queries never change the state

    sim.setBase(1, SE3({5, 0, 0}));
    q[0] = Eigen::VectorXd::Zero(7);
    EXPECT_TRUE(sim.isCollisionFreeAt(q));
    sim.addObstacle(std::make_shared<fcl::Boxd>(0.1, 0.1, 0.1), SE3({-0.8172, -0.2329, 0.0628}));
    EXPECT_FALSE(sim.isCollisionFreeAt(q));
    EXPECT_TRUE(sim.isCollisionFreeAt(home));
}

TEST(DualArmSimulatorTest, MoveJ)
{
    const std::shared_ptr<const RobotArm> ur5e = std::make_shared<UR5e>(MESH_DIR);
    DualArmSimulator sim(ur5e, SE3{}, ur5e, SE3({-0.6, 0, 0}));
    const State home = sim.state();
    const Eigen::VectorXd stretched = Eigen::VectorXd::Zero(7); // along -x, through arm 2

    const auto hit = sim.moveJ(0, stretched);
    ASSERT_TRUE(hit);
    EXPECT_TRUE(armsTouch(*hit));
    EXPECT_FALSE(sim.isCollisionFreeAt(hit->q));
    EXPECT_TRUE(sim.isCollisionFreeNow()); // stopped one step before the contact
    EXPECT_LE((hit->q[0] - sim.state()[0]).lpNorm<Eigen::Infinity>(), sim.resolution() + 1e-12);

    EXPECT_FALSE(sim.moveJ(0, home[0])); // a free move arrives exactly
    EXPECT_EQ(sim.state()[0], home[0]);
    EXPECT_FALSE(sim.moveJ(0, stretched, 0.1)); // a limited one gets this far
    EXPECT_NEAR((sim.state()[0] - home[0]).lpNorm<Eigen::Infinity>(), 0.1, 1e-12);
}

TEST(ContactSearchTest, FindsContactsBetweenTheArms)
{
    const std::shared_ptr<const RobotArm> ur5e = std::make_shared<UR5e>(MESH_DIR);
    DualArmSimulator sim(ur5e, SE3{}, ur5e, SE3({-0.6, 0, 0}));
    ContactSearch search(sim, 2, 0);
    while (search.step(std::numeric_limits<double>::infinity()) && search.cycles() < 100) {
    }
    ASSERT_EQ(search.contacts().size(), 2);
    for (const SimulationCollisionReport& report : search.contacts()) {
        EXPECT_TRUE(armsTouch(report));
        EXPECT_FALSE(sim.isCollisionFreeAt(report.q));
    }
    EXPECT_TRUE(sim.isCollisionFreeNow());
}

} // namespace
} // namespace dabfc
