#include "dabfc/dual_arm_simulator.h"
#include "dabfc/ur5e.h"

#include <gtest/gtest.h>

namespace dabfc {
namespace {

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
    DualArmSimulator::State home = sim.state(), q = home;
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

} // namespace
} // namespace dabfc
