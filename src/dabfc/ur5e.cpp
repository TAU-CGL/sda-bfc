#include "dabfc/ur5e.h"

#include <numbers>
#include <vector>

namespace dabfc {

namespace {

constexpr double pi = std::numbers::pi;
constexpr int tool0 = 7;              // index of tool0 in the chain
constexpr int gripperJoint = 6;       // index of finger_joint in q
constexpr double gripperClosed = 0.8; // finger_joint upper limit

Eigen::VectorXd homeConfiguration()
{
    Eigen::VectorXd q(7);
    q << 0, -pi / 2, 0, -pi / 2, 0, 0, 0;
    return q;
}

// Kinematic tree from ur5e.urdf and robotiq_arg2f_85_model.urdf; SE3(xyz, rpy) is a URDF <origin>.
// The Robotiq finger_joint drives the other finger joints (mimic).
std::vector<Link> ur5eLinks()
{
    const Eigen::Vector3d x = Eigen::Vector3d::UnitX();
    const std::vector<std::string> ur = {"JointGrey", "Black", "URBlue"};
    const std::vector<std::string> urLink = {"JointGrey", "Black", "URBlue", "LinkGrey"};
    // clang-format off
    return {
        {.name = "base",     .mesh = "base",     .meshOrigin = SE3({0, 0, 0}, {0, 0, pi}), .parts = {"JointGrey", "Black"}},
        {.name = "shoulder", .parent = 0, .joint = 0, .origin = SE3({0, 0, 0.1625}), .mesh = "shoulder", .meshOrigin = SE3({0, 0, 0}, {0, 0, pi}), .parts = ur},
        {.name = "upperarm", .parent = 1, .joint = 1, .origin = SE3({0, 0, 0}, {pi / 2, 0, 0}), .mesh = "upperarm", .meshOrigin = SE3({0, 0, 0.138}, {pi / 2, 0, -pi / 2}), .parts = urLink},
        {.name = "forearm",  .parent = 2, .joint = 2, .origin = SE3({-0.425, 0, 0}), .mesh = "forearm", .meshOrigin = SE3({0, 0, 0.007}, {pi / 2, 0, -pi / 2}), .parts = urLink},
        {.name = "wrist1",   .parent = 3, .joint = 3, .origin = SE3({-0.3922, 0, 0.1333}), .mesh = "wrist1", .meshOrigin = SE3({0, 0, -0.127}, {pi / 2, 0, 0}), .parts = ur},
        {.name = "wrist2",   .parent = 4, .joint = 4, .origin = SE3({0, -0.0997, 0}, {pi / 2, 0, 0}), .mesh = "wrist2", .meshOrigin = SE3({0, 0, -0.0997}), .parts = ur},
        {.name = "wrist3",   .parent = 5, .joint = 5, .origin = SE3({0, 0.0996, 0}, {pi / 2, pi, pi}), .mesh = "wrist3", .meshOrigin = SE3({0, 0, -0.0989}, {pi / 2, 0, 0}), .parts = {"LinkGrey"}},
        {.name = "tool0",    .parent = 6},
        {.name = "gripper",         .parent = 7, .mesh = "rq_base_link", .parts = {"RobotiqBlack", "RobotiqGrey"}},
        {.name = "l_outer_knuckle", .parent = 8,  .joint = 6, .origin = SE3({0, -0.0306011, 0.054904}, {0, 0, pi}), .axis = x, .mesh = "rq_outer_knuckle", .parts = {"RobotiqBlack"}},
        {.name = "l_outer_finger",  .parent = 9,  .origin = SE3({0, 0.0315, -0.0041}), .mesh = "rq_outer_finger", .parts = {"RobotiqBlack"}},
        {.name = "l_inner_finger",  .parent = 10, .joint = 6, .mult = -1, .origin = SE3({0, 0.0061, 0.0471}), .axis = x, .mesh = "rq_inner_finger", .parts = {"RobotiqBlack", "RobotiqGrey"}},
        {.name = "l_inner_knuckle", .parent = 8,  .joint = 6, .origin = SE3({0, -0.0127, 0.06142}, {0, 0, pi}), .axis = x, .mesh = "rq_inner_knuckle", .parts = {"RobotiqBlack"}},
        {.name = "r_outer_knuckle", .parent = 8,  .joint = 6, .origin = SE3({0, 0.0306011, 0.054904}), .axis = x, .mesh = "rq_outer_knuckle", .parts = {"RobotiqBlack"}},
        {.name = "r_outer_finger",  .parent = 13, .origin = SE3({0, 0.0315, -0.0041}), .mesh = "rq_outer_finger", .parts = {"RobotiqBlack"}},
        {.name = "r_inner_finger",  .parent = 14, .joint = 6, .mult = -1, .origin = SE3({0, 0.0061, 0.0471}), .axis = x, .mesh = "rq_inner_finger", .parts = {"RobotiqBlack", "RobotiqGrey"}},
        {.name = "r_inner_knuckle", .parent = 8,  .joint = 6, .origin = SE3({0, 0.0127, 0.06142}), .axis = x, .mesh = "rq_inner_knuckle", .parts = {"RobotiqBlack"}},
    };
    // clang-format on
}

} // namespace

UR5e::UR5e(const std::string& meshDir) : RobotArm(ur5eLinks(), meshDir)
{
    // The finger linkage touches itself as the gripper closes: allow those pairs, over its range
    constexpr int steps = 80;
    Eigen::VectorXd q = homeConfiguration();
    for (int k = 0; k <= steps; ++k) {
        q[gripperJoint] = gripperClosed * k / steps;
        allowSelfCollisions(q);
    }
}

Eigen::VectorXd UR5e::home() const
{
    return homeConfiguration();
}

int UR5e::toolLink() const
{
    return tool0;
}

} // namespace dabfc
