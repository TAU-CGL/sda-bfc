#ifndef DABFC_UR5E_H_
#define DABFC_UR5E_H_

#include <string>

#include "dabfc/robot_arm.h"

namespace dabfc {

// UR5e with a Robotiq 2F-85 gripper (ur_description and robotiq URDFs). URDF frames: Z-up, metres.
// q = (shoulder_pan, shoulder_lift, elbow, wrist_1, wrist_2, wrist_3, gripper) in radians;
// gripper 0 = open, 0.8 = closed
class UR5e : public RobotArm {
public:
    // meshDir: folder with the visual STLs (resources/ur5e/visual)
    explicit UR5e(const std::string& meshDir);

    Eigen::VectorXd home() const override;
    int toolLink() const override; // tool0 (coincides with wrist_3_link); the gripper mounts here
};

} // namespace dabfc

#endif // DABFC_UR5E_H_
