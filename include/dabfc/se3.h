#ifndef DABFC_SE3_H_
#define DABFC_SE3_H_

#include <Eigen/Geometry>

namespace dabfc {

class SE3 {
public:
    SE3() = default; // identity

    // URDF <origin xyz rpy>: rotation about the fixed axes X (roll), Y (pitch), Z (yaw), then xyz
    explicit SE3(const Eigen::Vector3d& xyz, const Eigen::Vector3d& rpy = Eigen::Vector3d::Zero())
        : SE3(xyz, Eigen::AngleAxisd(rpy.z(), Eigen::Vector3d::UnitZ()) *
                       Eigen::AngleAxisd(rpy.y(), Eigen::Vector3d::UnitY()) *
                       Eigen::AngleAxisd(rpy.x(), Eigen::Vector3d::UnitX()))
    {
    }
    SE3(const Eigen::Vector3d& translation, const Eigen::Quaterniond& rotation)
        : m_isometry(Eigen::Translation3d(translation) * rotation.normalized())
    {
    }
    explicit SE3(const Eigen::Isometry3d& isometry) : m_isometry(isometry) {}

    Eigen::Vector3d translation() const { return m_isometry.translation(); }
    Eigen::Quaterniond rotation() const { return Eigen::Quaterniond(m_isometry.linear()); }
    const Eigen::Isometry3d& isometry() const { return m_isometry; } // e.g. as an fcl::Transform3d

    // Composition: if b is a frame given in frame a, a * b is that frame given in a's parent frame
    friend SE3 operator*(const SE3& a, const SE3& b) { return SE3(a.m_isometry * b.m_isometry); }

private:
    Eigen::Isometry3d m_isometry = Eigen::Isometry3d::Identity();
};

} // namespace dabfc

#endif // DABFC_SE3_H_
