#ifndef DABFC_SE3_H_
#define DABFC_SE3_H_

#include <cmath>

#include <Eigen/Geometry>

namespace dabfc {

using Vector6d = Eigen::Vector<double, 6>;

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

    // Exponential map of a twist (v, w): the pose reached by moving along v while turning about w
    static SE3 exp(const Vector6d& twist)
    {
        const Eigen::Vector3d v = twist.head<3>();
        const Eigen::Vector3d w = twist.tail<3>();
        const double theta = w.norm();
        const Eigen::Matrix3d W = skew(w);
        const Eigen::Matrix3d W2 = W * W;
        double b = 0.5;     // (1 - cos theta) / theta^2
        double c = 1. / 6;  // (theta - sin theta) / theta^3
        if (theta > 1e-9) {
            b = (1 - std::cos(theta)) / (theta * theta);
            c = (theta - std::sin(theta)) / (theta * theta * theta);
        }
        const Eigen::Matrix3d V = Eigen::Matrix3d::Identity() + b * W + c * W2;
        return SE3(V * v, Eigen::Quaterniond(Eigen::AngleAxisd(theta, theta > 1e-9 ? w / theta : w)));
    }

    SE3 inverse() const { return SE3(m_isometry.inverse()); }
    Eigen::Vector3d translation() const { return m_isometry.translation(); }
    Eigen::Quaterniond rotation() const { return Eigen::Quaterniond(m_isometry.linear()); }
    const Eigen::Isometry3d& isometry() const { return m_isometry; } // e.g. as an fcl::Transform3d

    // Composition: if b is a frame given in frame a, a * b is that frame given in a's parent frame
    friend SE3 operator*(const SE3& a, const SE3& b) { return SE3(a.m_isometry * b.m_isometry); }
    // A point given in this frame, in the parent frame
    friend Eigen::Vector3d operator*(const SE3& a, const Eigen::Vector3d& p)
    {
        return a.m_isometry * p;
    }

private:
    static Eigen::Matrix3d skew(const Eigen::Vector3d& w)
    {
        Eigen::Matrix3d S;
        S << 0, -w.z(), w.y(), w.z(), 0, -w.x(), -w.y(), w.x(), 0;
        return S;
    }

    Eigen::Isometry3d m_isometry = Eigen::Isometry3d::Identity();
};

} // namespace dabfc

#endif // DABFC_SE3_H_
