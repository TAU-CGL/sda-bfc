#include "dabfc/robot_arm.h"

#include <algorithm>
#include <iterator>
#include <stdexcept>

namespace dabfc {

namespace {

int jointCount(const std::vector<Link>& links)
{
    int n = 0;
    for (const Link& link : links)
        if (link.joint) n = std::max(n, *link.joint + 1);
    return n;
}

} // namespace

RobotArm::RobotArm(std::vector<Link> links, const std::string& meshDir)
    : m_links(std::move(links)), m_dof(jointCount(m_links)), m_geometry(m_links.size())
{
    for (int i = 0; i < std::ssize(m_links); ++i) {
        const Link& link = m_links[i];
        if (link.mesh.empty()) continue;
        std::vector<std::string> paths;
        for (const std::string& part : link.parts)
            paths.push_back(meshDir + "/" + link.mesh + "_" + part + ".stl");
        m_geometry[i] = collision::loadStl(paths);
        std::optional<int> p = link.parent; // nearest ancestor with a mesh: touches by construction
        while (p && m_links[*p].mesh.empty()) p = m_links[*p].parent;
        if (p) m_allowed.emplace(*p, i);
    }
}

std::vector<SE3> RobotArm::fk(const Eigen::VectorXd& q, const SE3& base) const
{
    if (q.size() != m_dof) throw std::invalid_argument("RobotArm::fk: q.size() != dof()");
    std::vector<SE3> frames(m_links.size());
    for (int i = 0; i < std::ssize(m_links); ++i) {
        const Link& link = m_links[i];
        const double angle = link.joint ? link.mult * q[*link.joint] : 0.0;
        const SE3 jointRotation(Eigen::Vector3d::Zero(),
                                Eigen::Quaterniond(Eigen::AngleAxisd(angle, link.axis)));
        frames[i] = (link.parent ? frames[*link.parent] : base) * link.origin * jointRotation;
    }
    return frames;
}

std::vector<SE3> RobotArm::meshPoses(const Eigen::VectorXd& q, const SE3& base) const
{
    std::vector<SE3> poses = fk(q, base);
    for (int i = 0; i < std::ssize(m_links); ++i) poses[i] = poses[i] * m_links[i].meshOrigin;
    return poses;
}

bool RobotArm::isSelfCollisionFree(const std::vector<SE3>& meshPoses) const
{
    expectMeshPoses(meshPoses);
    for (int i = 0; i < std::ssize(m_links); ++i)
        for (int j = i + 1; j < std::ssize(m_links); ++j)
            if (!m_allowed.contains({i, j}) && touching(i, j, meshPoses)) return false;
    return true;
}

void RobotArm::allowSelfCollisions(const Eigen::VectorXd& q)
{
    const std::vector<SE3> poses = meshPoses(q);
    for (int i = 0; i < std::ssize(m_links); ++i)
        for (int j = i + 1; j < std::ssize(m_links); ++j)
            if (touching(i, j, poses)) m_allowed.emplace(i, j);
}

bool RobotArm::touches(const std::vector<SE3>& meshPoses, const collision::Shape& shape,
                       const SE3& pose) const
{
    expectMeshPoses(meshPoses);
    for (int i = 0; i < std::ssize(m_links); ++i)
        if (m_geometry[i] && collision::touches(*m_geometry[i], meshPoses[i], shape, pose))
            return true;
    return false;
}

bool RobotArm::touches(const std::vector<SE3>& meshPoses, const RobotArm& other,
                       const std::vector<SE3>& otherMeshPoses) const
{
    expectMeshPoses(meshPoses);
    for (int i = 0; i < std::ssize(m_links); ++i)
        if (m_geometry[i] && other.touches(otherMeshPoses, *m_geometry[i], meshPoses[i]))
            return true;
    return false;
}

void RobotArm::expectMeshPoses(const std::vector<SE3>& meshPoses) const
{
    if (meshPoses.size() != m_links.size())
        throw std::invalid_argument("RobotArm: expected one mesh pose per link");
}

bool RobotArm::touching(int i, int j, const std::vector<SE3>& meshPoses) const
{
    return m_geometry[i] && m_geometry[j] &&
           collision::touches(*m_geometry[i], meshPoses[i], *m_geometry[j], meshPoses[j]);
}

} // namespace dabfc
