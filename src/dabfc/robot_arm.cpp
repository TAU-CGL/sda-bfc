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

std::vector<RobotArm::LinkContact> RobotArm::selfContacts(const std::vector<SE3>& meshPoses) const
{
    expectMeshPoses(meshPoses);
    std::vector<LinkContact> contacts;
    for (int i = 0; i < std::ssize(m_links); ++i) {
        if (!m_geometry[i]) continue;
        for (int j = i + 1; j < std::ssize(m_links); ++j) {
            if (m_allowed.contains({i, j})) continue;
            if (const auto point = contact(j, meshPoses[j], *m_geometry[i], meshPoses[i]))
                contacts.push_back({.link = i, .otherLink = j, .point = *point});
        }
    }
    return contacts;
}

std::vector<RobotArm::LinkContact> RobotArm::contacts(const std::vector<SE3>& meshPoses,
                                                      const collision::Shape& shape,
                                                      const SE3& pose) const
{
    expectMeshPoses(meshPoses);
    std::vector<LinkContact> contacts;
    for (int i = 0; i < std::ssize(m_links); ++i)
        if (const auto point = contact(i, meshPoses[i], shape, pose))
            contacts.push_back({.link = i, .point = *point});
    return contacts;
}

std::vector<RobotArm::LinkContact> RobotArm::contacts(const std::vector<SE3>& meshPoses,
                                                      const RobotArm& other,
                                                      const std::vector<SE3>& otherMeshPoses) const
{
    expectMeshPoses(meshPoses);
    other.expectMeshPoses(otherMeshPoses);
    std::vector<LinkContact> contacts;
    for (int i = 0; i < std::ssize(m_links); ++i) {
        if (!m_geometry[i]) continue;
        for (int j = 0; j < std::ssize(other.m_links); ++j)
            if (const auto point =
                    other.contact(j, otherMeshPoses[j], *m_geometry[i], meshPoses[i]))
                contacts.push_back({.link = i, .otherLink = j, .point = *point});
    }
    return contacts;
}

void RobotArm::allowSelfCollisions(const Eigen::VectorXd& q)
{
    for (const LinkContact& c : selfContacts(meshPoses(q))) m_allowed.emplace(c.link, *c.otherLink);
}

void RobotArm::expectMeshPoses(const std::vector<SE3>& meshPoses) const
{
    if (meshPoses.size() != m_links.size())
        throw std::invalid_argument("RobotArm: expected one mesh pose per link");
}

std::optional<Eigen::Vector3d> RobotArm::contact(int link, const SE3& meshPose,
                                                 const collision::Shape& shape,
                                                 const SE3& pose) const
{
    if (!m_geometry[link]) return std::nullopt;
    return collision::contact(*m_geometry[link], meshPose, shape, pose);
}

} // namespace dabfc
