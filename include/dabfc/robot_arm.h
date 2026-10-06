#ifndef DABFC_ROBOT_ARM_H_
#define DABFC_ROBOT_ARM_H_

#include <memory>
#include <optional>
#include <set>
#include <string>
#include <utility>
#include <vector>

#include <Eigen/Core>

#include "dabfc/collision.h"
#include "dabfc/se3.h"

namespace dabfc {

// One link of a kinematic tree
struct Link {
    std::string name;
    std::optional<int> parent; // index in the chain (none: the robot's base frame)
    std::optional<int> joint;  // index in q (none: fixed); several links may share a joint (mimic)
    double mult = 1;           // mimic multiplier; used for gripper fingers
    SE3 origin;
    Eigen::Vector3d axis = Eigen::Vector3d::UnitZ();
    std::string mesh;               // mesh base name (empty: no geometry)
    SE3 meshOrigin;                 // mesh frame in the link frame
    std::vector<std::string> parts; // mesh files <mesh>_<part>.stl, one per visual material
};

class RobotArm {
public:
    RobotArm(const RobotArm&) = delete;
    RobotArm& operator=(const RobotArm&) = delete;
    RobotArm(RobotArm&&) = delete;
    RobotArm& operator=(RobotArm&&) = delete;
    virtual ~RobotArm() = default;

    virtual Eigen::VectorXd home() const = 0; // a collision-free rest configuration
    virtual int toolLink() const = 0;         // index of the tool (flange) frame in links()

    const std::vector<Link>& links() const { return m_links; }
    int dof() const { return m_dof; } // size of q

    // World frame of every link, and of every link's mesh, at joint values q.
    // Throws std::invalid_argument if q.size() != dof()
    std::vector<SE3> fk(const Eigen::VectorXd& q, const SE3& base = {}) const;
    std::vector<SE3> meshPoses(const Eigen::VectorXd& q, const SE3& base = {}) const;

    // Collision queries, given this arm's meshPoses() (std::invalid_argument if the size is wrong).
    // Self-collisions exclude adjacent links and allowed pairs.
    bool isSelfCollisionFree(const std::vector<SE3>& meshPoses) const;
    bool touches(const std::vector<SE3>& meshPoses, const collision::Shape& shape,
                 const SE3& pose) const;
    bool touches(const std::vector<SE3>& meshPoses, const RobotArm& other,
                 const std::vector<SE3>& otherMeshPoses) const;

protected:
    // Loads each link's parts from <meshDir>/<mesh>_<part>.stl; adjacent links may always touch
    RobotArm(std::vector<Link> links, const std::string& meshDir);
    void allowSelfCollisions(const Eigen::VectorXd& q); // allows every pair of links touching at q

private:
    void expectMeshPoses(const std::vector<SE3>& meshPoses) const;
    bool touching(int i, int j, const std::vector<SE3>& meshPoses) const;

    std::vector<Link> m_links;
    int m_dof;
    std::vector<std::unique_ptr<const collision::Mesh>> m_geometry; // per link, may be null
    std::set<std::pair<int, int>> m_allowed;                        // link pairs (i, j), i < j
};

} // namespace dabfc

#endif // DABFC_ROBOT_ARM_H_
