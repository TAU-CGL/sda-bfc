#ifndef DABFC_COLLISION_H_
#define DABFC_COLLISION_H_

#include <memory>
#include <optional>
#include <string>
#include <vector>

#include <Eigen/Core>
#include <fcl/fcl.h>

#include "dabfc/se3.h"

namespace dabfc::collision {

using Shape = fcl::CollisionGeometryd;
using Mesh = fcl::BVHModel<fcl::OBBRSSd>; // triangle mesh with a bounding volume hierarchy

// A world point where the two shapes, at the given poses, touch or overlap (exact test)
std::optional<Eigen::Vector3d> contact(const Shape& a, const SE3& poseA, const Shape& b,
                                       const SE3& poseB);

// One mesh made of the triangles of binary STL files (metres); throws std::runtime_error
std::unique_ptr<Mesh> loadStl(const std::vector<std::string>& paths);

} // namespace dabfc::collision

#endif // DABFC_COLLISION_H_
