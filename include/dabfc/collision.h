#ifndef DABFC_COLLISION_H_
#define DABFC_COLLISION_H_

#include <memory>
#include <string>
#include <vector>

#include <fcl/fcl.h>

#include "dabfc/se3.h"

namespace dabfc::collision {

using Shape = fcl::CollisionGeometryd;
using Mesh = fcl::BVHModel<fcl::OBBRSSd>; // triangle mesh with a bounding volume hierarchy

// Exact test: do the two shapes, at the given poses, touch or overlap?
bool touches(const Shape& a, const SE3& poseA, const Shape& b, const SE3& poseB);

// One mesh made of the triangles of binary STL files (metres); throws std::runtime_error
std::unique_ptr<Mesh> loadStl(const std::vector<std::string>& paths);

} // namespace dabfc::collision

#endif // DABFC_COLLISION_H_
