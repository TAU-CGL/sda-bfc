#ifndef DABFC_SDF_H_
#define DABFC_SDF_H_

#include <string>
#include <vector>

#include <CGAL/AABB_traits_3.h>
#include <CGAL/AABB_tree.h>
#include <CGAL/AABB_triangle_primitive_3.h>
#include <CGAL/Simple_cartesian.h>
#include <glm/glm.hpp>

namespace dabfc {

// Signed distance field: negative inside
class Sdf {
public:
    Sdf() = default;
    Sdf(const Sdf&) = delete;
    Sdf& operator=(const Sdf&) = delete;
    Sdf(Sdf&&) = delete;
    Sdf& operator=(Sdf&&) = delete;
    virtual ~Sdf() = default;

    virtual float valueAt(glm::vec3 p) const = 0;
    virtual glm::vec3 gradAt(glm::vec3 p) const = 0; // unit vector away from the closest surface
};

// A closed triangle mesh; the sign is the generalized winding number
class SdfMesh : public Sdf {
public:
    explicit SdfMesh(const std::string& meshPath); // any Assimp format; throws std::runtime_error

    float valueAt(glm::vec3 p) const override;
    glm::vec3 gradAt(glm::vec3 p) const override;

private:
    using Kernel = CGAL::Simple_cartesian<double>;
    using Point = Kernel::Point_3;
    using Triangle = Kernel::Triangle_3;
    using Primitive =
        CGAL::AABB_triangle_primitive_3<Kernel, std::vector<Triangle>::const_iterator>;
    using Tree = CGAL::AABB_tree<CGAL::AABB_traits_3<Kernel, Primitive>>;

    bool isInside(const Point& q) const;

    std::vector<Triangle> m_triangles;
    Tree m_tree;
};

} // namespace dabfc

#endif // DABFC_SDF_H_
