#ifndef DABFC_SDF_HPP
#define DABFC_SDF_HPP

#include <string>

#include <glm/glm.hpp>
#include <glm/gtc/constants.hpp>

#include <CGAL/AABB_traits_3.h>
#include <CGAL/AABB_tree.h>
#include <CGAL/AABB_triangle_primitive_3.h>
#include <CGAL/Simple_cartesian.h>
using Kernel = CGAL::Simple_cartesian<double>;
using Point = Kernel::Point_3;
using Vector = Kernel::Vector_3;
using Triangle = Kernel::Triangle_3;
using Primitive = CGAL::AABB_triangle_primitive_3<Kernel, std::vector<Triangle>::const_iterator>;
using AABB_tree = CGAL::AABB_tree<CGAL::AABB_traits_3<Kernel, Primitive>>;


namespace dabfc {

class Sdf {
public:
    virtual float value_at(glm::vec3 p) const = 0;
    virtual float value_at(float x, float y, float z) const { return value_at(glm::vec3(x, y, z)); }
    virtual glm::vec3 grad_at(glm::vec3 p) const = 0;
    virtual glm::vec3 grad_at(float x, float y, float z) const { return grad_at(glm::vec3(x, y, z)); }
};

class SdfMesh : public Sdf {
public:
    void build(const std::string& meshPath);

    virtual float value_at(glm::vec3 q) const;
    virtual glm::vec3 grad_at(glm::vec3 q) const;

private:
    void load_triangle_mesh(const std::string& path);
    void build_aabb_tree();

    float sign(const Point& q) const { return is_inside(q) ? -1.f : 1.f; }
    bool is_inside(const Point& q) const;

    std::vector<Triangle> m_triangles;
    AABB_tree m_tree;
};

} // namespace dabfc

#endif // DABFC_SDF_HPP
