#ifndef DABFC_SDF_H_
#define DABFC_SDF_H_

#include <string>

#include <glm/glm.hpp>
#include <glm/gtc/constants.hpp>

#include "dabfc/cgal.h"

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

#endif // DABFC_SDF_H_
