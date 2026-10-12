#ifndef DABFC_SDF_H_
#define DABFC_SDF_H_

#include <memory>
#include <random>
#include <string>
#include <vector>

#include <CGAL/AABB_traits_3.h>
#include <CGAL/AABB_tree.h>
#include <CGAL/AABB_triangle_primitive_3.h>
#include <CGAL/Simple_cartesian.h>
#include <glm/glm.hpp>

#include "dabfc/se3.h"

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

struct Box {
    glm::vec3 min;
    glm::vec3 max;
};

// A closed triangle mesh, or the union of several; the sign is the generalized winding number
class SdfMesh : public Sdf {
public:
    explicit SdfMesh(const std::string& meshPath); // any Assimp format; throws std::runtime_error
    explicit SdfMesh(const std::vector<std::string>& meshPaths);

    float valueAt(glm::vec3 p) const override;
    glm::vec3 gradAt(glm::vec3 p) const override;

    Box bounds() const; // of the triangles
    double area() const;
    std::vector<glm::vec3> surfaceSamples(int count, std::mt19937& rng) const; // uniform by area

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

// Another SDF sampled on a regular grid over a box and interpolated trilinearly, for fast queries.
// Outside the box the value grows with the distance to the box
class SdfGrid : public Sdf {
public:
    SdfGrid(const Sdf& source, const Box& box, float spacing);
    SdfGrid(const Box& box, float spacing, std::vector<float> values); // e.g. read from a file

    float valueAt(glm::vec3 p) const override;
    glm::vec3 gradAt(glm::vec3 p) const override;
    const Box& box() const { return m_box; }
    float spacing() const { return m_spacing; }
    const std::vector<float>& values() const { return m_values; } // x fastest, then y, then z

private:
    struct Interpolation {
        float value;
        glm::vec3 grad; // of the trilinear interpolant inside the box
    };

    Interpolation at(glm::vec3 p) const;
    float sample(glm::ivec3 i) const { return m_values[(i.z * m_size.y + i.y) * m_size.x + i.x]; }

    Box m_box;
    float m_spacing;
    glm::ivec3 m_size; // samples per axis
    std::vector<float> m_values;
};

// The union of several SDFs, each placed by a pose: the smallest of their values. A part is
// skipped when the query point is farther from its bounding sphere than the best value so far
class SdfUnion : public Sdf {
public:
    struct Part {
        std::shared_ptr<const Sdf> sdf;
        SE3 pose;             // the part's frame in the union's frame
        Eigen::Vector3d center; // a sphere enclosing the part, in the part's frame
        double radius;
    };

    explicit SdfUnion(std::vector<Part> parts);

    float valueAt(glm::vec3 p) const override;
    glm::vec3 gradAt(glm::vec3 p) const override;
    Box bounds() const; // enclosing every part's sphere

private:
    struct Nearest {
        int part;
        float value;
        Eigen::Vector3d local; // the query point in the part's frame
    };

    Nearest nearest(glm::vec3 p) const;

    std::vector<Part> m_parts;
    std::vector<SE3> m_inversePoses;
    std::vector<Eigen::Vector3d> m_centers; // in the union's frame
};

} // namespace dabfc

#endif // DABFC_SDF_H_
