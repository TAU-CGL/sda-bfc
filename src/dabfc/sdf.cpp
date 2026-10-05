#include <algorithm>
#include <cmath>
#include <cstddef>
#include <stdexcept>

#include <thread>
#include <vector>

#include <assimp/Importer.hpp>
#include <assimp/postprocess.h>
#include <assimp/scene.h>


#include "dabfc/sdf.h"
using namespace dabfc;

void SdfMesh::build(const std::string& meshPath)
{
    load_triangle_mesh(meshPath);
    build_aabb_tree();
}

float SdfMesh::value_at(glm::vec3 p) const
{
    Point q(p.x, p.y, p.z);
    return std::sqrt(m_tree.squared_distance(q)) * sign(q);
}

glm::vec3 SdfMesh::grad_at(glm::vec3 p) const
{
    Point q(p.x, p.y, p.z);
    Point closest = m_tree.closest_point(q);
    glm::vec3 away(q.x() - closest.x(), q.y() - closest.y(), q.z() - closest.z());
    return glm::normalize(away) * sign(q);
}

void SdfMesh::load_triangle_mesh(const std::string& path)
{
    Assimp::Importer importer;
    const aiScene* scene = importer.ReadFile(path, aiProcess_Triangulate);
    if (!scene)
        throw std::runtime_error("SDF: cannot load " + path + ": " + importer.GetErrorString());
    m_triangles.clear();
    for (unsigned m = 0; m < scene->mNumMeshes; ++m) {
        const aiMesh& mesh = *scene->mMeshes[m];
        auto vertex = [&mesh](const aiFace& face, int k) {
            const aiVector3D& v = mesh.mVertices[face.mIndices[k]];
            return Point(v.x, v.y, v.z);
        };
        for (unsigned i = 0; i < mesh.mNumFaces; ++i) {
            const aiFace& face = mesh.mFaces[i];
            if (face.mNumIndices == 3)
                m_triangles.emplace_back(vertex(face, 0), vertex(face, 1), vertex(face, 2));
        }
    }
    if (m_triangles.empty()) throw std::runtime_error("SDF: no triangles in " + path);
}

void SdfMesh::build_aabb_tree()
{
    m_tree.rebuild(m_triangles.begin(), m_triangles.end());
    m_tree.accelerate_distance_queries();
}

static double solid_angle(const Triangle& t, const Point& q)
{
    Vector a = t[0] - q;
    Vector b = t[1] - q;
    Vector c = t[2] - q;
    double la = std::sqrt(a.squared_length());
    double lb = std::sqrt(b.squared_length());
    double lc = std::sqrt(c.squared_length());
    double denominator = la * lb * lc + (a * b) * lc + (b * c) * la + (c * a) * lb;
    return 2 * std::atan2(CGAL::determinant(a, b, c), denominator);
}

bool SdfMesh::is_inside(const Point& q) const
{
    double winding = 0;
    for (const Triangle& t : m_triangles) winding += solid_angle(t, q);
    return std::abs(winding) > 2 * glm::pi<double>();
}