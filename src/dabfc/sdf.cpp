#include "dabfc/sdf.h"

#include <cmath>
#include <numbers>
#include <span>
#include <stdexcept>

#include <assimp/Importer.hpp>
#include <assimp/postprocess.h>
#include <assimp/scene.h>

namespace dabfc {

SdfMesh::SdfMesh(const std::string& meshPath)
{
    Assimp::Importer importer;
    const aiScene* scene = importer.ReadFile(meshPath, aiProcess_Triangulate);
    if (!scene)
        throw std::runtime_error("SdfMesh: cannot load " + meshPath + ": " +
                                 importer.GetErrorString());
    for (const aiMesh* mesh : std::span(scene->mMeshes, scene->mNumMeshes)) {
        const std::span vertices(mesh->mVertices, mesh->mNumVertices);
        for (const aiFace& face : std::span(mesh->mFaces, mesh->mNumFaces)) {
            const std::span indices(face.mIndices, face.mNumIndices);
            if (indices.size() != 3) continue;
            auto vertex = [&](int k) {
                const aiVector3D& v = vertices[indices[k]];
                return Point(v.x, v.y, v.z);
            };
            m_triangles.emplace_back(vertex(0), vertex(1), vertex(2));
        }
    }
    if (m_triangles.empty()) throw std::runtime_error("SdfMesh: no triangles in " + meshPath);
    m_tree.rebuild(m_triangles.begin(), m_triangles.end());
    m_tree.accelerate_distance_queries();
}

float SdfMesh::valueAt(glm::vec3 p) const
{
    const Point q(p.x, p.y, p.z);
    const double distance = std::sqrt(m_tree.squared_distance(q));
    return static_cast<float>(isInside(q) ? -distance : distance);
}

glm::vec3 SdfMesh::gradAt(glm::vec3 p) const
{
    const Point q(p.x, p.y, p.z);
    const Point closest = m_tree.closest_point(q);
    const glm::vec3 away =
        glm::normalize(glm::vec3(q.x() - closest.x(), q.y() - closest.y(), q.z() - closest.z()));
    return isInside(q) ? -away : away;
}

bool SdfMesh::isInside(const Point& q) const
{
    // Solid angle of the whole mesh seen from q (Van Oosterom & Strackee): 4 pi inside, 0 outside
    double solidAngle = 0;
    for (const Triangle& t : m_triangles) {
        const Kernel::Vector_3 a = t[0] - q;
        const Kernel::Vector_3 b = t[1] - q;
        const Kernel::Vector_3 c = t[2] - q;
        const double la = std::sqrt(a.squared_length());
        const double lb = std::sqrt(b.squared_length());
        const double lc = std::sqrt(c.squared_length());
        const double denominator = la * lb * lc + (a * b) * lc + (b * c) * la + (c * a) * lb;
        solidAngle += 2 * std::atan2(CGAL::determinant(a, b, c), denominator);
    }
    return std::abs(solidAngle) > 2 * std::numbers::pi;
}

} // namespace dabfc
