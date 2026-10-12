#include "dabfc/sdf.h"

#include <algorithm>
#include <cmath>
#include <numbers>
#include <span>
#include <stdexcept>

#include <assimp/Importer.hpp>
#include <assimp/postprocess.h>
#include <assimp/scene.h>

namespace dabfc {

SdfMesh::SdfMesh(const std::string& meshPath) : SdfMesh(std::vector{meshPath}) {}

SdfMesh::SdfMesh(const std::vector<std::string>& meshPaths)
{
    for (const std::string& meshPath : meshPaths) {
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
    }
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
    const auto [closest, triangle] = m_tree.closest_point_and_primitive(q);
    Kernel::Vector_3 away = q - closest;
    if (away.squared_length() < 1e-24) { // on the surface: the triangle's outward normal
        away = triangle->supporting_plane().orthogonal_vector();
        return glm::normalize(glm::vec3(away.x(), away.y(), away.z()));
    }
    const glm::vec3 unit = glm::normalize(glm::vec3(away.x(), away.y(), away.z()));
    return isInside(q) ? -unit : unit;
}

Box SdfMesh::bounds() const
{
    const CGAL::Bbox_3 b = m_tree.bbox();
    return {.min = {b.xmin(), b.ymin(), b.zmin()}, .max = {b.xmax(), b.ymax(), b.zmax()}};
}

std::vector<glm::vec3> SdfMesh::surfaceSamples(int count, std::mt19937& rng) const
{
    std::vector<double> areas(m_triangles.size());
    std::ranges::transform(m_triangles, areas.begin(),
                           [](const Triangle& t) { return std::sqrt(t.squared_area()); });
    std::discrete_distribution<std::size_t> triangle(areas.begin(), areas.end());
    std::uniform_real_distribution<double> uniform(0, 1);
    std::vector<glm::vec3> samples(count);
    for (glm::vec3& sample : samples) {
        const Triangle& t = m_triangles[triangle(rng)];
        const double r = std::sqrt(uniform(rng)); // uniform over the triangle
        const double s = uniform(rng);
        const Point p = CGAL::ORIGIN + (1 - r) * (t[0] - CGAL::ORIGIN) +
                        r * (1 - s) * (t[1] - CGAL::ORIGIN) + r * s * (t[2] - CGAL::ORIGIN);
        sample = {p.x(), p.y(), p.z()};
    }
    return samples;
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

SdfGrid::SdfGrid(const Sdf& source, const Box& box, float spacing)
    : m_box(box), m_spacing(spacing), m_size(glm::ceil((box.max - box.min) / spacing) + 1.f)
{
    m_box.max = m_box.min + glm::vec3(m_size - 1) * spacing;
    m_values.resize(static_cast<std::size_t>(m_size.x) * m_size.y * m_size.z);
#pragma omp parallel for schedule(dynamic, 64)
    for (int k = 0; k < m_size.z; ++k)
        for (int j = 0; j < m_size.y; ++j)
            for (int i = 0; i < m_size.x; ++i)
                m_values[(k * m_size.y + j) * m_size.x + i] =
                    source.valueAt(m_box.min + glm::vec3(i, j, k) * spacing);
}

float SdfGrid::valueAt(glm::vec3 p) const
{
    const glm::vec3 inside = glm::clamp(p, m_box.min, m_box.max);
    return at(inside).value + glm::length(p - inside);
}

glm::vec3 SdfGrid::gradAt(glm::vec3 p) const
{
    const glm::vec3 inside = glm::clamp(p, m_box.min, m_box.max);
    const glm::vec3 grad = p == inside ? at(inside).grad : p - inside;
    const float length = glm::length(grad);
    return length > 0 ? grad / length : grad;
}

SdfGrid::Interpolation SdfGrid::at(glm::vec3 p) const
{
    const glm::vec3 u = (p - m_box.min) / m_spacing;
    const glm::ivec3 cell = glm::min(glm::ivec3(glm::floor(u)), m_size - 2);
    const glm::vec3 f = u - glm::vec3(cell);
    Interpolation result{0, glm::vec3(0)};
    for (int c = 0; c < 8; ++c) {
        const glm::ivec3 corner(c & 1, (c >> 1) & 1, (c >> 2) & 1);
        const glm::vec3 w = glm::mix(1.f - f, f, glm::vec3(corner)); // weight per axis
        const float value = sample(cell + corner);
        result.value += w.x * w.y * w.z * value;
        result.grad += (2.f * glm::vec3(corner) - 1.f) * glm::vec3(w.y * w.z, w.x * w.z, w.x * w.y) *
                       value;
    }
    result.grad /= m_spacing;
    return result;
}

} // namespace dabfc
