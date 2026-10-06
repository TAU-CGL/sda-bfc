#include "dabfc/collision.h"

#include <array>
#include <cstdint>
#include <fstream>
#include <istream>
#include <stdexcept>
#include <type_traits>

namespace dabfc::collision {

namespace {

template <class T> std::istream& readBytes(std::istream& in, T& value)
{
    static_assert(std::is_trivially_copyable_v<T>);
    return in.read(reinterpret_cast<char*>(&value), sizeof value);
}

} // namespace

bool touches(const Shape& a, const SE3& poseA, const Shape& b, const SE3& poseB)
{
    const fcl::CollisionRequestd request; // stops at the first contact
    fcl::CollisionResultd result;
    return fcl::collide(&a, poseA.isometry(), &b, poseB.isometry(), request, result) > 0;
}

std::unique_ptr<Mesh> loadStl(const std::vector<std::string>& paths)
{
    std::vector<fcl::Vector3d> vertices;
    std::vector<fcl::Triangle> triangles;
    for (const std::string& path : paths) {
        std::ifstream file(path, std::ios::binary);
        std::uint32_t n = 0; // binary STL: 80-byte header, triangle count, 50 bytes per triangle
        if (!readBytes(file.seekg(80), n))
            throw std::runtime_error("collision::loadStl: cannot read " + path);
        for (std::uint32_t i = 0; i < n; ++i) {
            std::array<float, 12> data{}; // normal, then 3 vertices
            readBytes(file, data).ignore(2);
            for (int k = 1; k <= 3; ++k)
                vertices.emplace_back(data[3 * k], data[3 * k + 1], data[3 * k + 2]);
            triangles.emplace_back(vertices.size() - 3, vertices.size() - 2, vertices.size() - 1);
        }
        if (!file) throw std::runtime_error("collision::loadStl: truncated file " + path);
    }
    auto mesh = std::make_unique<Mesh>();
    mesh->beginModel();
    mesh->addSubModel(vertices, triangles);
    mesh->endModel();
    return mesh;
}

} // namespace dabfc::collision
