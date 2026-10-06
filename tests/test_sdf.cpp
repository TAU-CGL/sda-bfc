#include "dabfc/sdf.h"

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <random>
#include <stdexcept>
#include <vector>

#include <gtest/gtest.h>

namespace dabfc {
namespace {

namespace fs = std::filesystem;

// SDF of the unit cube [0,1]^3 against the analytic box distance.
TEST(SdfMeshTest, UnitCube)
{
    fs::path path = fs::temp_directory_path() / "dabfc_cube.obj";
    std::ofstream(path) << "v 0 0 0\nv 1 0 0\nv 1 1 0\nv 0 1 0\n"
                           "v 0 0 1\nv 1 0 1\nv 1 1 1\nv 0 1 1\n"
                           "f 1 4 3 2\nf 5 6 7 8\nf 1 2 6 5\n"
                           "f 2 3 7 6\nf 3 4 8 7\nf 4 1 5 8\n";
    const SdfMesh sdf(path.string());
    EXPECT_NEAR(sdf.valueAt({0.5f, 0.5f, 0.5f}), -0.5f, 1e-6);
    EXPECT_NEAR(sdf.valueAt({0.5f, 0.2f, 0.5f}), -0.2f, 1e-6);
    EXPECT_NEAR(sdf.valueAt({2, 0.5f, 0.5f}), 1, 1e-6);
    EXPECT_NEAR(sdf.valueAt({2, 2, 0.5f}), std::sqrt(2.f), 1e-6);
    EXPECT_EQ(sdf.gradAt({0.5f, 0.2f, 0.5f}), glm::vec3(0, -1, 0));
    EXPECT_EQ(sdf.gradAt({2, 0.5f, 0.5f}), glm::vec3(1, 0, 0));
}

TEST(SdfMeshTest, MissingFileThrows)
{
    EXPECT_THROW(SdfMesh("does_not_exist.stl"), std::runtime_error);
}

// On each UR5e visual mesh: p - F(p) * grad F(p) lands on the surface, the
// gradient is unit length, and it agrees with central finite differences.
class SdfMeshFileTest : public testing::TestWithParam<fs::path> {};

TEST_P(SdfMeshFileTest, Consistency)
{
    constexpr int num_samples = 50;
    constexpr float h = 1e-4f;
    const SdfMesh sdf(GetParam().string());
    EXPECT_GT(sdf.valueAt({10, 10, 10}), 0);
    std::mt19937 rng(0);
    std::uniform_real_distribution<float> uniform(-0.2f, 0.2f);
    int fd_mismatches = 0;
    for (int s = 0; s < num_samples; ++s) {
        glm::vec3 p(uniform(rng), uniform(rng), uniform(rng));
        float value = sdf.valueAt(p);
        glm::vec3 grad = sdf.gradAt(p);
        EXPECT_NEAR(glm::length(grad), 1, 1e-4);
        EXPECT_NEAR(sdf.valueAt(p - value * grad), 0, 1e-5);
        glm::vec3 fd;
        for (int i = 0; i < 3; ++i) {
            glm::vec3 e(0);
            e[i] = h;
            fd[i] = (sdf.valueAt(p + e) - sdf.valueAt(p - e)) / (2 * h);
        }
        fd_mismatches += glm::length(fd - grad) > 0.05f; // medial axis / open-mesh sign flips
    }
    EXPECT_LT(fd_mismatches, num_samples * 5 / 100);
}

std::vector<fs::path> mesh_paths()
{
    std::vector<fs::path> paths(fs::directory_iterator(MESH_DIR), {});
    std::sort(paths.begin(), paths.end());
    return paths;
}

INSTANTIATE_TEST_SUITE_P(UR5eVisual, SdfMeshFileTest, testing::ValuesIn(mesh_paths()),
                         [](const testing::TestParamInfo<fs::path>& info) {
                             return info.param.stem().string();
                         });

} // namespace
} // namespace dabfc
