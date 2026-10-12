#include "dabfc/contact_solver.h"

#include <cmath>
#include <filesystem>
#include <fstream>
#include <memory>
#include <numbers>
#include <random>

#include <gtest/gtest.h>

namespace dabfc {
namespace {

namespace fs = std::filesystem;

std::shared_ptr<const SdfMesh> unitCube()
{
    const fs::path path = fs::temp_directory_path() / "dabfc_solver_cube.obj";
    std::ofstream(path) << "v 0 0 0\nv 1 0 0\nv 1 1 0\nv 0 1 0\n"
                           "v 0 0 1\nv 1 0 1\nv 1 1 1\nv 0 1 1\n"
                           "f 1 4 3 2\nf 5 6 7 8\nf 1 2 6 5\n"
                           "f 2 3 7 6\nf 3 4 8 7\nf 4 1 5 8\n";
    return std::make_shared<SdfMesh>(path.string());
}

std::shared_ptr<const LinkShape> cubeShape(std::shared_ptr<const Sdf> sdf, const SdfMesh& mesh)
{
    std::mt19937 rng(0);
    LinkShape shape{.sdf = std::move(sdf), .surface = {}};
    for (const glm::vec3& p : mesh.surfaceSamples(600, rng)) shape.surface.emplace_back(p.x, p.y, p.z);
    return std::make_shared<const LinkShape>(std::move(shape));
}

TEST(SE3Test, Exp)
{
    const SE3 t = SE3::exp((Vector6d() << 1, 2, 3, 0, 0, 0).finished());
    EXPECT_TRUE(t.translation().isApprox(Eigen::Vector3d(1, 2, 3)));
    EXPECT_TRUE(t.rotation().isApprox(Eigen::Quaterniond::Identity()));
    const SE3 r = SE3::exp((Vector6d() << 0, 0, 0, 0, 0, std::numbers::pi / 2).finished());
    EXPECT_TRUE((r * Eigen::Vector3d::UnitX()).isApprox(Eigen::Vector3d::UnitY()));
    const Vector6d twist = (Vector6d() << 0.1, -0.2, 0.3, 0.4, -0.5, 0.6).finished();
    const SE3 identity = SE3::exp(twist) * SE3::exp(-twist);
    EXPECT_TRUE(identity.isometry().matrix().isApprox(Eigen::Matrix4d::Identity()));
}

TEST(SdfGridTest, MatchesTheSource)
{
    const std::shared_ptr<const SdfMesh> cube = unitCube();
    const SdfGrid grid(*cube, {.min = {-0.5f, -0.5f, -0.5f}, .max = {1.5f, 1.5f, 1.5f}}, 0.1f);
    for (const glm::vec3 p : {glm::vec3(0.5f, 0.5f, 0.3f), glm::vec3(0.5f, 0.2f, 0.5f),
                              glm::vec3(1.25f, 0.5f, 0.5f), glm::vec3(3, 0.5f, 0.5f)}) {
        EXPECT_NEAR(grid.valueAt(p), cube->valueAt(p), 1e-5);
        EXPECT_LT(glm::length(grid.gradAt(p) - cube->gradAt(p)), 1e-4);
    }
}

// Unit cubes on both arms; with arm 2's base at `truth` the pairs touch face to face at the
// planes x = 1, z = 1 and z = 1.2
class ContactSolverTest : public testing::Test {
protected:
    ContactSolverTest()
    {
        const std::shared_ptr<const SdfMesh> cube = unitCube();
        const std::shared_ptr<const LinkShape> shape = cubeShape(cube, *cube);
        contacts = {{.shape1 = shape, .pose1 = SE3({0, 0, 0}), .shape2 = shape, .pose2 = SE3({0, 0, 0})},
                    {.shape1 = shape, .pose1 = SE3({0, -1, 0}), .shape2 = shape, .pose2 = SE3({-0.5, -1.2, 0.8})},
                    {.shape1 = shape, .pose1 = SE3({0, 0, 1.2}), .shape2 = shape, .pose2 = SE3({-0.6, 0, 0})}};
    }

    const SE3 truth{{1, 0.1, 0.2}};
    std::vector<TouchingLinks> contacts;
};

TEST_F(ContactSolverTest, JacobianMatchesFiniteDifferences)
{
    const ContactSolver solver(contacts);
    const SE3 X({1.3, 0.2, -0.1}, {0.1, 0.2, 0.3});
    constexpr double h = 1e-4; // the SDF is single precision
    for (int t = 0; t < std::ssize(contacts); ++t) {
        const ContactSolver::Residual r = solver.residual(X, t);
        for (int k = 0; k < 6; ++k) {
            const Vector6d e = Vector6d::Unit(k) * h;
            const double fd = (solver.residual(X * SE3::exp(e), t).value -
                               solver.residual(X * SE3::exp(-e), t).value) /
                              (2 * h);
            EXPECT_NEAR(r.jacobian[k], fd, 5e-3) << "pair " << t << " entry " << k;
        }
    }
}

TEST_F(ContactSolverTest, RecoversATouchingPose)
{
    const ContactSolver solver(contacts);
    EXPECT_LT(solver.cost(truth), 1e-10);
    const SE3 X = solver.solve(truth * SE3::exp((Vector6d() << 0.03, -0.02, 0.02, 0.02, -0.03, 0.01).finished()));
    EXPECT_LT(solver.cost(X), 1e-8);
}

} // namespace
} // namespace dabfc
