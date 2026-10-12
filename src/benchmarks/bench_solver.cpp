// How well ContactSolver recovers arm 2's base from the contact configurations of a benchmark
// file (make_benchmark), given which links touch. Every link's SDF is its visual mesh sampled on a
// grid; every link's surface gets the same number of random samples.
// Usage: bench_solver [benchmark = benchmark/contacts.json] [samples = 2000] [grid spacing m = 0.005]
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <map>
#include <memory>
#include <numbers>
#include <random>
#include <string>

#include <nlohmann/json.hpp>

#include "dabfc/contact_solver.h"
#include "dabfc/ur5e.h"

using namespace dabfc;
using nlohmann::json;

namespace {

constexpr float gridMargin = 0.05f; // m around each link's mesh

template <typename F> double seconds(F f)
{
    const auto start = std::chrono::steady_clock::now();
    f();
    return std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
}

Eigen::VectorXd vector(const json& values)
{
    Eigen::VectorXd v(values.size());
    for (int k = 0; k < v.size(); ++k) v[k] = values[k];
    return v;
}

SE3 base2(const json& placement)
{
    const Eigen::VectorXd t = vector(placement["base2"]["translation"]);
    const Eigen::VectorXd q = vector(placement["base2"]["quaternion_wxyz"]);
    return SE3(t, Eigen::Quaterniond(q[0], q[1], q[2], q[3]));
}

// Every meshed link's shape, built on first use
class Shapes {
public:
    Shapes(const RobotArm& arm, int samples, float spacing)
        : m_arm(arm), m_samples(samples), m_spacing(spacing)
    {
    }

    std::shared_ptr<const LinkShape> operator[](int link)
    {
        std::shared_ptr<const LinkShape>& shape = m_shapes[link];
        if (shape) return shape;
        const Link& l = m_arm.links()[link];
        std::vector<std::string> paths;
        for (const std::string& part : l.parts)
            paths.push_back(std::string(MESH_DIR) + "/" + l.mesh + "_" + part + ".stl");
        const SdfMesh mesh(paths);
        Box box = mesh.bounds();
        box.min -= gridMargin;
        box.max += gridMargin;
        LinkShape built;
        const double time = seconds(
            [&] { built.sdf = std::make_shared<const SdfGrid>(mesh, box, m_spacing); });
        std::mt19937 rng(0);
        for (const glm::vec3& p : mesh.surfaceSamples(m_samples, rng))
            built.surface.emplace_back(p.x, p.y, p.z);
        std::fprintf(stderr, "%s grid built in %.1f s\n", l.name.c_str(), time);
        shape = std::make_shared<const LinkShape>(std::move(built));
        return shape;
    }

private:
    const RobotArm& m_arm;
    int m_samples;
    float m_spacing;
    std::map<int, std::shared_ptr<const LinkShape>> m_shapes;
};

int linkIndex(const RobotArm& arm, const std::string& name)
{
    for (int i = 0; i < std::ssize(arm.links()); ++i)
        if (arm.links()[i].name == name) return i;
    throw std::runtime_error("unknown link " + name);
}

void report(const char* start, const ContactSolver& solver, const SE3& X0, const SE3& truth)
{
    SE3 X;
    const double time = seconds([&] { X = solver.solve(X0); });
    const double translation = (X.translation() - truth.translation()).norm();
    const double rotation =
        Eigen::AngleAxisd(truth.rotation().inverse() * X.rotation()).angle() * 180 / std::numbers::pi;
    std::printf("  %-9s %10.2f %10.3f %10.2e %8.2f\n", start, 1e3 * translation, rotation,
                solver.cost(X), time);
}

} // namespace

int main(int argc, char** argv)
{
    const std::string path = argc > 1 ? argv[1] : std::string(PROJECT_DIR) + "/benchmark/contacts.json";
    const int samples = argc > 2 ? std::atoi(argv[2]) : 2000;
    const float spacing = argc > 3 ? static_cast<float>(std::atof(argv[3])) : 0.005f;
    const json benchmark = json::parse(std::ifstream(path));

    const UR5e ur5e(MESH_DIR);
    Shapes shapes(ur5e, samples, spacing);
    std::mt19937 rng(0);
    std::normal_distribution<double> gauss;
    for (const json& placement : benchmark["placements"]) {
        std::vector<TouchingLinks> contacts;
        for (const json& contact : placement["contacts"]) {
            const std::vector<SE3> poses1 = ur5e.meshPoses(vector(contact["q1"]));
            const std::vector<SE3> poses2 = ur5e.meshPoses(vector(contact["q2"]));
            for (const json& pair : contact["pairs"]) {
                const int link1 = linkIndex(ur5e, pair["link1"]);
                const int link2 = linkIndex(ur5e, pair["link2"]);
                contacts.push_back({.shape1 = shapes[link1],
                                    .pose1 = poses1[link1],
                                    .shape2 = shapes[link2],
                                    .pose2 = poses2[link2]});
            }
        }
        const ContactSolver solver(contacts);
        const SE3 truth = base2(placement);
        std::printf("placement %d: %zu touching pairs, arm 2 %.2f m away, cost at the truth %.2e\n",
                    placement["seed"].get<int>(), contacts.size(), truth.translation().norm(),
                    solver.cost(truth));
        std::printf("  %-9s %10s %10s %10s %8s\n", "start", "pos mm", "rot deg", "cost", "time s");
        Vector6d noise; // 5 cm, 5 degrees
        noise << 0.05 * Eigen::Vector3d(gauss(rng), gauss(rng), gauss(rng)).normalized(),
            5 * std::numbers::pi / 180 * Eigen::Vector3d(gauss(rng), gauss(rng), gauss(rng)).normalized();
        report("truth", solver, truth, truth);
        report("near", solver, truth * SE3::exp(noise), truth);
        report("identity", solver, SE3{}, truth);
    }
}
