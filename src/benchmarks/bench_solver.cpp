// How well ContactSolver recovers arm 2's base from the contact configurations of a benchmark
// file (make_benchmark). Three solvers per placement:
//   known links   the touching link pair of every contact is given (one body per arm)
//   whole arms    every arm at its configuration is one body: the union of its link SDFs, sampled
//                 on a coarse grid. Cheap enough to scan (x, y, z, yaw) over the placement range,
//                 then refine the best scan poses
//   per link      the exact link SDFs, every link a candidate; polishes the whole-arm result
// The base does not move and the shoulder only turns about its own symmetry axis, so their
// surfaces stay put in the arm's base frame: they are fixed bodies, kept apart from the moving
// links (two fixed bodies touching would explain every contact at once, see Contact).
// The touching links in the file are used only to grade the result.
// Usage: bench_solver [benchmark = benchmark/contacts.json] [samples per link = 2000]
//                     [link grid spacing m = 0.005] [arm grid spacing m = 0.02]
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <map>
#include <memory>
#include <numbers>
#include <random>
#include <string>
#include <utility>

#include <nlohmann/json.hpp>

#include "dabfc/contact_solver.h"
#include "dabfc/ur5e.h"

using namespace dabfc;
using nlohmann::json;

namespace {

constexpr float linkGridMargin = 0.05f; // m around each link's mesh
constexpr float armGridMargin = 0.1f;   // m around each arm
constexpr int samplesPerArm = 1000;     // surface samples of a whole arm
constexpr double scanRange = 0.75;      // m: x and y of the scanned base poses
constexpr double scanStep = 0.1;        // m
constexpr double scanHeights[] = {-0.1, 0, 0.1}; // m: z of the scanned base poses
constexpr int scanYaws = 24;            // every 15 degrees
constexpr int refinedScanPoses = 20;    // best scan poses refined on the whole-arm cost
constexpr int polishedPoses = 3;        // best refined poses polished on the exact per-link cost

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

SE3 trueArm2Base(const json& placement)
{
    const json& pose = placement["arm2_base_in_arm1_frame"];
    const Eigen::VectorXd t = vector(pose["translation"]);
    const Eigen::VectorXd q = vector(pose["quaternion_wxyz"]);
    return SE3(t, Eigen::Quaterniond(q[0], q[1], q[2], q[3]));
}

glm::vec3 toGlm(const Eigen::Vector3d& p)
{
    return {static_cast<float>(p.x()), static_cast<float>(p.y()), static_cast<float>(p.z())};
}

// A link's grid, cached under build/sdf_cache so later runs skip the minutes of building it
std::shared_ptr<const SdfGrid> cachedGrid(const std::string& name, const SdfMesh& mesh,
                                          const Box& box, float spacing)
{
    const std::filesystem::path dir = std::string(PROJECT_DIR) + "/build/sdf_cache";
    const std::filesystem::path file = dir / (name + "_" + std::to_string(spacing) + ".bin");
    if (std::ifstream in(file, std::ios::binary); in) {
        Box stored;
        in.read(reinterpret_cast<char*>(&stored), sizeof stored);
        std::vector<float> values((std::filesystem::file_size(file) - sizeof stored) / sizeof(float));
        in.read(reinterpret_cast<char*>(values.data()), std::ssize(values) * sizeof(float));
        return std::make_shared<const SdfGrid>(stored, spacing, std::move(values));
    }
    std::shared_ptr<const SdfGrid> grid;
    const double time = seconds([&] { grid = std::make_shared<const SdfGrid>(mesh, box, spacing); });
    std::fprintf(stderr, "%s grid built in %.1f s\n", name.c_str(), time);
    std::filesystem::create_directories(dir);
    std::ofstream out(file, std::ios::binary);
    out.write(reinterpret_cast<const char*>(&grid->box()), sizeof(Box));
    out.write(reinterpret_cast<const char*>(grid->values().data()),
              std::ssize(grid->values()) * sizeof(float));
    return grid;
}

// Every meshed link's shape, built on first use
class LinkShapes {
public:
    LinkShapes(const RobotArm& arm, int samples, float spacing)
        : m_arm(arm), m_samples(samples), m_spacing(spacing)
    {
    }

    std::shared_ptr<const Shape> operator[](int link)
    {
        std::shared_ptr<const Shape>& shape = m_shapes[link];
        if (shape) return shape;
        const Link& l = m_arm.links()[link];
        std::vector<std::string> paths;
        for (const std::string& part : l.parts)
            paths.push_back(std::string(MESH_DIR) + "/" + l.mesh + "_" + part + ".stl");
        const SdfMesh mesh(paths);
        m_areas[link] = mesh.area();
        Box box = mesh.bounds();
        const glm::vec3 center = 0.5f * (box.min + box.max);
        Shape built{.sdf = nullptr,
                    .surface = {},
                    .center = {center.x, center.y, center.z},
                    .radius = 0.5 * glm::length(box.max - box.min)};
        box.min -= linkGridMargin;
        box.max += linkGridMargin;
        built.sdf = cachedGrid(l.name, mesh, box, m_spacing);
        std::mt19937 rng(0);
        for (const glm::vec3& p : mesh.surfaceSamples(m_samples, rng))
            built.surface.emplace_back(p.x, p.y, p.z);
        shape = std::make_shared<const Shape>(std::move(built));
        return shape;
    }

    double area(int link) { return m_areas.at(link); }

private:
    const RobotArm& m_arm;
    int m_samples;
    float m_spacing;
    std::map<int, std::shared_ptr<const Shape>> m_shapes;
    std::map<int, double> m_areas;
};

// Some links of an arm at configuration q as one body in the arm's base frame: the union of
// their SDFs on a coarse grid, and their samples in proportion to their areas
std::shared_ptr<const Shape> wholeArm(const RobotArm& arm, const Eigen::VectorXd& q,
                                      const std::vector<int>& links, LinkShapes& shapes,
                                      float spacing)
{
    const std::vector<SE3> poses = arm.meshPoses(q);
    std::vector<SdfUnion::Part> parts;
    double area = 0;
    for (int link : links) {
        const std::shared_ptr<const Shape> shape = shapes[link];
        parts.push_back({shape->sdf, poses[link], shape->center, shape->radius});
        area += shapes.area(link);
    }
    const SdfUnion exact(std::move(parts));
    Box box = exact.bounds();
    const glm::vec3 center = 0.5f * (box.min + box.max);
    Shape built{.sdf = nullptr,
                .surface = {},
                .center = {center.x, center.y, center.z},
                .radius = 0.5 * glm::length(box.max - box.min)};
    box.min -= armGridMargin;
    box.max += armGridMargin;
    built.sdf = std::make_shared<const SdfGrid>(exact, box, spacing);
    for (int link : links) {
        const std::shared_ptr<const Shape> shape = shapes[link];
        const int count = static_cast<int>(std::round(samplesPerArm * shapes.area(link) / area));
        for (int k = 0; k < std::min(count, static_cast<int>(shape->surface.size())); ++k)
            built.surface.push_back(poses[link] * shape->surface[k]);
    }
    return std::make_shared<const Shape>(std::move(built));
}

int linkIndex(const RobotArm& arm, const std::string& name)
{
    for (int i = 0; i < std::ssize(arm.links()); ++i)
        if (arm.links()[i].name == name) return i;
    throw std::runtime_error("unknown link " + name);
}

struct Errors {
    double translation; // m
    double rotation;    // rad
};

Errors errors(const SE3& X, const SE3& truth)
{
    return {(X.translation() - truth.translation()).norm(),
            Eigen::AngleAxisd(truth.rotation().inverse() * X.rotation()).angle()};
}

void printRow(const char* label, const Errors& e, double cost, double time)
{
    std::printf("  %-22s %8.2f %8.3f %10.2e %8.1f\n", label, 1e3 * e.translation,
                e.rotation * 180 / std::numbers::pi, cost, time);
    std::fflush(stdout);
}

// The base poses on a grid of x, y, z and yaw, ordered by their cost
std::vector<SE3> scan(const ContactSolver& solver)
{
    std::vector<std::pair<double, SE3>> poses;
    for (double x = -scanRange; x <= scanRange; x += scanStep)
        for (double y = -scanRange; y <= scanRange; y += scanStep)
            for (double z : scanHeights)
                for (int k = 0; k < scanYaws; ++k) {
                    const SE3 X({x, y, z}, {0, 0, 2 * std::numbers::pi * k / scanYaws});
                    poses.emplace_back(solver.cost(X), X);
                }
    std::ranges::sort(poses, {}, &std::pair<double, SE3>::first);
    std::vector<SE3> ordered;
    for (const auto& [cost, X] : poses) ordered.push_back(X);
    return ordered;
}

} // namespace

int main(int argc, char** argv)
{
    const std::string path =
        argc > 1 ? argv[1] : std::string(PROJECT_DIR) + "/benchmark/contacts.json";
    const int samples = argc > 2 ? std::atoi(argv[2]) : 2000;
    const float linkSpacing = argc > 3 ? static_cast<float>(std::atof(argv[3])) : 0.005f;
    const float armSpacing = argc > 4 ? static_cast<float>(std::atof(argv[4])) : 0.02f;
    const json benchmark = json::parse(std::ifstream(path));

    const UR5e ur5e(MESH_DIR);
    LinkShapes shapes(ur5e, samples, linkSpacing);
    const std::vector<int> fixedLinks = {linkIndex(ur5e, "base"), linkIndex(ur5e, "shoulder")};
    std::vector<int> movingLinks; // every other meshed link
    for (int i = 0; i < std::ssize(ur5e.links()); ++i)
        if (!ur5e.links()[i].mesh.empty() && std::ranges::find(fixedLinks, i) == fixedLinks.end())
            movingLinks.push_back(i);
    std::vector<int> candidateLinks = movingLinks; // the per-link bodies, in order
    candidateLinks.insert(candidateLinks.end(), fixedLinks.begin(), fixedLinks.end());

    for (const json& placement : benchmark["placements"]) {
        std::vector<Contact> known;
        std::vector<Contact> perLink;
        std::vector<Contact> wholeArms;
        std::vector<std::vector<std::pair<int, int>>> truePairs; // per contact, link indices
        const double buildTime = seconds([&] {
            for (const json& contact : placement["contacts"]) {
                const Eigen::VectorXd q1 = vector(contact["q1"]);
                const Eigen::VectorXd q2 = vector(contact["q2"]);
                const std::vector<SE3> poses1 = ur5e.meshPoses(q1);
                const std::vector<SE3> poses2 = ur5e.meshPoses(q2);
                truePairs.emplace_back();
                for (const json& pair : contact["pairs"]) {
                    const int link1 = linkIndex(ur5e, pair["link1"]);
                    const int link2 = linkIndex(ur5e, pair["link2"]);
                    truePairs.back().emplace_back(link1, link2);
                    known.push_back({.bodies1 = {{shapes[link1], poses1[link1]}},
                                     .bodies2 = {{shapes[link2], poses2[link2]}}});
                }
                Contact links;
                for (int link : candidateLinks) {
                    const bool moves = std::ranges::find(fixedLinks, link) == fixedLinks.end();
                    links.bodies1.push_back({shapes[link], poses1[link], moves});
                    links.bodies2.push_back({shapes[link], poses2[link], moves});
                }
                perLink.push_back(std::move(links));
                wholeArms.push_back(
                    {.bodies1 = {{wholeArm(ur5e, q1, movingLinks, shapes, armSpacing), SE3{}},
                                 {wholeArm(ur5e, q1, fixedLinks, shapes, armSpacing), SE3{}, false}},
                     .bodies2 = {{wholeArm(ur5e, q2, movingLinks, shapes, armSpacing), SE3{}},
                                 {wholeArm(ur5e, q2, fixedLinks, shapes, armSpacing), SE3{}, false}}});
            }
        });
        const SE3 truth = trueArm2Base(placement);
        std::printf("placement %d: %zu contacts, arm 2 %.2f m away, bodies built in %.1f s\n",
                    placement["seed"].get<int>(), wholeArms.size(), truth.translation().norm(),
                    buildTime);
        std::printf("  %-22s %8s %8s %10s %8s\n", "solver", "pos mm", "rot deg", "cost",
                    "time s");

        const ContactSolver knownSolver(known);
        for (const auto& [label, X0] :
             {std::pair{"known, from truth", truth}, std::pair{"known, from identity", SE3{}}}) {
            SE3 X;
            const double time = seconds([&] { X = knownSolver.solve(X0); });
            printRow(label, errors(X, truth), knownSolver.cost(X), time);
        }

        const ContactSolver armSolver(wholeArms);
        const ContactSolver linkSolver(perLink);
        printRow("whole arms, at truth", errors(truth, truth), armSolver.cost(truth), 0);
        printRow("per link, at truth", errors(truth, truth), linkSolver.cost(truth), 0);
        std::vector<SE3> scanned;
        const double scanTime = seconds([&] { scanned = scan(armSolver); });
        std::fprintf(stderr, "  scanned %zu base poses in %.1f s\n", scanned.size(), scanTime);
        std::vector<std::pair<double, SE3>> refined; // whole-arm cost, pose
        const double refineTime = seconds([&] {
            for (int s = 0; s < refinedScanPoses; ++s) {
                const SE3 X = armSolver.solve(scanned[s], 50);
                refined.emplace_back(armSolver.cost(X), X);
                const Errors e = errors(X, truth);
                std::fprintf(stderr, "  scan pose %2d: cost %.2e, off by %.0f mm %.0f deg\n",
                             s + 1, refined.back().first, 1e3 * e.translation,
                             e.rotation * 180 / std::numbers::pi);
            }
            std::ranges::sort(refined, {}, &std::pair<double, SE3>::first);
        });
        printRow("whole arms, scanned", errors(refined[0].second, truth), refined[0].first,
                 scanTime + refineTime);

        SE3 polished;
        double polishedCost = std::numeric_limits<double>::infinity();
        const double polishTime = seconds([&] {
            for (int s = 0; s < polishedPoses; ++s) {
                const SE3 X = linkSolver.solve(refined[s].second);
                const double cost = linkSolver.cost(X);
                if (cost < polishedCost) {
                    polished = X;
                    polishedCost = cost;
                }
            }
        });
        printRow("per link, polished", errors(polished, truth), polishedCost, polishTime);
        int identified = 0;
        for (int t = 0; t < std::ssize(perLink); ++t) {
            const ContactSolver::Residual r = linkSolver.residuals(polished, t).front();
            const std::pair found{candidateLinks[r.body1], candidateLinks[r.body2]};
            identified += std::ranges::find(truePairs[t], found) != truePairs[t].end();
        }
        std::printf("  touching links identified in %d of %zu contacts\n", identified,
                    perLink.size());
        std::fflush(stdout);
    }
}
