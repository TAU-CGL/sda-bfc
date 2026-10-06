// Throughput of SdfMesh construction / valueAt / gradAt on the UR5e visual meshes,
// serial vs. OpenMP. Usage: bench_sdf [num_queries]
#include "dabfc/sdf.h"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <optional>
#include <random>
#include <vector>

#include <omp.h>

using namespace dabfc;
namespace fs = std::filesystem;

template <typename F> double seconds(F f)
{
    const auto start = std::chrono::steady_clock::now();
    f();
    return std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
}

int main(int argc, char** argv)
{
    const int n = argc > 1 ? std::atoi(argv[1]) : 1000;
    std::vector<glm::vec3> points(n);
    std::mt19937 rng(0);
    std::uniform_real_distribution<float> uniform(-0.2f, 0.2f);
    for (glm::vec3& p : points) p = {uniform(rng), uniform(rng), uniform(rng)};
    std::vector<float> values(n);
    std::vector<glm::vec3> grads(n);

    std::vector<fs::path> paths(fs::directory_iterator(MESH_DIR), {});
    std::sort(paths.begin(), paths.end());
    std::printf("%d queries, %d threads; times in us/query\n", n, omp_get_max_threads());
    std::printf("%-32s %9s %9s %9s %9s %9s\n", "mesh", "build ms", "value", "value||", "grad",
                "grad||");
    for (const fs::path& path : paths) {
        std::optional<SdfMesh> sdf;
        const double build = seconds([&] { sdf.emplace(path.string()); });
        const double value = seconds([&] {
            for (int i = 0; i < n; ++i) values[i] = sdf->valueAt(points[i]);
        });
        const double valuePar = seconds([&] {
#pragma omp parallel for schedule(dynamic, 16)
            for (int i = 0; i < n; ++i) values[i] = sdf->valueAt(points[i]);
        });
        const double grad = seconds([&] {
            for (int i = 0; i < n; ++i) grads[i] = sdf->gradAt(points[i]);
        });
        const double gradPar = seconds([&] {
#pragma omp parallel for schedule(dynamic, 16)
            for (int i = 0; i < n; ++i) grads[i] = sdf->gradAt(points[i]);
        });
        std::printf("%-32s %9.1f %9.1f %9.1f %9.1f %9.1f\n", path.stem().string().c_str(),
                    1e3 * build, 1e6 * value / n, 1e6 * valuePar / n, 1e6 * grad / n,
                    1e6 * gradPar / n);
    }
}
