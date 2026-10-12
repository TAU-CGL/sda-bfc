// Writes a benchmark of contact configurations: several random placements of arm 2 near arm 1
// (DualArmSimulator::randomBase2), each with the arm 2 base transform and the contact
// configurations a ContactSearch finds there (joint values of both arms and the touching link
// pairs with their world contact point), as JSON.
// Usage: make_benchmark [placements = 10] [contacts = 50] [seed = 0] [cycles = 1000] [out = benchmark/contacts.json]
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <memory>
#include <random>
#include <string>

#include "dabfc/contact_search.h"
#include "dabfc/dual_arm_simulator.h"
#include "dabfc/ur5e.h"

using namespace dabfc;

namespace {

struct Settings {
    int placements = 10;
    int contacts = 50;    // per placement
    unsigned seed = 0;    // placement i uses seed + i for its base and its search
    int maxCycles = 1000; // a search gives up after this many cycles
    std::string out = std::string(PROJECT_DIR) + "/benchmark/contacts.json";
};

void writeArray(std::FILE* f, const Eigen::VectorXd& values)
{
    std::fputc('[', f);
    for (int k = 0; k < values.size(); ++k) std::fprintf(f, k ? ", %.6f" : "%.6f", values[k]);
    std::fputc(']', f);
}

void writePlacement(std::FILE* f, const DualArmSimulator& sim, const ContactSearch& search,
                    unsigned seed)
{
    const Eigen::Vector3d p = sim.base(1).translation();
    const Eigen::Quaterniond r = sim.base(1).rotation();
    std::fprintf(f, "    {\n      \"seed\": %u,\n      \"cycles\": %d,\n", seed, search.cycles());
    std::fprintf(f, "      \"base2\": {\"translation\": [%.6f, %.6f, %.6f], \"quaternion_wxyz\": "
                    "[%.6f, %.6f, %.6f, %.6f]},\n",
                 p.x(), p.y(), p.z(), r.w(), r.x(), r.y(), r.z());
    std::fprintf(f, "      \"contacts\": [");
    const std::vector<Link>& links = sim.arm(0)->links();
    for (std::size_t i = 0; i < search.contacts().size(); ++i) {
        const SimulationCollisionReport& report = search.contacts()[i];
        std::fprintf(f, i ? ",\n        {\"q1\": " : "\n        {\"q1\": ");
        writeArray(f, report.q[0]);
        std::fprintf(f, ", \"q2\": ");
        writeArray(f, report.q[1]);
        std::fprintf(f, ", \"pairs\": [");
        bool first = true;
        for (const SimulationCollisionReport::Contact& c : report.contacts) {
            if (!c.otherArm || *c.otherArm == c.arm) continue;
            std::fprintf(f, "%s{\"link1\": \"%s\", \"link2\": \"%s\", \"point\": [%.6f, %.6f, %.6f]}",
                         first ? "" : ", ", links[c.link].name.c_str(),
                         links[c.otherLink].name.c_str(), c.point.x(), c.point.y(), c.point.z());
            first = false;
        }
        std::fprintf(f, "]}");
    }
    std::fprintf(f, "\n      ]\n    }");
}

} // namespace

int main(int argc, char** argv)
{
    Settings s;
    if (argc > 1) s.placements = std::atoi(argv[1]);
    if (argc > 2) s.contacts = std::atoi(argv[2]);
    if (argc > 3) s.seed = static_cast<unsigned>(std::atoi(argv[3]));
    if (argc > 4) s.maxCycles = std::atoi(argv[4]);
    if (argc > 5) s.out = argv[5];

    std::FILE* f = std::fopen(s.out.c_str(), "w");
    if (!f) {
        std::fprintf(stderr, "cannot write %s\n", s.out.c_str());
        return 1;
    }
    std::fprintf(f, "{\n  \"settings\": {\"placements\": %d, \"contacts\": %d, \"seed\": %u, "
                    "\"max_cycles\": %d},\n  \"placements\": [\n",
                 s.placements, s.contacts, s.seed, s.maxCycles);

    const std::shared_ptr<const RobotArm> ur5e = std::make_shared<UR5e>(MESH_DIR);
    for (int i = 0; i < s.placements; ++i) {
        const unsigned seed = s.seed + static_cast<unsigned>(i);
        DualArmSimulator sim(ur5e, SE3{}, ur5e, SE3{});
        std::mt19937 rng(seed);
        for (int tries = 0; tries < 100; ++tries) {
            sim.setBase(1, sim.randomBase2(rng));
            if (sim.isCollisionFreeNow()) break;
        }
        ContactSearch search(sim, s.contacts, seed);
        while (search.step(std::numeric_limits<double>::infinity()) &&
               search.cycles() <= s.maxCycles) {}
        std::fprintf(stderr, "placement %d: arm 2 %.2f m away, %zu contacts in %d cycles\n", i,
                     sim.base(1).translation().norm(), search.contacts().size(), search.cycles());
        std::fprintf(f, i ? ",\n" : "");
        writePlacement(f, sim, search, seed);
    }
    std::fprintf(f, "\n  ]\n}\n");
    std::fclose(f);
}
