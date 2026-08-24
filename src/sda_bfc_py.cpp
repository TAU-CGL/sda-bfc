#include <nanobind/nanobind.h>
#include <nanobind/eigen/dense.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/pair.h>
#include <nanobind/stl/vector.h>

#include <fk/fk_ur5e.hpp>
#include <geometry.hpp>
#include <solvers/solver_newton.hpp>
#include <solvers/solver_annealing_lp.hpp>
#include <solvers/solver_smc.hpp>
#include <solvers/solver_adam.hpp>
#include <sim/experiment.hpp>

namespace nb = nanobind;

NB_MODULE(_sda_bfc, m) {
    m.doc() = "SDA-BFC C++ kernels";
    m.attr("__version__") = "0.1.0";

    nb::class_<sda_bfc::UR5e>(m, "UR5e")
        .def(nb::init<>())
        .def("get_cylinder_transform", &sda_bfc::UR5e::getCylinderTransform,
             nb::arg("link_index"), nb::arg("q"), nb::arg("z_offset") = 0.007)
        .def("get_link_radius", &sda_bfc::UR5e::getLinkRadius,
             nb::arg("link_index"));

    nb::class_<sda_bfc::CylinderPose>(m, "CylinderPose")
        .def(nb::init<sda_bfc::R3, sda_bfc::S2, double>(),
             nb::arg("p"), nb::arg("u"), nb::arg("r"))
        .def_rw("p", &sda_bfc::CylinderPose::p)
        .def_rw("u", &sda_bfc::CylinderPose::u)
        .def_rw("r", &sda_bfc::CylinderPose::r)
        .def_static("from_se3", &sda_bfc::CylinderPose::fromSE3,
             nb::arg("T"), nb::arg("r"))
        .def("to_se3", &sda_bfc::CylinderPose::toSE3)
        .def("signed_distance",
             [](const sda_bfc::CylinderPose& self, const sda_bfc::CylinderPose& other) {
                 return sda_bfc::CylinderPose::signedDistance(self, other);
             },
             nb::arg("other"))
        .def("signed_distance",
             nb::overload_cast<double>(&sda_bfc::CylinderPose::signedDistance, nb::const_),
             nb::arg("other_r"))
        .def("implicit_touch_condition", &sda_bfc::CylinderPose::implicitTouchCondition,
             nb::arg("other_r"));

    nb::class_<sda_bfc::SolverNewton>(m, "SolverNewton")
        .def(nb::init<const std::vector<sda_bfc::SE3>&, const std::vector<sda_bfc::SE3>&, double, double>(),
             nb::arg("As"), nb::arg("Bs"), nb::arg("radius_a"), nb::arg("radius_b"))
        .def("residual",
             [](const sda_bfc::SolverNewton& self, const sda_bfc::SE3& X, size_t t) {
                 return self.residual(X, t);
             },
             nb::arg("X"), nb::arg("t"))
        .def("cost", &sda_bfc::SolverNewton::cost, nb::arg("X"))
        .def("solve", &sda_bfc::SolverNewton::solve,
             nb::arg("x0"), nb::arg("max_iterations") = 100, nb::arg("tolerance") = 1e-14)
        .def("solve_multistart", &sda_bfc::SolverNewton::solveMultistart,
             nb::arg("num_starts") = 2000, nb::arg("translation_range") = 1.2,
             nb::arg("max_iterations") = 150, nb::arg("seed") = 0);

    nb::class_<sda_bfc::SolverAnnealingLP>(m, "SolverAnnealingLP")
        .def(nb::init<const std::vector<sda_bfc::SE3>&, const std::vector<sda_bfc::SE3>&,
                      double, double, int, int, int, double, unsigned, std::vector<double>>(),
             nb::arg("As"), nb::arg("Bs"), nb::arg("radius_a"), nb::arg("radius_b"),
             nb::arg("n_initial") = 5000, nb::arg("n_elites") = 40, nb::arg("n_per_elite") = 40,
             nb::arg("eps") = 2e-3, nb::arg("seed") = 0,
             nb::arg("sigma_schedule") = std::vector<double>{0.3, 0.15, 0.075, 0.03, 0.01})
        .def("cost", &sda_bfc::SolverAnnealingLP::cost, nb::arg("X"))
        .def("solve",
             [](const sda_bfc::SolverAnnealingLP& self) { return self.solve(sda_bfc::SE3::Identity()); });

    nb::class_<sda_bfc::SolverSMC>(m, "SolverSMC")
        .def(nb::init<const std::vector<sda_bfc::SE3>&, const std::vector<sda_bfc::SE3>&,
                      double, double, int, double, double, double, unsigned, std::vector<double>>(),
             nb::arg("As"), nb::arg("Bs"), nb::arg("radius_a"), nb::arg("radius_b"),
             nb::arg("num_particles") = 1000, nb::arg("fresh_fraction") = 0.1,
             nb::arg("ess_target") = 0.02, nb::arg("eps") = 2e-3, nb::arg("seed") = 0,
             nb::arg("sigma_schedule") = std::vector<double>{0.3, 0.2, 0.15, 0.1, 0.075, 0.05, 0.03, 0.02, 0.01})
        .def("cost", &sda_bfc::SolverSMC::cost, nb::arg("X"))
        .def("solve",
             [](const sda_bfc::SolverSMC& self) { return self.solve(sda_bfc::SE3::Identity()); });

    nb::class_<sda_bfc::SolverAdam>(m, "SolverAdam")
        .def(nb::init<const std::vector<sda_bfc::SE3>&, const std::vector<sda_bfc::SE3>&,
                      double, double, double, double, double, double>(),
             nb::arg("As"), nb::arg("Bs"), nb::arg("radius_a"), nb::arg("radius_b"),
             nb::arg("learning_rate") = 0.01, nb::arg("beta1") = 0.9,
             nb::arg("beta2") = 0.999, nb::arg("epsilon") = 1e-8)
        .def("cost", &sda_bfc::SolverAdam::cost, nb::arg("X"))
        .def("solve", &sda_bfc::SolverAdam::solve,
             nb::arg("x0"), nb::arg("max_iterations") = 2000, nb::arg("tolerance") = 1e-12)
        .def("solve_multistart", &sda_bfc::SolverAdam::solveMultistart,
             nb::arg("num_starts") = 2000, nb::arg("translation_range") = 1.2,
             nb::arg("max_iterations") = 2000, nb::arg("seed") = 0);

    nb::class_<sda_bfc::Capsule>(m, "Capsule")
        .def(nb::init<sda_bfc::R3, sda_bfc::R3, double>(),
             nb::arg("a"), nb::arg("b"), nb::arg("r"))
        .def_rw("a", &sda_bfc::Capsule::a)
        .def_rw("b", &sda_bfc::Capsule::b)
        .def_rw("r", &sda_bfc::Capsule::r);

    nb::class_<sda_bfc::SegmentClosest>(m, "SegmentClosest")
        .def_ro("distance", &sda_bfc::SegmentClosest::distance)
        .def_ro("s", &sda_bfc::SegmentClosest::s)
        .def_ro("t", &sda_bfc::SegmentClosest::t);

    nb::class_<sda_bfc::UncertaintyRanges>(m, "UncertaintyRanges")
        .def(nb::init<>())
        .def_rw("x", &sda_bfc::UncertaintyRanges::x)
        .def_rw("y", &sda_bfc::UncertaintyRanges::y)
        .def_rw("z", &sda_bfc::UncertaintyRanges::z)
        .def_rw("roll", &sda_bfc::UncertaintyRanges::roll)
        .def_rw("pitch", &sda_bfc::UncertaintyRanges::pitch)
        .def_rw("yaw", &sda_bfc::UncertaintyRanges::yaw);

    m.def("capsule_obb_vertices",
          [](const sda_bfc::Capsule& c, double extra_radius) {
              auto verts = sda_bfc::capsuleObbVertices(c, extra_radius);
              Eigen::Matrix<double, 8, 3> out;
              for (int i = 0; i < 8; i++) out.row(i) = verts[i].transpose();
              return out;
          },
          nb::arg("capsule"), nb::arg("extra_radius") = 0.0);
    m.def("expand_capsule", &sda_bfc::expandCapsule,
          nb::arg("capsule"), nb::arg("ranges"));

    m.def("segment_closest", &sda_bfc::segmentClosest,
          nb::arg("p1"), nb::arg("q1"), nb::arg("p2"), nb::arg("q2"));
    m.def("segment_distance", &sda_bfc::segmentDistance,
          nb::arg("p1"), nb::arg("q1"), nb::arg("p2"), nb::arg("q2"));
    m.def("fold_config", &sda_bfc::foldConfig, nb::arg("yaw"));

    nb::class_<sda_bfc::PairClearance>(m, "PairClearance")
        .def_ro("i", &sda_bfc::PairClearance::i)
        .def_ro("j", &sda_bfc::PairClearance::j)
        .def_ro("clearance", &sda_bfc::PairClearance::clearance);

    nb::class_<sda_bfc::TwoArmScene>(m, "TwoArmScene")
        .def(nb::init<const sda_bfc::UR5e&, const sda_bfc::SE3&>(),
             nb::arg("robot"), nb::arg("X"))
        .def("self_clearances", &sda_bfc::TwoArmScene::selfClearances, nb::arg("q"))
        .def("cross_clearances", &sda_bfc::TwoArmScene::crossClearances,
             nb::arg("q_a"), nb::arg("q_b"))
        .def("first_cross_contact", &sda_bfc::TwoArmScene::firstCrossContact,
             nb::arg("q_a"), nb::arg("q_b"))
        .def("min_self_clearance", &sda_bfc::TwoArmScene::minSelfClearance, nb::arg("q"))
        .def("contact_interior", &sda_bfc::TwoArmScene::contactInterior,
             nb::arg("q_a"), nb::arg("q_b"), nb::arg("i"), nb::arg("j"),
             nb::arg("margin") = 0.03)
        .def("collision_free_at_home", &sda_bfc::TwoArmScene::collisionFreeAtHome,
             nb::arg("yaw_a") = 0.0, nb::arg("yaw_b") = 0.0)
        .def("valid_contact", &sda_bfc::TwoArmScene::validContact,
             nb::arg("q_a"), nb::arg("q_b"), nb::arg("ti"), nb::arg("tj"),
             nb::arg("touch_tol") = 1e-6, nb::arg("others_margin") = 5e-3,
             nb::arg("interior_margin") = 0.03)
        .def("link_extents",
             [](const sda_bfc::TwoArmScene& self, int i) {
                 return self.model().extents(i);
             },
             nb::arg("i"))
        .def("link_radius",
             [](const sda_bfc::TwoArmScene& self, int i) {
                 return self.model().linkRadius(i);
             },
             nb::arg("i"))
        .def("link_z_offset",
             [](const sda_bfc::TwoArmScene& self, int i) {
                 return self.model().linkZOffset(i);
             },
             nb::arg("i"))
        .def("capsules",
             [](const sda_bfc::TwoArmScene& self, const sda_bfc::JointConfig& q) {
                 auto caps = self.model().capsules(q);
                 return std::vector<sda_bfc::Capsule>(caps.begin(), caps.end());
             },
             nb::arg("q"));

    nb::class_<sda_bfc::PlacementParams>(m, "PlacementParams")
        .def(nb::init<>())
        .def_rw("min_distance", &sda_bfc::PlacementParams::minDistance)
        .def_rw("max_distance", &sda_bfc::PlacementParams::maxDistance)
        .def_rw("max_tilt_deg", &sda_bfc::PlacementParams::maxTiltDeg)
        .def_rw("max_abs_z", &sda_bfc::PlacementParams::maxAbsZ)
        .def_rw("max_attempts", &sda_bfc::PlacementParams::maxAttempts);

    nb::class_<sda_bfc::ContactParams>(m, "ContactParams")
        .def(nb::init<>())
        .def_rw("max_iterations", &sda_bfc::ContactParams::maxIterations)
        .def_rw("tolerance", &sda_bfc::ContactParams::tolerance)
        .def_rw("damping", &sda_bfc::ContactParams::damping)
        .def_rw("min_denom", &sda_bfc::ContactParams::minDenom)
        .def_rw("joint_range", &sda_bfc::ContactParams::jointRange)
        .def_rw("max_proximal_restarts", &sda_bfc::ContactParams::maxProximalRestarts)
        .def_rw("max_distal_attempts", &sda_bfc::ContactParams::maxDistalAttempts)
        .def_rw("clearance_tol", &sda_bfc::ContactParams::clearanceTol)
        .def_rw("others_margin", &sda_bfc::ContactParams::othersMargin)
        .def_rw("interior_margin", &sda_bfc::ContactParams::interiorMargin)
        .def_rw("touch_link", &sda_bfc::ContactParams::touchLink);

    nb::class_<sda_bfc::ExperimentParams>(m, "ExperimentParams")
        .def(nb::init<>())
        .def_rw("placement", &sda_bfc::ExperimentParams::placement)
        .def_rw("contact", &sda_bfc::ExperimentParams::contact)
        .def_rw("max_placement_attempts", &sda_bfc::ExperimentParams::maxPlacementAttempts);

    m.def("sample_placement",
          [](unsigned seed, const sda_bfc::PlacementParams& params) {
              std::mt19937 gen(seed);
              return sda_bfc::samplePlacement(gen, params);
          },
          nb::arg("seed"), nb::arg("params") = sda_bfc::PlacementParams{});
    m.def("sample_valid_placement",
          [](unsigned seed, const sda_bfc::UR5e& robot,
             const sda_bfc::PlacementParams& params) {
              std::mt19937 gen(seed);
              return sda_bfc::sampleValidPlacement(gen, robot, params);
          },
          nb::arg("seed"), nb::arg("robot") = sda_bfc::UR5e{},
          nb::arg("params") = sda_bfc::PlacementParams{});

    nb::class_<sda_bfc::ContactPose>(m, "ContactPose")
        .def_ro("q_a", &sda_bfc::ContactPose::qA)
        .def_ro("q_b", &sda_bfc::ContactPose::qB);

    nb::class_<sda_bfc::ContactGenerator>(m, "ContactGenerator")
        .def(nb::init<const sda_bfc::UR5e&, const sda_bfc::SE3&, sda_bfc::ContactParams>(),
             nb::arg("robot"), nb::arg("X"),
             nb::arg("params") = sda_bfc::ContactParams{})
        .def("residual", &sda_bfc::ContactGenerator::residual, nb::arg("theta"))
        .def("residual_with_jacobian",
             [](const sda_bfc::ContactGenerator& self,
                const sda_bfc::ContactGenerator::Theta& theta) {
                 sda_bfc::ContactGenerator::Theta J;
                 double f = self.residualWithJacobian(theta, J);
                 return std::make_pair(f, J);
             },
             nb::arg("theta"))
        .def("generate",
             [](const sda_bfc::ContactGenerator& self, unsigned seed) {
                 std::mt19937 gen(seed);
                 return self.generate(gen);
             },
             nb::arg("seed"));

    nb::class_<sda_bfc::Experiment>(m, "Experiment")
        .def_ro("X", &sda_bfc::Experiment::X)
        .def_ro("q_as", &sda_bfc::Experiment::qAs)
        .def_ro("q_bs", &sda_bfc::Experiment::qBs);

    m.def("generate_experiment",
          [](int num_contacts, unsigned seed, const sda_bfc::UR5e& robot,
             const sda_bfc::ExperimentParams& params) {
              return sda_bfc::generateExperiment(robot, num_contacts, seed, params);
          },
          nb::arg("num_contacts"), nb::arg("seed") = 0,
          nb::arg("robot") = sda_bfc::UR5e{},
          nb::arg("params") = sda_bfc::ExperimentParams{});
}
