#include <nanobind/nanobind.h>
#include <nanobind/eigen/dense.h>
#include <nanobind/stl/vector.h>

#include <fk/fk_ur5e.hpp>
#include <geometry.hpp>
#include <solvers/solver_newton.hpp>
#include <solvers/solver_annealing_lp.hpp>
#include <solvers/solver_smc.hpp>
#include <solvers/solver_adam.hpp>

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
}
