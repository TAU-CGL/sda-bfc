#include <nanobind/nanobind.h>
#include <nanobind/eigen/dense.h>

#include <sda_bfc.hpp>
#include <fk_ur5e.hpp>
#include <geometry.hpp>

namespace nb = nanobind;

NB_MODULE(_sda_bfc, m) {
    m.doc() = "SDA-BFC C++ kernels";
    m.attr("__version__") = "0.1.0";

    m.def("test", &sda_bfc::test);

    nb::class_<sda_bfc::UR5e>(m, "UR5e")
        .def(nb::init<>())
        .def("get_cylinder_pose", &sda_bfc::UR5e::getCylinderPose,
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
             nb::arg("other_r"));
}
