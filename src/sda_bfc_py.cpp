#include <nanobind/nanobind.h>
#include <nanobind/eigen/dense.h>

#include <sda_bfc.hpp>
#include <fk_ur5e.hpp>

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
}
