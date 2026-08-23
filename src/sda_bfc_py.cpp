#include <nanobind/nanobind.h>

#include <sda_bfc.hpp>

namespace nb = nanobind;

NB_MODULE(_sda_bfc, m) {
    m.doc() = "SDA-BFC C++ kernels";
    m.attr("__version__") = "0.1.0";

    m.def("test", &sda_bfc::test);
}
