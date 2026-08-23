#pragma once

#define _USE_MATH_DEFINES
#include <cmath>
#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

#include <Eigen/Dense>

namespace sda_bfc {
    using SE3 = Eigen::Matrix4d;
}