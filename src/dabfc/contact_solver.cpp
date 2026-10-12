#include "dabfc/contact_solver.h"

#include <algorithm>
#include <limits>
#include <utility>

namespace dabfc {

namespace {

using Matrix6d = Eigen::Matrix<double, 6, 6>;

glm::vec3 toGlm(const Eigen::Vector3d& p)
{
    return {static_cast<float>(p.x()), static_cast<float>(p.y()), static_cast<float>(p.z())};
}

Eigen::Vector3d grad(const Sdf& sdf, const Eigen::Vector3d& p)
{
    const glm::vec3 g = sdf.gradAt(toGlm(p));
    return {g.x, g.y, g.z};
}

// The surface sample of one link closest to (or deepest in) the other link, whose SDF is
// evaluated at T * sample
struct Nearest {
    double distance;
    Eigen::Vector3d sample;
};

Nearest nearest(const Sdf& sdf, const SE3& T, const std::vector<Eigen::Vector3d>& samples)
{
    Nearest best{std::numeric_limits<double>::infinity(), Eigen::Vector3d::Zero()};
    for (const Eigen::Vector3d& p : samples) {
        const double distance = sdf.valueAt(toGlm(T * p));
        if (distance < best.distance) best = {distance, p};
    }
    return best;
}

} // namespace

ContactSolver::ContactSolver(std::vector<TouchingLinks> contacts) : m_contacts(std::move(contacts))
{
}

ContactSolver::Residual ContactSolver::residual(const SE3& X, int t) const
{
    const TouchingLinks& c = m_contacts.at(t);
    const SE3 M = c.pose1.inverse() * X;           // arm 2's base in link 1's frame
    const SE3 N = c.pose2.inverse() * X.inverse(); // arm 1's base in link 2's frame
    const Nearest in1 = nearest(*c.shape1->sdf, M * c.pose2, c.shape2->surface);
    const Nearest in2 = nearest(*c.shape2->sdf, N * c.pose1, c.shape1->surface);
    Vector6d jacobian;
    if (in1.distance <= in2.distance) {
        const Eigen::Vector3d b = c.pose2 * in1.sample; // in arm 2's base frame
        const Eigen::Vector3d g = M.rotation().inverse() * grad(*c.shape1->sdf, M * b);
        jacobian << g, b.cross(g);
        return {in1.distance, jacobian};
    }
    const Eigen::Vector3d a = c.pose1 * in2.sample; // in arm 1's base frame
    const Eigen::Vector3d u = X.inverse() * a;      // in arm 2's base frame
    const Eigen::Vector3d h = c.pose2.rotation() * grad(*c.shape2->sdf, N * a);
    jacobian << -h, -u.cross(h);
    return {in2.distance, jacobian};
}

double ContactSolver::cost(const SE3& X) const
{
    double total = 0;
    for (int t = 0; t < std::ssize(m_contacts); ++t) {
        const double f = residual(X, t).value;
        total += f * f;
    }
    return total;
}

SE3 ContactSolver::solve(SE3 X, int maxIterations, double tolerance) const
{
    double lambda = 1e-3;
    double current = cost(X);
    for (int iteration = 0; iteration < maxIterations; ++iteration) {
        Matrix6d H = Matrix6d::Zero();
        Vector6d g = Vector6d::Zero();
        for (int t = 0; t < std::ssize(m_contacts); ++t) {
            const Residual r = residual(X, t);
            H += r.jacobian * r.jacobian.transpose();
            g += r.jacobian * r.value;
        }
        for (;;) { // damp more until the step lowers the cost
            const Vector6d delta = -(H + lambda * Matrix6d::Identity()).ldlt().solve(g);
            const SE3 candidate = X * SE3::exp(delta);
            const double next = cost(candidate);
            if (next < current) {
                X = candidate;
                current = next;
                lambda = std::max(lambda / 3, 1e-12);
                if (delta.norm() < tolerance) return X;
                break;
            }
            lambda *= 10;
            if (lambda > 1e12) return X;
        }
    }
    return X;
}

} // namespace dabfc
