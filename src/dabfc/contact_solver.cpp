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

ContactSolver::ContactSolver(std::vector<Contact> contacts) : m_contacts(std::move(contacts)) {}

std::vector<ContactSolver::Residual> ContactSolver::residuals(const SE3& X, int t) const
{
    const Contact& c = m_contacts.at(t);
    struct Candidate {
        double bound; // the pair's distance is at least this: bounding spheres
        int i;
        int j;
    };
    std::vector<Candidate> candidates;
    for (int i = 0; i < std::ssize(c.bodies1); ++i) {
        const Eigen::Vector3d ci = c.bodies1[i].pose * c.bodies1[i].shape->center;
        for (int j = 0; j < std::ssize(c.bodies2); ++j) {
            const Eigen::Vector3d cj = X * (c.bodies2[j].pose * c.bodies2[j].shape->center);
            candidates.push_back(
                {(ci - cj).norm() - c.bodies1[i].shape->radius - c.bodies2[j].shape->radius, i, j});
        }
    }
    std::ranges::sort(candidates, {}, &Candidate::bound);
    Residual touching{std::numeric_limits<double>::infinity(), Vector6d::Zero(), -1, -1};
    std::vector<Residual> penetrating;
    for (const Candidate& candidate : candidates) {
        if (candidate.bound >= std::max(touching.value, 0.)) break; // nothing closer remains
        const Body& body1 = c.bodies1[candidate.i];
        const Body& body2 = c.bodies2[candidate.j];
        const bool mayTouch = body1.moves || body2.moves;
        if (mayTouch ? candidate.bound >= touching.value : candidate.bound >= 0) continue;
        Residual r = residual(X, body1, body2);
        r.body1 = candidate.i;
        r.body2 = candidate.j;
        if (mayTouch && r.value < touching.value) touching = r;
        if (!mayTouch && r.value < 0) penetrating.push_back(r);
    }
    penetrating.insert(penetrating.begin(), touching);
    return penetrating;
}

ContactSolver::Residual ContactSolver::residual(const SE3& X, const Body& body1,
                                                const Body& body2) const
{
    const SE3 M = body1.pose.inverse() * X;           // arm 2's base in link 1's frame
    const SE3 N = body2.pose.inverse() * X.inverse(); // arm 1's base in link 2's frame
    const Nearest in1 = nearest(*body1.shape->sdf, M * body2.pose, body2.shape->surface);
    const Nearest in2 = nearest(*body2.shape->sdf, N * body1.pose, body1.shape->surface);
    Vector6d jacobian;
    if (in1.distance <= in2.distance) {
        const Eigen::Vector3d b = body2.pose * in1.sample; // in arm 2's base frame
        const Eigen::Vector3d g = M.rotation().inverse() * grad(*body1.shape->sdf, M * b);
        jacobian << g, b.cross(g);
        return {in1.distance, jacobian, 0, 0};
    }
    const Eigen::Vector3d a = body1.pose * in2.sample; // in arm 1's base frame
    const Eigen::Vector3d u = X.inverse() * a;         // in arm 2's base frame
    const Eigen::Vector3d h = body2.pose.rotation() * grad(*body2.shape->sdf, N * a);
    jacobian << -h, -u.cross(h);
    return {in2.distance, jacobian, 0, 0};
}

std::vector<ContactSolver::Residual> ContactSolver::residuals(const SE3& X) const
{
    std::vector<std::vector<Residual>> perContact(m_contacts.size());
#pragma omp parallel for schedule(dynamic)
    for (int t = 0; t < std::ssize(m_contacts); ++t) perContact[t] = residuals(X, t);
    std::vector<Residual> all;
    for (const std::vector<Residual>& rs : perContact) all.insert(all.end(), rs.begin(), rs.end());
    return all;
}

double ContactSolver::cost(const SE3& X) const
{
    double total = 0;
    for (const Residual& r : residuals(X)) total += r.value * r.value;
    return total;
}

SE3 ContactSolver::solve(SE3 X, int maxIterations, double tolerance) const
{
    double lambda = 1e-3;
    double current = cost(X);
    for (int iteration = 0; iteration < maxIterations; ++iteration) {
        Matrix6d H = Matrix6d::Zero();
        Vector6d g = Vector6d::Zero();
        for (const Residual& r : residuals(X)) {
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
