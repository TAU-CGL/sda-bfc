#ifndef DABFC_CONTACT_SOLVER_H_
#define DABFC_CONTACT_SOLVER_H_

#include <memory>
#include <vector>

#include <Eigen/Core>

#include "dabfc/sdf.h"
#include "dabfc/se3.h"

namespace dabfc {

// A link's geometry for the solver, both in the link's mesh frame
struct LinkShape {
    std::shared_ptr<const Sdf> sdf;
    std::vector<Eigen::Vector3d> surface; // points sampled on the link's surface
};

// Two links known to touch: one of arm 1, with its mesh pose in arm 1's base frame, and one of
// arm 2, with its mesh pose in arm 2's base frame
struct TouchingLinks {
    std::shared_ptr<const LinkShape> shape1;
    SE3 pose1;
    std::shared_ptr<const LinkShape> shape2;
    SE3 pose2;
};

// Finds X, the pose of arm 2's base in arm 1's base frame, from pairs of touching links. For pair
// t the signed distance between the links at X,
//   f_t(X) = min(min_p F1(pose1^-1 X pose2 p), min_p F2(pose2^-1 X^-1 pose1 p))
// with p over the surface samples of the other link, must vanish. Least squares over all pairs,
// Levenberg-Marquardt with the update X <- X exp(delta)
class ContactSolver {
public:
    struct Residual {
        double value;      // f_t(X)
        Vector6d jacobian; // d f_t(X exp(delta)) / d delta at delta = 0
    };

    explicit ContactSolver(std::vector<TouchingLinks> contacts);

    Residual residual(const SE3& X, int t) const;
    double cost(const SE3& X) const; // sum of squared residuals
    SE3 solve(SE3 X, int maxIterations = 100, double tolerance = 1e-10) const;

private:
    std::vector<TouchingLinks> m_contacts;
};

} // namespace dabfc

#endif // DABFC_CONTACT_SOLVER_H_
