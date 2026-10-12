#ifndef DABFC_CONTACT_SOLVER_H_
#define DABFC_CONTACT_SOLVER_H_

#include <memory>
#include <vector>

#include <Eigen/Core>

#include "dabfc/sdf.h"
#include "dabfc/se3.h"

namespace dabfc {

// A rigid body's geometry for the solver (a link, or a whole arm at one configuration), all in
// the body's own frame
struct Shape {
    std::shared_ptr<const Sdf> sdf;
    std::vector<Eigen::Vector3d> surface; // points sampled on the body's surface
    Eigen::Vector3d center;               // a sphere enclosing the whole body
    double radius;
};

struct Body {
    std::shared_ptr<const Shape> shape;
    SE3 pose;          // the body's frame in its arm's base frame
    bool moves = true; // false for a body fixed in its arm's base frame, such as the base link
};

// A configuration where the arms touch: the bodies of arm 1 and of arm 2 that may be the touching
// ones (one body on each side when the touching pair is known, or when each arm is one body).
// Two fixed bodies never explain a contact: their distance is the same in every configuration,
// so they would explain all of them at once. They only must not penetrate
struct Contact {
    std::vector<Body> bodies1;
    std::vector<Body> bodies2;
};

// Finds X, the pose of arm 2's base in arm 1's base frame, from configurations where the arms
// touch. The signed distance between body i of arm 1 and body j of arm 2 at X is
//   f_ij(X) = min(min_p Fi(posei^-1 X posej p), min_p Fj(posej^-1 X^-1 posei p)),
// p over the surface samples of the other body. For every contact, the smallest f_ij over the
// pairs that may explain it must vanish, and no other pair may penetrate (min(f_ij, 0) must
// vanish).
// Least squares over these residuals, Levenberg-Marquardt with the update X <- X exp(delta).
// The closest pair is found at every evaluation; pairs whose bounding spheres are farther than
// the best pair so far are skipped
class ContactSolver {
public:
    struct Residual {
        double value;
        Vector6d jacobian; // d value(X exp(delta)) / d delta at delta = 0
        int body1;         // the pair: indices in the contact's bodies1 and bodies2
        int body2;
    };

    explicit ContactSolver(std::vector<Contact> contacts);

    // The touching pair's residual first, then one per penetrating pair of fixed bodies
    std::vector<Residual> residuals(const SE3& X, int t) const;
    double cost(const SE3& X) const; // sum of squared residuals
    SE3 solve(SE3 X, int maxIterations = 100, double tolerance = 1e-10) const;

private:
    Residual residual(const SE3& X, const Body& body1, const Body& body2) const;
    std::vector<Residual> residuals(const SE3& X) const; // of every contact

    std::vector<Contact> m_contacts;
};

} // namespace dabfc

#endif // DABFC_CONTACT_SOLVER_H_
