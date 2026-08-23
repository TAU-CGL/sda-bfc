#pragma once

#include <Eigen/Dense>

#include "geometry.hpp"

namespace sda_bfc {

    template <int d>
    class ForwardKinematics {
    public:
        using JointSpaceConfig = Eigen::Vector<double, d>;
        using DHParameter = Eigen::Vector<double, d>;
        using LinkRadii = Eigen::Vector<double, d + 1>;

        ForwardKinematics(DHParameter dhD, DHParameter dhA, DHParameter dhAlpha, LinkRadii linkRadii) :
            dhD(dhD), dhA(dhA), dhAlpha(dhAlpha), linkRadii(linkRadii) {
        }

        double getLinkRadius(int linkIndex) const {
            return linkRadii[linkIndex];
        }

        static SE3 dhTransform(double theta, double dist, double a, double alpha) {
            double ct = cos(theta), st = sin(theta);
            double ca = cos(alpha), sa = sin(alpha);
            SE3 T;
            T <<
                ct, -st * ca, st * sa, a * ct,
                st, ct * ca, -ct * sa, a * st,
                0.0, sa, ca, dist,
                0.0, 0.0, 0.0, 1.0;
            return T;
        }

        SE3 getCylinderTransform(int linkIndex, JointSpaceConfig q, double zOffset = 0.007) {
            SE3 T; T.setIdentity();
            for (int i = 0; i < linkIndex; i++) {
                T = T * dhTransform(q[i], dhD[i], dhA[i], dhAlpha[i]);
            }
            T.block<3,1>(0,3) += zOffset * T.block<3,1>(0,2);
            return T * Z_TO_X;
        }

    private:
        DHParameter dhD, dhA, dhAlpha;
        LinkRadii linkRadii;

        static inline const SE3 Z_TO_X = (
            SE3() << 
                0, 0, 1, 0,  
                0, 1, 0, 0,  
                -1, 0, 0, 0,  
                0, 0, 0, 1).finished(); 
    };

}