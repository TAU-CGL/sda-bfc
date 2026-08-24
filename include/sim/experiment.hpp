#pragma once

#include "contact_newton.hpp"
#include "placement.hpp"

namespace sda_bfc {

    struct Experiment {
        SE3 X;
        std::vector<JointConfig> qAs, qBs;
    };

    struct ExperimentParams {
        PlacementParams placement{};
        ContactParams contact{};
        int maxPlacementAttempts = 20;
    };

    // Sample a valid placement, then generate numContacts touching joint
    // configurations in it.  A placement where contact generation fails is
    // discarded and resampled (constructive touch-feasibility check).
    inline Experiment generateExperiment(const ForwardKinematics<6>& fk,
                                         int numContacts, unsigned seed = 0,
                                         const ExperimentParams& params = {}) {
        std::mt19937 gen(seed);
        for (int attempt = 0; attempt < params.maxPlacementAttempts; attempt++) {
            SE3 X = sampleValidPlacement(gen, fk, params.placement);
            ContactGenerator generator(fk, X, params.contact);
            Experiment experiment{X, {}, {}};
            bool complete = true;
            for (int k = 0; k < numContacts; k++) {
                std::optional<ContactPose> contact = generator.generate(gen);
                if (!contact) {
                    complete = false;
                    break;
                }
                experiment.qAs.push_back(contact->qA);
                experiment.qBs.push_back(contact->qB);
            }
            if (complete) return experiment;
        }
        throw std::runtime_error("generateExperiment: no touch-feasible placement found");
    }

}
