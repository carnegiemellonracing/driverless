#include "cuda_globals.cuh"

namespace controls {
    namespace cuda_globals {

        __constant__ const float perturbation_std[action_dims * action_dims] = {
            1, 0, 0,
            0, 1, 0,
            0, 0, 0
        };

    }
}
