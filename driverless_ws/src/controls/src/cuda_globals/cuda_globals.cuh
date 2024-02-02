#pragma once

#include <state/state_estimator.cuh>

#include "constants.hpp"


namespace controls {
    namespace cuda_globals {

        // host symbols (may still point to device)

        extern cudaTextureDesc spline_texture_desc;
        extern cudaTextureObject_t spline_texture_object;
        extern bool spline_texture_created;


        // device symbols

        extern __constant__ const float perturbation_std[action_dims * action_dims];

    }
}
