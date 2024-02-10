#pragma once

#include <state/state_estimator.cuh>
#include <mutex>

#include "constants.hpp"


namespace controls {
    namespace cuda_globals {

        // host symbols (may still point to device)

        extern cudaArray_t spline_array;
        extern cudaTextureObject_t spline_texture_object;
        extern bool spline_texture_created;

        // device symbols

        extern __constant__ size_t spline_texture_elems;

        extern __constant__ float curr_state[state_dims];

        extern __constant__ const float perturbs_incr_std[action_dims * action_dims];

    }
}
