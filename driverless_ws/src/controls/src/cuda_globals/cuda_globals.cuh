#pragma once

#include <state/state_estimator.cuh>

#include "constants.hpp"


namespace controls {
    namespace cuda_globals {

        // host symbols (may still point to device)

        extern cudaTextureDesc spline_texture_desc;
        extern cudaTextureObject_t spline_texture_object;
        extern bool spline_texture_created;

        extern __constant__ float curr_state_buf1[state_dims];
        extern __constant__ float curr_state_buf2[state_dims];

        extern float* curr_state_read;
        extern float* curr_state_write;


        // device symbols

        extern __constant__ const float perturbs_incr_std[action_dims * action_dims];

    }
}
