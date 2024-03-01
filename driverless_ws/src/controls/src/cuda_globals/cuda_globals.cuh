#pragma once

#include <cuda_constants.cuh>
#include <state/state_estimator.cuh>

#include "constants.hpp"


namespace controls {
    namespace cuda_globals {

        // host symbols (may still point to device)

        extern float4* spline_texture_buf;
        extern cudaTextureObject_t spline_texture_object;
        extern bool spline_texture_created;

        extern float curr_world_state_host[state_dims];

        // device symbols

        extern __constant__ cudaTextureObject_t d_spline_texture_object;

        extern __constant__ size_t spline_texture_elems;

        extern __constant__ float curr_state[state_dims];

        extern __constant__ const float perturbs_incr_std[action_dims * action_dims];

        extern __constant__ const float action_min[action_dims];
        extern __constant__ const float action_max[action_dims];
        extern __constant__ const float action_deriv_min[action_dims];
        extern __constant__ const float action_deriv_max[action_dims];
    }
}
