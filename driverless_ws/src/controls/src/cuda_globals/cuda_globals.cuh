#pragma once

#include <cuda_constants.cuh>

#include "constants.hpp"


namespace controls {
    namespace cuda_globals {
        struct CurvFrameLookupTexInfo {
            float xcenter;
            float ycenter;
            float width;
        };


        // device symbols

        extern __constant__ cudaTextureObject_t curv_frame_lookup_tex;
        extern __constant__ CurvFrameLookupTexInfo curv_frame_lookup_tex_info;

        extern __constant__ size_t spline_texture_elems;

        extern __constant__ float curr_state[state_dims];

        extern __constant__ const float perturbs_incr_std[action_dims * action_dims];

        extern __constant__ const float perturbs_incr_var_inv[action_dims * action_dims];

        extern __constant__ const float action_min[action_dims];
        extern __constant__ const float action_max[action_dims];
        extern __constant__ const float action_deriv_min[action_dims];
        extern __constant__ const float action_deriv_max[action_dims];
    }
}
