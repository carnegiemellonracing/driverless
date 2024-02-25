#include "cuda_globals.cuh"

namespace controls {
    namespace cuda_globals {

        float4* spline_texture_buf;
        cudaTextureObject_t spline_texture_object;
        bool spline_texture_created = false;

        float curr_world_state_host[state_dims] = {};

        __constant__ cudaTextureObject_t d_spline_texture_object;

        __constant__ size_t spline_texture_elems = 0;

        __constant__ float curr_state[state_dims] = {10, 0, 0, 0, 0, 0, 0, 0, 0, 0};

        __constant__ const float perturbs_incr_std[action_dims * action_dims] = {
            0.25, 0,
            0, 100
        };

        __constant__ const float action_min[action_dims] = {
            -0.5, -1000
        };
        __constant__ const float action_max[action_dims] = {
            0.5, 1000
        };

        __constant__ const float action_deriv_min[action_dims] = {
            -1.0, -std::numeric_limits<float>::infinity()
        };
        __constant__ const float action_deriv_max[action_dims] = {
            1.0, std::numeric_limits<float>::infinity()
        };
    }
}
