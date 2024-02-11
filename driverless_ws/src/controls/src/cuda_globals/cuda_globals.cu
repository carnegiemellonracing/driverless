#include "cuda_globals.cuh"

namespace controls {
    namespace cuda_globals {

        cudaArray_t spline_array;
        cudaTextureObject_t spline_texture_object;
        bool spline_texture_created = false;

        __constant__ cudaTextureObject_t d_spline_texture_object;

        __constant__ size_t spline_texture_elems = 0;

        __constant__ float curr_state[state_dims] = {
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0
        };

        __constant__ const float perturbs_incr_std[action_dims * action_dims] = {
            1, 0, 0,
            0, 2, 0,
            0, 0, 3
        };

    }
}
