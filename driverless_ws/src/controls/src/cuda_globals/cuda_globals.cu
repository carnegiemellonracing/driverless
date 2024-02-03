#include "cuda_globals.cuh"

namespace controls {
    namespace cuda_globals {

        cudaTextureDesc spline_frame_texture_desc;
        cudaTextureObject_t spline_texture_object;
        bool spline_texture_created = false;

        __constant__ const float perturbs_incr_std[action_dims * action_dims] = {
            1, 0, 0,
            0, 1, 0,
            0, 0, 0
        };

    }
}
