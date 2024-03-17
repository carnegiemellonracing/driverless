#include "cuda_globals.cuh"

namespace controls {
    namespace cuda_globals {
        __constant__ cudaTextureObject_t curv_frame_lookup_tex;
        __constant__ CurvFrameLookupTexInfo curv_frame_lookup_tex_info;

        __constant__ size_t spline_texture_elems = 0;

        __constant__ float curr_state[state_dims] = {10, 0, 0, 0, 0, 0, 0, 0, 0, 0};


        // NOTE:WHEN CHANGING ENSURE YOU ALSO CHANGE MAGIC MATRIX AND MAGIC NUMBER
        __constant__ const float perturbs_incr_std[action_dims * action_dims] = {
            0.25, 0,
            0, 1000
        };

        // covariance matrix is perturbs_incr_std squared:
        // (1/16, 0)
        // (0, 1E6)

        // NOTE: magic_matrix and magic_number are dependent on perturbs_incr_std
        // hard coded for efficiency reasons (determinant/inverse/sqrt can't be calculated at compile time)
        __constant__ const float magic_matrix[action_dims * action_dims] = {
            16, 0,
            0, 1E-6
        };

        //-ln(sqrt((2pi)^action_dims * det(covariance matrix)))
        __constant__ const float magic_number= 3.68358385145;

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
