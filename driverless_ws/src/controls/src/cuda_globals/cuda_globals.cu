#include "cuda_globals.cuh"
#include "utils/cuda_utils.cuh"


namespace controls {
    namespace cuda_globals {
        __constant__ cudaTextureObject_t curv_frame_lookup_tex;
        __constant__ CurvFrameLookupTexInfo curv_frame_lookup_tex_info;

        __constant__ size_t spline_texture_elems = 0;

        __constant__ float curr_state[state_dims] = {0, 0, 0, 0};


        constexpr float swangle_swangle_std = 0.1;
        constexpr float torque_torque_std = 10;
        // NOTE:WHEN CHANGING ENSURE YOU ALSO CHANGE MAGIC MATRIX AND MAGIC NUMBER
        __constant__ const float perturbs_incr_std[action_dims * action_dims] = {
            swangle_swangle_std, 0,
            0, torque_torque_std
        };


        // TODO: make sigma inverse more general (for non diagonal A)
        __constant__ const float perturbs_incr_var_inv[action_dims * action_dims] = {
            1 / (swangle_swangle_std * swangle_swangle_std), 0,
            0, 1 / (torque_torque_std * torque_torque_std)
        };

        __constant__ const float action_min[action_dims] = {
            -radians(19), -saturating_motor_torque
        };
        __constant__ const float action_max[action_dims] = {
            radians(19), saturating_motor_torque
        };

        __constant__ const float action_deriv_min[action_dims] = {
            -1.0, -std::numeric_limits<float>::infinity()
        };
        __constant__ const float action_deriv_max[action_dims] = {
            1.0, std::numeric_limits<float>::infinity()
        };
    }
}
