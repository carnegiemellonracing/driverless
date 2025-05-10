#include "cuda_globals.cuh"
#include "utils/cuda_utils.cuh"
#include "constants.hpp"

namespace controls {
    namespace cuda_globals {
        __constant__ cudaTextureObject_t curv_frame_lookup_tex;
        __constant__ CurvFrameLookupTexInfo curv_frame_lookup_tex_info;

        __constant__ float curr_state[state_dims] = {0, 0, 0, 0};

        //TODO: throttle vs torque (write down units, maybe also add in mapping to IRL)
        constexpr float swangle_swangle_std = 0.1; ///< Standard deviation for swangle distribution
        constexpr float torque_torque_std = 10; ///< Standard deviation for throttle distribution

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
            min_swangle_rad, min_torque
        };
        __constant__ const float action_max[action_dims] = {
            max_swangle_rad, max_torque
        };

        __constant__ const float action_deriv_min[action_dims] = {
            -max_swangle_rate, -max_torque_rate
        };
        __constant__ const float action_deriv_max[action_dims] = {
            max_swangle_rate, max_torque_rate
        };
    }
}
