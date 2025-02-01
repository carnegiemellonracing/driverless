#pragma once

#include <cuda_constants.cuh>

#include "constants.hpp"


namespace controls {
    namespace cuda_globals {
        struct CurvFrameLookupTexInfo {
            float xcenter; ///< x-coordinate of the center of the lookup texture
            float ycenter; ///< y-coordinate of the center of the lookup texture
            float width; ///< scale factor for lookup texture
        };

        // device symbols
        extern __constant__ cudaTextureObject_t left_curv_frame_lookup_tex; ///< this is a pointer to the CUDA texture
        extern __constant__ cudaTextureObject_t right_curv_frame_lookup_tex; ///< this is a pointer to the CUDA texture
        extern __constant__ CurvFrameLookupTexInfo curv_frame_lookup_tex_info; ///< before sampling into texture, need to apply the same transformation to world pose

        /// State information is stored here. Written to by state_estimator and read by mppi_controller.
        extern __constant__ float curr_state[state_dims]; ///< \f$x_0\f$

        /// Cholesky factor of covariance matrix (assuming swangle and throttle are uncorrelated)
        extern __constant__ const float perturbs_incr_std[action_dims * action_dims];

        /// Inverse of covariance matrix
        extern __constant__ const float perturbs_incr_var_inv[action_dims * action_dims];


        extern __constant__ const float action_min[action_dims]; ///< Minimum control action request
        extern __constant__ const float action_max[action_dims]; ///< Maximum control action request
        extern __constant__ const float action_deriv_min[action_dims]; ///< Minimum perturbation to control action
        extern __constant__ const float action_deriv_max[action_dims]; ///< Maximum perturbation to control action
    }
}

//TODO: consider splitting tunable parameters and storage variables YES