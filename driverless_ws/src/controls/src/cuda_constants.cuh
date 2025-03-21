#pragma once

#include <curand.h>
#include <model/slipless/model.cuh>

#include "constants.hpp"

#define ONLINE_DYNAMICS_FUNC controls::model::slipless::dynamics


namespace controls {
    /// Used for curandCreateGenerator
    constexpr curandRngType_t rng_type = CURAND_RNG_PSEUDO_MTGP32;
    /// Dimensions of MPPI's @c m_action_trajectories, outermost to innermost dimensions
    constexpr dim3 action_trajectories_dims {num_samples, num_timesteps, action_dims};

}