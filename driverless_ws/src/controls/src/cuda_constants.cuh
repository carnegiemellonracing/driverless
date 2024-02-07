#pragma once

#include "constants.hpp"
#include "curand.h"

namespace controls {
    constexpr curandRngType_t rng_type = CURAND_RNG_PSEUDO_MTGP32;
    constexpr dim3 action_trajectories_dims {num_samples, num_timesteps, action_dims};
    constexpr size_t max_spline_texture_elems = 2048;
}