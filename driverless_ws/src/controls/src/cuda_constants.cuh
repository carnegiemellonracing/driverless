#pragma once

#include <curand.h>
#include <model/bicycle/model.cuh>
// #include <model/dummy/model.cuh>

#include "constants.hpp"

#define ONLINE_DYNAMICS_FUNC controls::model::bicycle::dynamics


namespace controls {
    constexpr curandRngType_t rng_type = CURAND_RNG_PSEUDO_MTGP32;
    // outermost to innermost dimensions
    constexpr dim3 action_trajectories_dims {num_samples, num_timesteps, action_dims};
    constexpr size_t max_spline_texture_elems = 2048;
}