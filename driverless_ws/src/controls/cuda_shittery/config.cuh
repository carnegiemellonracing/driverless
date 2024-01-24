#pragma once

#include <curand.h>

constexpr size_t action_dims = 5;
constexpr size_t num_timesteps = 128;
constexpr size_t num_samples = 1024;
constexpr size_t num_perturbs = action_dims * num_timesteps * num_samples;
constexpr dim3 perturbs_dims {num_samples, num_timesteps, action_dims};

constexpr curandRngType_t rng_type = CURAND_RNG_PSEUDO_MTGP32;
constexpr unsigned long long seed = 0;
constexpr float sqrt_timestep = 1.0f;

__constant__ const float perturbs_incr_std[] = {
    1, 0, 0, 0, 0,
    0, 2, 0, 0, 0,
    0, 0, 1, 0, 0,
    0, 0, 0, 1, 0,
    0, 0, 0, 0, 10
};
