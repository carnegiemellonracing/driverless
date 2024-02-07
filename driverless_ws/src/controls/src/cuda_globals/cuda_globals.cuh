#pragma once

#include <state/state_estimator.cuh>
#include <mutex>

#include "constants.hpp"


namespace controls {
    namespace cuda_globals {

        // host symbols (may still point to device)

        extern cudaArray_t spline_array;
        extern cudaTextureObject_t spline_texture_object;
        extern bool spline_texture_created;

        extern float* curr_state_read, * curr_state_write;
        extern std::mutex state_swapping_mutex;
        extern bool state_pointers_created;


        // device symbols

        extern __constant__ size_t spline_texture_elems;

        extern __constant__ float curr_state_buf1[state_dims];
        extern __constant__ float curr_state_buf2[state_dims];

        extern __constant__ const float perturbs_incr_std[action_dims * action_dims];

        void lock_and_swap_state_buffers();
    }
}
