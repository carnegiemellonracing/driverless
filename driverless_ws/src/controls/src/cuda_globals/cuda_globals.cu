#include "cuda_globals.cuh"

namespace controls {
    namespace cuda_globals {

        cudaArray_t spline_array;
        cudaTextureObject_t spline_texture_object;
        bool spline_texture_created = false;

        float* curr_state_read, * curr_state_write;
        std::mutex state_swapping_mutex;
        bool state_pointers_created = false;

        __constant__ size_t spline_texture_elems = 0;

        __constant__ float curr_state_buf1[state_dims];
        __constant__ float curr_state_buf2[state_dims];

        __constant__ const float perturbs_incr_std[action_dims * action_dims] = {
            1, 0, 0,
            0, 1, 0,
            0, 0, 1
        };


        void lock_and_swap_state_buffers() {
            std::lock_guard<std::mutex> guard {state_swapping_mutex};

            std::swap(curr_state_read, curr_state_write);
        }

    }
}
