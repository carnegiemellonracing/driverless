#include "cuda_constants.cuh"

namespace controls {
    __constant__ float maximum_speed_ms;

    void set_maximum_speed(float maximum_speed) {
        cudaMemcpyToSymbol(maximum_speed_ms, &maximum_speed, sizeof(float));
    }
}