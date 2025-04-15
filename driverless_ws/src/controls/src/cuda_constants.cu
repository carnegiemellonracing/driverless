#include <cuda_constants.cuh>

namespace controls {
    __constant__ float maximum_speed_ms; // Definition in CUDA constant memory

    // Host function to update the CUDA constant
    void set_maximum_speed(float speed) {
        cudaMemcpyToSymbol(maximum_speed_ms, &speed, sizeof(float));
    }
}