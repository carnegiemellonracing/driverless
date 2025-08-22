#include <cstdio>
#include <cuda_runtime.h>  // Include the CUDA runtime header

int main() {
    int n; 
    cudaError_t err = cudaGetDeviceCount(&n);  // Capture return status

    if (err != cudaSuccess) {
        printf("cudaGetDeviceCount failed: %s\n", cudaGetErrorString(err));
        return 1;
    }

    printf("Found %d CUDA device%s\n", n, (n == 1 ? "" : "s"));
    return 0;
}
