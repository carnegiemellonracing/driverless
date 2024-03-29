#pragma once

#include <curand.h>
#include <cstdio>
#include <iostream>

#include "config.cuh"

#define IDX_2D_PITCHED(tensor, idx, pitch) ((decltype(tensor))((char*)tensor + idx.y * pitch + idx.x * sizeof(std::remove_pointer<decltype(tensor)>::type)))
#define IDX_3D_PITCHED(tensor, dims, idx, pitch) ((decltype(tensor))((char*)tensor + idx.x * dims.y * pitch + idx.y * pitch + idx.z * sizeof(std::remove_pointer<decltype(tensor)>::type)))
#define IDX_3D(tensor, dims, idx) (&tensor[idx.x * dims.y * dims.z + idx.y * dims.z + idx.z])

#define CUDA_CALL(x) (cuda_assert(x, __FILE__, __LINE__))

#define CURAND_CALL(x) (curand_assert(x, __FILE__, __LINE__))

/**
 * @brief 3D pitched tensor
 */
template<typename T>
struct cudaTensor3D {
    cudaPitchedPtr pitched_ptr;

    /** @brief Extent in bytes */
    cudaExtent extent;
};

static const char* curandGetErrorString(curandStatus_t error)
{
    switch (error)
    {
        case CURAND_STATUS_SUCCESS:
            return "CURAND_STATUS_SUCCESS";

        case CURAND_STATUS_VERSION_MISMATCH:
            return "CURAND_STATUS_VERSION_MISMATCH";

        case CURAND_STATUS_NOT_INITIALIZED:
            return "CURAND_STATUS_NOT_INITIALIZED";

        case CURAND_STATUS_ALLOCATION_FAILED:
            return "CURAND_STATUS_ALLOCATION_FAILED";

        case CURAND_STATUS_TYPE_ERROR:
            return "CURAND_STATUS_TYPE_ERROR";

        case CURAND_STATUS_OUT_OF_RANGE:
            return "CURAND_STATUS_OUT_OF_RANGE";

        case CURAND_STATUS_LENGTH_NOT_MULTIPLE:
            return "CURAND_STATUS_LENGTH_NOT_MULTIPLE";

        case CURAND_STATUS_DOUBLE_PRECISION_REQUIRED:
            return "CURAND_STATUS_DOUBLE_PRECISION_REQUIRED";

        case CURAND_STATUS_LAUNCH_FAILURE:
            return "CURAND_STATUS_LAUNCH_FAILURE";

        case CURAND_STATUS_PREEXISTING_FAILURE:
            return "CURAND_STATUS_PREEXISTING_FAILURE";

        case CURAND_STATUS_INITIALIZATION_FAILED:
            return "CURAND_STATUS_INITIALIZATION_FAILED";

        case CURAND_STATUS_ARCH_MISMATCH:
            return "CURAND_STATUS_ARCH_MISMATCH";

        case CURAND_STATUS_INTERNAL_ERROR:
            return "CURAND_STATUS_INTERNAL_ERROR";
    }

    return "<unknown>";
}

static void cuda_assert(cudaError_t code, const char *file, int line, bool abort=true)
{
    if (code != cudaSuccess)
    {
        fprintf(stderr,"cuda assert: (%i) %s. Location: %s:%d\n", code, cudaGetErrorString(code), file, line);
        if (abort) exit(code);
    }
}

static void curand_assert(curandStatus_t code, const char *file, int line, bool abort=true)
{
    if (code != CURAND_STATUS_SUCCESS)
    {
        fprintf(stderr,"curand assert: %s. Location: %s:%d\n", curandGetErrorString(code), file, line);
        if (abort) exit(code);
    }
}

template<typename T>
static cudaError_t alloc_tensor3D_unpitched(cudaTensor3D<T>& tensor, dim3 dims) {
    // yes, dims.z is correct (cuda uses column major here)
    tensor.extent = make_cudaExtent(dims.z * sizeof(T), dims.y, dims.x);
    tensor.pitched_ptr = make_cudaPitchedPtr(nullptr, tensor.extent.width, tensor.extent.width, tensor.extent.height);
    return cudaMalloc(&tensor.pitched_ptr.ptr, tensor.extent.width * tensor.extent.height * tensor.extent.depth);
}

__device__ static uint3 total_idx(dim3 blockDim, uint3 blockIdx, uint3 threadIdx) {
    return {
        blockDim.x * blockIdx.x + threadIdx.x,
        blockDim.y * blockIdx.y + threadIdx.y,
        blockDim.z * blockIdx.z + threadIdx.z
    };
}

__device__ static size_t flat_block_idx(dim3 blockDim, uint3 threadIdx) {
    return threadIdx.x * blockDim.y * blockDim.z + threadIdx.y * blockDim.z + threadIdx.z;
}

template<typename T>
__device__ static void thread_parallel_memcpy(T* dst, T* src, size_t elems, dim3 blockDim, uint3 threadIdx) {
    const size_t offset = flat_block_idx(blockDim, threadIdx);
    for (size_t i = offset; i < elems; i += blockDim.x * blockDim.y * blockDim.z) {
        dst[i] = src[i];
    }
}

template<typename T>
void print_tensor_3D(T tensor, dim3 dims) {
    for (int i = 0; i < dims.x; i++) {
        for (int j = 0; j < dims.y; j++) {
            std::cout << "{ ";
            for (int k = 0; k < dims.z; k++) {
                std::cout << *IDX_3D(tensor, dims, dim3(i, j, k)) << " ";
            }
            std::cout << "} ";
        }
        std::cout << std::endl;
    }
    std::cout << std::endl;
}

template<size_t k>
struct MultiplyBy {
    __host__ __device__ size_t operator() (size_t i) const {
        return i * k;
    }
};

struct Action {
    float data[action_dims];
};

__host__ __device__ static Action operator+ (const Action& a1, const Action& a2) {
    Action res;
    for (size_t i = 0; i < action_dims; i++) {
        res.data[i] = a1.data[i] + a2.data[i];
    }
    return res;
}

template<typename T>
__host__ __device__ static Action operator* (T scalar, const Action& action) {
    Action res;
    for (size_t i = 0; i < action_dims; i++) {
        res.data[i] = scalar * action.data[i];
    }
    return res;
}

template<typename T>
__host__ __device__ static Action operator* (const Action& action, T scalar) {
    return scalar * action;
}

template<typename T>
__host__ __device__ static Action operator/ (const Action& action, T scalar) {
    Action res;
    for (size_t i = 0; i < action_dims; i++) {
        res.data[i] = action.data[i] / scalar;
    }
    return res;
}

struct AddActions {
    __host__ __device__ Action operator() (const Action& a1, const Action& a2) const {
        return a1 + a2;
    }
};

template<size_t k>
struct DivBy {
    __host__ __device__ size_t operator() (size_t i) const {
        return i / k;
    }
};

template<typename T>
struct Equal {
    __host__ __device__ bool operator() (T a, T b) {
        return a == b;
    }
};

