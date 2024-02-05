#pragma once

#include <curand.h>
#include <iostream>

#include "constants.hpp"

#define IDX_3D(tensor, dims, idx) (&tensor[idx.x * dims.y * dims.z + idx.y * dims.z + idx.z])
#define CUDA_CALL(x) (cuda_assert(x, __FILE__, __LINE__))
#define CURAND_CALL(x) (curand_assert(x, __FILE__, __LINE__))


namespace controls {

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
            fprintf(stderr,"cuda assert: %s. Location: %s:%d\n", cudaGetErrorString(code), file, line);
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
    __host__ __device__ T dot(const T* a, const T* b, size_t n) {
        T res {};
        for (size_t i = 0; i < n; i++) {
            res += a[i] * b[i];
        }
        return res;
    }

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

}