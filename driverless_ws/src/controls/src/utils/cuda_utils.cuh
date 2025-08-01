#pragma once

#include <curand.h>
#include <iostream>
#include <sstream>
#include "general_utils.hpp"
#include "constants.hpp"
#include <curand_kernel.h>


/** @brief indexes into a 3-dimensional tensor that is represented in memory as a single nested array.
 * Note: 0 <= idx.x < dim.x
 * \param tensor the array to be indexed into
 * \param dims (outermost, middle, innermost) dimensions
 * \param idx (outermost, middle, innermost) index
 * \returns address of the desired element
*/
#define IDX_3D(tensor, dims, idx) (&tensor[idx.x * dims.y * dims.z + idx.y * dims.z + idx.z])
/// Wrapper for CUDA API calls to do error checking
#define CUDA_CALL(x) (cuda_assert(x, __FILE__, __LINE__))
/// Wrapper for CURAND API calls to do error checking
#define CURAND_CALL(x) (curand_assert(x, __FILE__, __LINE__))


namespace controls {

    /**
     * @brief 3D pitched tensor
     */
    template<typename T>
    struct cudaTensor3D {
        /// @brief Pointer to the data */
        cudaPitchedPtr pitched_ptr;

        //* @brief Extent in bytes */
        cudaExtent extent;
    };

    /// Converts an opaque error into a human-readable string
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

    /// Checks for CUDA errors and prints them to stderr
    static void cuda_assert(cudaError_t code, const char *file, int line, bool abort=true)
    {
        if (code != cudaSuccess)
        {
            std::stringstream error_stream;
            error_stream << "cuda assert: " << cudaGetErrorString(code) << " (" << code << "). Location: " << file << ":" << line << "\n" << std::endl;
            if (abort) {
                throw ControllerError(error_stream.str());
            }
        }
    }

    /// Checks for CURAND errors and prints them to stderr
    static void curand_assert(curandStatus_t code, const char *file, int line, bool abort=true)
    {
        if (code != CURAND_STATUS_SUCCESS)
        {
            std::stringstream error_stream;
            error_stream << "curand assert: " << curandGetErrorString(code) << " (" << code << "). Location: " << file << ":" << line << "\n" << std::endl;
            if (abort) {
                throw ControllerError(error_stream.str());
            }
        }
    }

    /// Checks for NaN values inside the input vector.
    __host__ __device__ static bool any_nan(const float* vec, size_t n) {
        for (size_t i = 0; i < n; i++) {
            if (std::isnan(vec[i])) {
                return true;
            }
        }
        return false;
    }

    /// Arbitrary dot product function.
    template<typename T>
    __host__ __device__ T dot(const T* a, const T* b, size_t n) {
        T res {};
        for (size_t i = 0; i < n; i++) {
            res += a[i] * b[i];
        }
        return res;
    }

    /// Pretty printing function for a 3D tensor
    template<typename T>
    static void print_tensor(T tensor, dim3 dims) {
        for (uint i = 0; i < dims.x; i++) {
            for (uint j = 0; j < dims.y; j++) {
                std::cout << "{ ";
                for (uint k = 0; k < dims.z; k++) {
                    std::cout << *IDX_3D(tensor, dims, dim3(i, j, k)) << " ";
                }
                std::cout << "} ";
            }
            std::cout << std::endl;
        }
    }

    /// Pretty printing function for a 1D vector
    __device__ static void printf_vector(const float vec[], size_t n) {
        printf("{ ");
        for (size_t i = 0; i < n; i++) {
            printf("%f ", vec[i]);
        }
        printf(" }\n");
    }

    /// Forces a value to be within a certain range
    template<typename T>
    __host__ __device__ static T clamp(T n, T low, T high) {
        return n > high ? high : n < low ? low : n;
    }

    __device__ static float clamp_uniform(float n, float low, float high, curandState* state) {
        if (n < low || n > high) {
            return curand_uniform(state) * (high - low) + low;
        } else {
            return n;
        }
    }

    /// Converts degrees to radians
    constexpr float radians(float degrees) {
        return degrees * M_PI / 180;
    }
}