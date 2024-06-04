#pragma once

#include <math_constants.h>
#include <types.hpp>
#include <utils/cuda_utils.cuh>


namespace controls {
    namespace mppi {
        /// Control action
        struct DeviceAction {
            float data[action_dims];
        };

        /// Control action coupled with weight for reduction
        struct ActionWeightTuple {
            DeviceAction action; // weighted average action
            float log_weight; // total weight
        };

        /// Operator Overloads for Device Action
        __host__ __device__ static DeviceAction operator+ (const DeviceAction& a1, const DeviceAction& a2) {
            DeviceAction res;
            for (size_t i = 0; i < action_dims; i++) {
                res.data[i] = a1.data[i] + a2.data[i];
            }
            return res;
        }

        template<typename T>
        __host__ __device__ static DeviceAction operator* (T scalar, const DeviceAction& action) {
            DeviceAction res;
            for (size_t i = 0; i < action_dims; i++) {
                res.data[i] = scalar * action.data[i];
            }
            return res;
        }

        template<typename T>
        __host__ __device__ static DeviceAction operator* (const DeviceAction& action, T scalar) {
            return scalar * action;
        }

        template<typename T>
        __host__ __device__ static DeviceAction operator/ (const DeviceAction& action, T scalar) {
            DeviceAction res;
            for (size_t i = 0; i < action_dims; i++) {
                res.data[i] = action.data[i] / scalar;
            }
            return res;
        }
    }
}
