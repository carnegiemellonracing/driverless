/**
 * @file model.cuh
 * @brief Slipless dynamics model for MPPI and state estimation.
 *
 * For more information, see the @rst :doc:`overview </source/explainers/slipless_model>`. @endrst
 */

#pragma once
#include <constants.hpp>
#include "slipless_model.cuh"
#include <utils/cuda_utils.cuh>

namespace controls {
    namespace model {
        namespace slipless_swangle {
            static float calc_swangle(const float curr_swangle, const float requested_swangle, float timestep) {
                float delta_swangle = (requested_swangle - curr_swangle) / actuator_angular_speed;
                // (theta0 - theta1) / (theta/t) = t

                float swangle_ = curr_swangle + delta_swangle * timestep;

                return swangle_;
            }


            /**
             * @brief Slipless dynamics model with swangle. This is the core function that calculates the next state given the current state and action. Can be used in-place (i.e. next_state = state).
             * @param[in] state Current state of the car.
             * @param[in] action Action to take.
             * @param[out] next_state Next state of the car.
             * @param[in] timestep Model time step in seconds.
             */
            __host__ __device__ static void dynamics(const float state[], const float action[], float next_state[], float timestep) {
                const float swangle = action[action_swangle_idx];
                const float curr_swangle = state[state_swangle_request_idx];
                float swangle_ = calc_swangle(curr_swangle, swangle, timestep);

                const float state_tmp[4] = {
                    state[state_x_idx],
                    state[state_y_idx],
                    state[state_yaw_idx],
                    state[state_speed_idx],
                };

                const float action_tmp[2] = {
                    action[action_swangle_idx],
                    action[action_torque_idx]
                };

                float next_state_tmp[4];

                slipless::dynamics(state_tmp, action_tmp, next_state_tmp, timestep);

                memcpy(next_state, next_state_tmp, sizeof(next_state_tmp));

                next_state[state_swangle_request_idx] = swangle_;
            }
        }
    }
}