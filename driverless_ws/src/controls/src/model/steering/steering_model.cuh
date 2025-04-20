/**
 * @file model.cuh
 * @brief Slipless dynamics model for MPPI and state estimation.
 *
 * For more information, see the @rst :doc:`overview </source/explainers/slipless_model>`. @endrst
 */

#pragma once
#include <constants.hpp>
#include <model/sysid/sysid_model.cuh>

namespace controls {
    constexpr float device_controller_actuator_angular_speed = 2.0f;

    namespace model {
        namespace steering {
            // moves curr_swangle towards requested_swangle (both have the same axes)
                __host__ __device__ static float calc_swangle_integrated(const float curr_swangle, const float requested_swangle, const float timestep) {
                    int num_steps = int(floor(timestep / 0.01f)) - 1; // sim step
                    float swangle = curr_swangle;
                    for (int i = 0; i < num_steps; i++)
                    {
                        swangle += (requested_swangle - swangle) * device_controller_actuator_angular_speed * 0.01f;
                    }
                    swangle += (requested_swangle - swangle) * device_controller_actuator_angular_speed * (timestep - num_steps * 0.01f);

                    return swangle;
                }

                __host__ __device__ static float calc_swangle(const float curr_swangle, const float requested_swangle, const float timestep)
                {
                    float delta_swangle = (requested_swangle - curr_swangle) * device_controller_actuator_angular_speed;
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
                const float requested_swangle = action[action_requested_swangle_idx];
                const float curr_swangle = state[state_actual_swangle_idx];

                const float state_tmp[4] = {
                    state[state_x_idx],
                    state[state_y_idx],
                    state[state_yaw_idx],
                    state[state_speed_idx],
                };

                const float action_tmp[2] = {
                    curr_swangle,
                    action[action_torque_idx]
                };

                float next_state_tmp[4];

                controls::model::sysid::dynamics(state, action_tmp, next_state_tmp, timestep);

                next_state[state_x_idx] = next_state_tmp[state_x_idx];
                next_state[state_y_idx] = next_state_tmp[state_y_idx];
                next_state[state_yaw_idx] = next_state_tmp[state_yaw_idx];
                next_state[state_speed_idx] = next_state_tmp[state_speed_idx];
                const float swangle = calc_swangle(curr_swangle, requested_swangle, timestep);

                next_state[state_actual_swangle_idx] = swangle;
            }
        }
    }
}