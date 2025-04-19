/**
 * @file model.cuh
 * @brief Slipless dynamics model for MPPI and state estimation.
 *
 * For more information, see the @rst :doc:`overview </source/explainers/slipless_model>`. @endrst
 */

#pragma once
#include <model/steering/controller_steering_model_host.h>

namespace controls {
    namespace model {
        namespace steering {



            /**
             * @brief Slipless dynamics model with swangle. This is the core function that calculates the next state given the current state and action. Can be used in-place (i.e. next_state = state).
             * @param[in] state Current state of the car.
             * @param[in] action Action to take.
             * @param[out] next_state Next state of the car.
             * @param[in] timestep Model time step in seconds.
             */
            __host__ __device__ static void dynamics(const float state[], const float action[], float next_state[], float timestep) {
                controls::model_host::steering::dynamics(state, action, next_state, timestep);
            }
        }
    }
}