#pragma once

#include <thrust/device_ptr.h>
#include <types.hpp>

#include "cuda_constants.cuh"
#include "types.cuh"
#include "mppi.hpp"


namespace controls {
    namespace mppi {

        __device__ static void model(const float state[], const float action[], float state_out[], float timestep) {
            float state_dot[action_dims];
            ONLINE_DYNAMICS_FUNC(state, action ,state_dot);
            for (uint8_t i = 0; i < state_dims; i++) {
                state_out[i] += state_dot[i] * timestep;
            }
        }

        __device__ static float cost(float state[]) {
            // sum the vector of state
            float sum = 0;
            for (size_t i = 0; i < state_dims; i++) {
                sum += state[i];
            }
            return sum;
        }

        class MppiController_Impl : public MppiController {
        public:
            MppiController_Impl();

            Action generate_action() override;

            ~MppiController_Impl() override;

        private:
            /**
             * num_samples x num_timesteps x actions_dims device tensor. Used to store action brownians,
             * perturbations, and action trajectories at different points in the algorithm.
             */
            thrust::device_ptr<float> m_action_trajectories;

            /**
             * num_samples x num_timesteps array of costs to go. Used for action weighting.
             */
            thrust::device_ptr<float> m_cost_to_gos;

            /**
             * num_timesteps x action_dims array. Best-guess action trajectory to which perturbations are added.
             */
            thrust::device_ptr<float> m_last_action_trajectory;


            void generate_brownians();

            /**
             * @brief Retrieves action based on cost to go using reduction.
             * @return Action
             */
            DeviceAction reduce_actions();

            /**
             * @brief Calculates costs to go
             */
            void populate_cost();

        };




    }
}