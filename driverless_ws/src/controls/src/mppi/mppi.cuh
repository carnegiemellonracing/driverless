#pragma once

#include <thrust/device_ptr.h>
#include <types.hpp>

#include "cuda_constants.cuh"
#include "types.cuh"
#include "mppi.hpp"


namespace controls {
    namespace mppi {

        class MppiController_Impl : public MppiController {
        public:
            MppiController_Impl();

            Action generate_action() override;

#ifdef PUBLISH_STATES
            std::vector<float> last_state_trajectories() override;
#endif

            ~MppiController_Impl() override;

        private:
            /**
             * num_samples x num_timesteps x actions_dims device tensor. Used to store brownians,
             * perturbed actions, and action trajectories at different points in the algorithm.
             */
            thrust::device_vector<float> m_action_trajectories;

            /**
             * num_samples x num_timesteps device tensor. Used to store the log probability densities (calculated after
             * brownian generations), then made "to-go"
             */
            thrust::device_vector<float> m_log_prob_densities;

            /**
             * num_samples x num_timesteps array of costs to go. Used for action weighting.
             */
            thrust::device_vector<float> m_cost_to_gos;

            /**
             * num_timesteps x action_dims array. Best-guess action trajectory to which perturbations are added.
             */
            thrust::device_vector<DeviceAction> m_last_action_trajectory;

#ifdef PUBLISH_STATES
            /**
             * \brief State trajectories generated from curr_state and action trajectories. Sent to rviz when
             *        debugging.
             */
            thrust::device_vector<float> m_state_trajectories;

            std::mutex m_state_trajectories_mutex;
#endif

            curandGenerator_t m_rng;

            void generate_brownians();

            void generate_log_probability_density();

            /**
             * @brief Retrieves action based on cost to go using reduction.
             * @return Action
             */
            thrust::device_vector<DeviceAction> reduce_actions();

            /**
             * @brief Calculates costs to go
             */
            void populate_cost();

        };




    }
}