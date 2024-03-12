#pragma once


#include <thrust/device_ptr.h>
#include <types.hpp>
#include <vector>
#include <vector>
#include <glm/glm.hpp>

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

            std::vector<glm::fvec2> last_reduced_state_trajectory() override;
#endif

             ~MppiController_Impl() override;

        private:
            /**
             * num_samples x num_timesteps x actions_dims device tensor. Used to store action brownians,
             * perturbations, and action trajectories at different points in the algorithm.
             */
            thrust::device_vector<float> m_action_trajectories;

            /**
            * num_samples x num_timesteps array of costs to go. Used for action weighting.
            */
            thrust::device_vector<float> m_cost_to_gos;


            /**
             * num_timesteps x action_dims array. Best-guess action trajectory to which perturbations are added.
             */
            thrust::device_vector<DeviceAction> m_last_action_trajectory;
#ifdef PUBLISH_STATES
            DeviceAction m_last_action;
            std::mutex m_last_action_trajectory_mutex;

            /**
             * State trajectories generated from curr_state and action trajectories. Sent to display when enabled
             */
            thrust::device_vector<float> m_state_trajectories;
            State m_last_curr_state;
            std::mutex m_state_trajectories_mutex;
#endif

            curandGenerator_t m_rng;

            void generate_brownians();

            /**
             * Retrieves action based on cost to go using reduction.
             * @returns Action
             */
            thrust::device_vector<DeviceAction> reduce_actions();

            /**
             * Calculates costs to go
             */
            void populate_cost();

        };
    }
}