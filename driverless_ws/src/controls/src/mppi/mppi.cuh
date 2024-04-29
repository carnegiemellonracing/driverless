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
            MppiController_Impl(std::mutex& mutex, LoggerFunc logger);

            Action generate_action() override;
            void set_logger(LoggerFunc logger) override;


#ifdef DISPLAY
            std::vector<float> last_state_trajectories(uint32_t num) override;

            std::vector<glm::fvec2> last_reduced_state_trajectory() override;
#endif

             ~MppiController_Impl() override;

        private:
            void generate_brownians();

            void generate_log_probability_density();

            /**
             * @brief Calculates costs to go
             */
            void populate_cost();

            void generate_action_weight_tuples();

            /**
             * Retrieves action based on cost to go using reduction.
             * @returns Action
             */
            thrust::device_vector<DeviceAction> reduce_actions();

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

            thrust::device_vector<ActionWeightTuple> m_action_weight_tuples;

            /**
             * num_samples x num_timesteps array of costs to go. Used for action weighting.
             */
            thrust::device_vector<float> m_cost_to_gos;


            /**
             * num_timesteps x action_dims array. Best-guess action trajectory to which perturbations are added.
             */
            thrust::device_vector<DeviceAction> m_last_action_trajectory;
            DeviceAction m_last_action;

#ifdef DISPLAY

            /**
             * State trajectories generated from curr_state and action trajectories. Sent to display when enabled
             */
            thrust::device_vector<float> m_state_trajectories;
            State m_last_curr_state;
#endif

            curandGenerator_t m_rng;

            LoggerFunc m_logger;

            std::mutex& m_mutex;
        };
    }
}