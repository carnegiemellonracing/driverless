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
            /**
             * @brief Construct a new MppiController_Impl object
             *
             * @param mutex Mutex to prevent multiple simultaneous Thrust calls (generate_action and display)
             * @param logger Logger function to log messages
             */
            MppiController_Impl(std::mutex& mutex, LoggerFunc logger);

            /**
             * @brief Generates an action based on state and spline (both are cuda globals), returns it.
             * Tunable parameters in @file constants.hpp and @file cuda_globals.cuh.
             *
             * @return The optimal action calculated by MPPI given sensor information
             */
            Action generate_action() override;
            void hardcode_last_action_trajectory(std::vector<Action> actions) override;

            void set_logger(LoggerFunc logger) override;


#ifdef DISPLAY
            std::vector<float> last_state_trajectories(uint32_t num) override;

            std::vector<glm::fvec2> last_reduced_state_trajectory() override;
#endif

            ~MppiController_Impl() override;

        private:
            // placing the member variables first to help with understanding what each member function interacts with.

            /// num_timesteps x action_dims device tensor. Best-guess action trajectory to which perturbations are added.
            /// The only device tensor that is retained between each call to generate_action.
            thrust::device_vector<DeviceAction> m_last_action_trajectory;
            /// num_samples x num_timesteps x actions_dims (num_action_trajectories) device tensor.
            /// Used to store brownians, perturbed actions, and action trajectories at different points in the algorithm.
            // TODO: what's the diff between perturbed actions and action trajectories?
            thrust::device_vector<float> m_action_trajectories;
            /// num_samples x num_timesteps device tensor. Stores costs to go.
            thrust::device_vector<float> m_cost_to_gos;
            /// num_samples x num_timesteps device tensor. Stores the log probability densities (calculated after
            /// brownian generations), then made "to-go". Used for importance sampling.
            thrust::device_vector<float> m_log_prob_densities;
            /// num_samples x num_timesteps device tensor. Used to store the control actions and their corresponding
            /// weights (based on cost and log probability density).
            thrust::device_vector<ActionWeightTuple> m_action_weight_tuples;


            /**
             * @brief Populates @c m_action_trajectories with samples taken from a zero-centered Brownian distrubtion.
             * Computed for each sample trajectory in parallel (num_samples wide).
             */
            void generate_brownians();
            /**
             * @brief Generates log probability densities from the Brownian samples.
             * Reads from @c m_action_trajectories and writes to @c m_log_prob_densities.
             * Computed over each "control action" in parallel (num_samples * num_timesteps wide).
             */
            void generate_log_probability_density();

            /**
             * @brief Calculates costs to go, storing them in @c m_cost_to_gos.
             * Heavy lifting function.
             * To calculate cost, reads from global state and spline information,
             * and simulates state into the future using the dynamics model.
             * Side effect of modifying @c m_action_trajectories by adding @c m_last_action_trajectory.
             * Side effect of making m_log_prob_densities "to-go".
             * Reads/writes to every device vector except @c m_action_weight_tuples.
             * Computed over each sample trajectory in parallel (num_samples wide).
             */
            //TODO: any way to make this more atomic without compromising speed?
         /// rollout states and populate cost
            void populate_cost();

            /**
             * @brief Populates m_action_weight_tuples in timestep-major order with actions and their associated weights.
             * All samples at time 0, all samples at time 1, etc.
             * This is transposed from m_cost_to_gos (which is sample-major order) so that the reduction has good
             * locality.
             * Reads from the corresponding indexes in m_action_trajectories, m_cost_to_gos and m_log_prob_densities,
             * bundles them, then places the tuple in the new position in m_action_weight_tuples.
             * Computed over each control action/cost in parallel (num_samples * num_timesteps wide)
             */
            void generate_action_weight_tuples();

            /**
             * @brief For every timestep, reduces the samples together using a weighted average.
             * Reads from m_action_weight_tuples.
             * @returns Optimal control action trajectory based on state and spline information.
             */
            thrust::device_vector<DeviceAction> reduce_actions();

            /// last action generated by the controller. Used for ML-like momentum and display purposes.
            DeviceAction m_last_action;

#ifdef DISPLAY

            /**
             * @brief State trajectories generated from curr_state and action trajectories. Sent to display when enabled.
             */
            thrust::device_vector<float> m_state_trajectories;
            State m_last_curr_state;
#endif



            /// Pseudo random generator used to generate brownian perturbations.
            curandGenerator_t m_rng;
            /// Logger for debugging. Can be attached to ROS or echo to std::cerr
            LoggerFunc m_logger;

            /// Mutex to prevent multiple simultaneous Thrust calls (generate_action and display)
            std::mutex& m_mutex;
        };
    }
}