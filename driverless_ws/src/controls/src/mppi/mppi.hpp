#pragma once

#include <types.hpp>
#include <vector>
#include <vector>
#include <glm/glm.hpp>


namespace controls {
    namespace mppi {
        /**
         * @brief MPPI Controller!
         * In a nutshell, given:
         * - current inertial state in @ref cuda_globals::curr_state
         * - a curvilinear lookup table in @ref cuda_globals::curv_frame_lookup_tex
         * - an existing control action trajectory from the previous run of MPPI
         *
         * it calculates the optimal control action to minimize a cost function.
         *
         * Implementation is in @ref MppiController_Impl.
         *
         * Refer to the @rst :doc:`explainer </source/explainers/mppi_algorithm>` @endrst for a more detailed overview.
         *
         */
        class MppiController {
        public:
            /**
             * @brief Essentially serves as a named constructor that grants ownership of a MPPIController to the caller.
             * This is necessary because MPPIController is an abstract base class (a pointer is needed)
             *
             * @param[in] mutex Reference to the mutex that mppi will use
             * @param[in] logger Function object of logger (string to void)
             * @return Pointer to the created MPPIController
             */
            static std::shared_ptr<MppiController> create(std::mutex& mutex, LoggerFunc logger = no_log);

            /**
             * @brief Generates an action based on state and spline (both are cuda globals), returns it.
             * Tunable parameters in @file constants.hpp and @file cuda_globals/cuda_globals.cuh.
             * Virtual because it should be called through its child (implementation) object rather than directly.
             * @return The optimal action calculated by MPPI given state and spline information.
             */
            virtual Action generate_action() =0;
            virtual void hardcode_last_action_trajectory(std::vector<Action> actions) =0;
            virtual void set_logger(LoggerFunc logger) =0;

#ifdef DISPLAY
            virtual std::vector<float> last_state_trajectories(uint32_t num) =0;
            virtual std::vector<glm::fvec2> last_reduced_state_trajectory() = 0;
#endif
#ifdef DATA 
            std::vector<Action> m_last_action_trajectory_logging;
            std::vector<Action> m_percentage_diff_trajectory;
            std::vector<Action> m_averaged_trajectory;

            struct DiffStatistics {
                float mean_swangle;
                float mean_throttle;
                float max_swangle;
                float max_throttle;
            };
            DiffStatistics m_diff_statistics;
#endif
            
            virtual ~MppiController() = default;
        };

    }
}
