#pragma once

#include <types.hpp>
#include <vector>
#include <vector>
#include <glm/glm.hpp>


namespace controls {
    namespace mppi {
        class MppiController {
        public:
            /**
             * @brief Essentially serves as a named constructor that grants ownership of a MPPIController to the caller.
             * This is necessary because MPPIController is an abstract base class (a pointer is needed)
             *
             * @param[in] mutex Reference to the mutex that mppi will use //TODO: who else uses it?
             * @param[in] logger Function object of logger (string to void) //TODO: is function object correct?
             */
            static std::shared_ptr<MppiController> create(std::mutex& mutex, LoggerFunc logger = no_log);

            /**
             * @brief Generates an action based on state and spline (both are cuda globals), returns it.
             * Tunable parameters in @file constants.hpp and @file cuda_globals.cuh.
             * Virtual because it should be called through its child (implementation) object rather than directly.
             *
             * @return The optimal action calculated by MPPI given state and spline information.
             */
            virtual Action generate_action() =0;
            virtual void set_logger(LoggerFunc logger) =0;

#ifdef DISPLAY
            virtual std::vector<float> last_state_trajectories(uint32_t num) =0;
            virtual std::vector<glm::fvec2> last_reduced_state_trajectory() = 0;
#endif

            virtual ~MppiController() = default;
        };

    }
}
