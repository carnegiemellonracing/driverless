#pragma once

#include <types.hpp>
#include <vector>
#include <vector>
#include <glm/glm.hpp>


namespace controls {
    namespace mppi {
        class MppiController {
        public:
            static std::shared_ptr<MppiController> create(std::mutex& mutex, LoggerFunc logger = no_log);

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
