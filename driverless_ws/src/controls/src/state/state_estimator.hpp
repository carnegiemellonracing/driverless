#pragma once

#include <types.hpp>
#include <glm/fwd.hpp>


namespace controls {
    namespace state {

        class StateEstimator {
        public:
            static std::shared_ptr<StateEstimator> create();

            virtual void on_spline(const SplineMsg& spline_msg) =0;
            virtual void on_state(const StateMsg& state_msg) =0;

            virtual void sync_to_device() =0;

            virtual std::vector<glm::fvec2> get_spline_frames() const =0;

            virtual ~StateEstimator() = default;
        };

    }
}
