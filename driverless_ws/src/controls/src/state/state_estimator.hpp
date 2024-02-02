#pragma once

#include <types.hpp>


namespace controls {
    namespace state {

        class StateEstimator {
        public:
            void on_spline(const SplineMsg& spline_msg);
            void on_slam(const SlamMsg& slam_msg);

            /**
             * @brief Retrieves curvilinear state. This copies from device (which can be expensive).
             * @return Curvilinear state
             */
            State get_curv_state() const;
        };

    }
}
