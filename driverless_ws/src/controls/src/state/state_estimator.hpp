#pragma once

#include <types.hpp>


namespace controls {
    namespace state {

        class StateEstimator {
        public:
            static std::unique_ptr<StateEstimator> create();

            virtual void on_spline(const SplineMsg& spline_msg) =0;
            virtual void on_slam(const SlamMsg& slam_msg) =0;

            /**
             * @brief Retrieves curvilinear state. This copies from device (which can be expensive).
             * @return Curvilinear state
             */
            virtual State get_curv_state() const =0;
        };

    }
}
