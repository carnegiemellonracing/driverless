#pragma once

#include "state_estimator.hpp"


namespace controls {
    namespace state {

        class StateEstimator_Impl : public StateEstimator {
        public:
            StateEstimator_Impl();

            void on_spline(const SplineMsg& spline_msg);
            void on_slam(const SlamMsg& slam_msg);

            State get_curv_state() const;
        };

    }
}