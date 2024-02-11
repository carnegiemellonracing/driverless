#pragma once

#include <types.hpp>


namespace controls {
    namespace state {

        class StateEstimator {
        public:
            static std::unique_ptr<StateEstimator> create();

            virtual void on_spline(const controls::SplineMsg& spline_msg) =0;
            virtual void on_slam(const SlamMsg& slam_msg) =0;

            virtual ~StateEstimator() = default;
        };

    }
}
