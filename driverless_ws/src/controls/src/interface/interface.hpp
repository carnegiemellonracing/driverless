#pragma once

#include <common/types.hpp>

namespace controls {
    namespace interface {
        class environment {
        public:
            virtual void update_spline(const spline_msg &msg) =0;
            virtual void update_slam(const slam_msg &msg) =0;
            virtual void update_gps(const gps_msg &msg) =0;

            virtual state get_state() const =0;

            static std::unique_ptr<environment> create_environment(device dev);
        };
    }
}