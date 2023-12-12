#pragma once

#include "types.hpp"

namespace controls {
    namespace interface {
        class environment {
        public:
            void update_spline(const spline_msg &msg);
            void update_slam(const slam_msg &msg);
            void update_gps(const gps_msg &msg);


        };
    }
}