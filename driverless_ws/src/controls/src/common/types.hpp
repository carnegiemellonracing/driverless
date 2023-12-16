#pragma once

#include "constants.hpp"
#include <interfaces/msg/spline_list.hpp>
#include <array>

namespace controls {
    constexpr size_t action_dims = 2;
    constexpr size_t state_dims = 13;

    using action = std::array<double, action_dims>;
    using state = std::array<double, state_dims>;

    using spline_msg = interfaces::msg::SplineList;
    using slam_msg = struct {};
    using gps_msg = struct {};

    enum class device {
        cpu,
        cuda
    };

    class controller {
    public:
        virtual action generate_action(const state &current_state) =0;

        virtual ~controller() =0;
    };
}
