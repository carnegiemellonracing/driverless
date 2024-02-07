#pragma once

#include <interfaces/msg/spline_frame_list.hpp>
#include <geometry_msgs/msg/pose2_d.hpp>
#include <array>

#include "constants.hpp"


namespace controls {
    using Action = std::array<float, action_dims>;
    using State = std::array<float, state_dims>;

    using SplineMsg = interfaces::msg::SplineFrameList;
    using SlamMsg = geometry_msgs::msg::Pose2D;

    class Controller {
    public:
        virtual Action generate_action() =0;

        virtual ~Controller() =0;
    };
}
