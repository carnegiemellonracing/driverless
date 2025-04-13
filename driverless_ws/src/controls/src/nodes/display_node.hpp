#pragma once

#include <ros_types_and_constants.hpp>

namespace controls {
    namespace nodes {
        class DisplayNode : public rclcpp::Node {
            private:
                rclcpp::Subscription<InfoMsg>::SharedPtr m_info_subscription;
                rclcpp::Subscription<Con
                
        }
    }


}