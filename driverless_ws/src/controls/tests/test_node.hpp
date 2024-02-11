#pragma once

#include "rclcpp/rclcpp.hpp"
#include "std_msgs/msg/string.hpp"

namespace controls {
    namespace tests {

        class TestNode : public rclcpp::Node {
        public:
            TestNode ();

        private:
            void print_message(const interfaces::msg::ControlAction& msg);
            void publish_spline();

            rclcpp::Subscription<interfaces::msg::ControlAction>::SharedPtr m_subscriber;
            rclcpp::Publisher<interfaces::msg::SplineFrameList>::SharedPtr m_spline_publisher;
            rclcpp::TimerBase::SharedPtr m_spline_timer;
        };

    }
}

