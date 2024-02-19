#pragma once

#include "rclcpp/rclcpp.hpp"
#include "std_msgs/msg/string.hpp"

namespace controls {
    namespace tests {

        class TestNode : public rclcpp::Node {
        public:
            TestNode ();

        private:
            void on_action(const ActionMsg& msg);
            void publish_spline();
            void publish_state();

            rclcpp::Subscription<ActionMsg>::SharedPtr m_subscriber;
            rclcpp::Publisher<SplineMsg>::SharedPtr m_spline_publisher;
            rclcpp::Publisher<StateMsg>::SharedPtr m_state_publisher;
            rclcpp::TimerBase::SharedPtr m_spline_timer;

            // thomas model state
            std::array<double, 13> m_world_state {};
        };

    }
}

