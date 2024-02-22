#pragma once

#include <rclcpp/rclcpp.hpp>

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
            std::array<double, 13> m_world_state {1, 0, 0, 0, 0, 0, 0, 0, -3.0411, 0, 0, 0, 0};
        };

    }
}

