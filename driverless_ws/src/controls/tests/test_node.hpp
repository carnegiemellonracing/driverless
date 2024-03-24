#pragma once

#include <rclcpp/rclcpp.hpp>
#include <random>

namespace controls {
    namespace tests {

        class TestNode : public rclcpp::Node {
        public:
            TestNode ();

        private:
            static constexpr float jitter_std = 0.00f;

            void on_action(const ActionMsg& msg);
            void publish_spline();
            void publish_state();
            void publish_quat();
            void publish_twist();
            void publish_pose();

            SplineMsg sine_spline(float period, float amplitude, float progress, float density);

            rclcpp::Subscription<ActionMsg>::SharedPtr m_subscriber;
            rclcpp::Publisher<SplineMsg>::SharedPtr m_spline_publisher;
            rclcpp::Publisher<StateMsg>::SharedPtr m_state_publisher;
            rclcpp::Publisher<QuatMsg>::SharedPtr m_quat_publisher;
            rclcpp::Publisher<PoseMsg>::SharedPtr m_pose_publisher;
            rclcpp::Publisher<TwistMsg>::SharedPtr m_twist_publisher;

            rclcpp::TimerBase::SharedPtr m_spline_timer;

            // thomas model state
            std::array<double, 13> m_world_state {-3, 0, 0, 0, 0, 0, 0, 0, -3.0411, 0, 0, 0, 0};

            double m_time = 0;
            std::mt19937 m_rng;
            std::normal_distribution<float> m_norm_dist {0, jitter_std};
        };

    }
}

