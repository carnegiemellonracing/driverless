#pragma once

#include <iosfwd>
#include <iosfwd>
#include <rclcpp/rclcpp.hpp>
#include <random>
#include <vector>
#include <vector>
#include <glm/common.hpp>
#include <glm/common.hpp>
#include <glm/common.hpp>
#include <glm/common.hpp>

namespace controls {
    namespace tests {

        class TestNode : public rclcpp::Node {
        public:
            TestNode (uint32_t seed);

        private:
            enum class SegmentType {
                NONE,
                STRAIGHT,
                ARC
            };

            static constexpr float jitter_std = 0.00f;
            static constexpr float straight_after_arc_prob = 0.75f;
            static constexpr float min_radius = 3.0f;
            static constexpr float max_radius = 10.0f;
            static constexpr float min_arc_rad = M_PI / 4.0f;
            static constexpr float max_arc_rad = M_PI;
            static constexpr float min_straight = 10.0f;
            static constexpr float max_straight = 30.0f;

            void on_action(const ActionMsg& msg);
            void publish_spline();
            void publish_quat();
            void publish_twist();
            void publish_pose();

            void next_segment();
            std::vector<glm::fvec2> arc_segment(float radius, glm::fvec2 start_pos, float start_heading, float end_heading);
            std::vector<glm::fvec2> straight_segment(glm::fvec2 start, float length, float heading);

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
            std::normal_distribution<float> m_norm_dist {0, 1};
            std::uniform_real_distribution<float> m_uniform_dist {0, 1};

            SegmentType m_last_segment_type = SegmentType::NONE;
            glm::fvec2 m_spline_end_pos = {0, 0};
            glm::fvec2 m_spline_mid_pos = {0, 0};
            float m_spline_end_heading = 0;
            std::vector<glm::fvec2> m_segment1;
            std::vector<glm::fvec2> m_segment2;
        };

    }
}

