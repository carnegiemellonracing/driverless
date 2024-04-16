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
            static constexpr float new_seg_dist = 10.0f;
            static constexpr uint8_t max_segs = 3;

            static constexpr float spline_period = 0.2f;
            static constexpr float gps_period = 0.05f;
            static constexpr float sim_step = 0.01f;

            void on_sim();
            void on_action(const ActionMsg& msg);
            void publish_spline();
            void publish_twist();

            void next_segment();
            std::vector<glm::fvec2> arc_segment(float radius, glm::fvec2 start_pos, float start_heading, float end_heading);
            std::vector<glm::fvec2> straight_segment(glm::fvec2 start, float length, float heading);

            rclcpp::Subscription<ActionMsg>::SharedPtr m_subscriber;
            rclcpp::Publisher<SplineMsg>::SharedPtr m_spline_publisher;
            rclcpp::Publisher<TwistMsg>::SharedPtr m_twist_publisher;

            rclcpp::TimerBase::SharedPtr m_spline_timer;
            rclcpp::TimerBase::SharedPtr m_gps_timer;
            rclcpp::TimerBase::SharedPtr m_sim_timer;

            // thomas model state
            std::array<double, 13> m_world_state {-3, 0, 0, 0, 0, 0, 0, 0, -3.0411, 0, 0, 0, 0};

            rclcpp::Time m_time;
            std::mt19937 m_rng;
            std::normal_distribution<float> m_norm_dist {0, 1};
            std::uniform_real_distribution<float> m_uniform_dist {0, 1};

            SegmentType m_last_segment_type = SegmentType::NONE;
            glm::fvec2 m_spline_end_pos = {0, 0};
            float m_spline_end_heading = 0;
            std::list<std::vector<glm::fvec2>> m_segments;
            ActionMsg m_last_action_msg;
        };

    }
}

