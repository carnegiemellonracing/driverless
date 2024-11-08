#pragma once

#include <iosfwd>
#include <iosfwd>
#include <rclcpp/rclcpp.hpp>
#include <random>
#include <vector>
#include <deque>
#include <glm/common.hpp>
#include <glm/common.hpp>
#include <glm/common.hpp>
#include <glm/common.hpp>

namespace controls {
    namespace tests {
        enum class SegmentType {
                NONE,
                STRAIGHT,
                ARC};
        struct StraightSegment {
            SegmentType type;
            float length;
        };
        struct ArcSegment {
            SegmentType type;
            float radius;
            float heading_change;
        };
        
        struct Segment {
            SegmentType type;
            union {
                float length;
                struct {
                    float radius;
                    float heading_change;
                };
            };
        };


        class TestNode : public rclcpp::Node {
        public:
            /// Constructor reads in the track specification, populates the vector of all segments and cones
            TestNode (const std::string& track_specification, float lookahead);
            

        private:

            /// Spline points, left cone points, right cone points
            using SplineAndCones = std::tuple<std::vector<glm::fvec2>, std::vector<glm::fvec2>, std::vector<glm::fvec2>>;

            static constexpr float jitter_std = 0.00f;
            static constexpr float straight_after_arc_prob = 0.75f;
            static constexpr float min_radius = 4.0f;
            static constexpr float max_radius = 10.0f;
            static constexpr float min_arc_rad = M_PI / 4.0f;
            static constexpr float max_arc_rad = M_PI;
            static constexpr float min_straight = 10.0f;
            static constexpr float max_straight = 30.0f;
            static constexpr float new_seg_dist = 15.0f;
            static constexpr uint8_t max_segs = 4;

            static constexpr float spline_period = 0.2f;
            static constexpr float gps_period = 0.05f;
            static constexpr float sim_step = 0.01f;
            static constexpr float track_width = 4.0f;
            


            void on_sim();
            void on_action(const ActionMsg& msg);
            void publish_track();
            void publish_twist();

            void next_segment();
            std::vector<glm::fvec2> arc_segment(float radius, glm::fvec2 start_pos, float start_heading, float end_heading);
            std::vector<glm::fvec2> straight_segment(glm::fvec2 start, float length, float heading);

            SplineAndCones straight_segment_with_cones(glm::fvec2 start, float length, float heading);
            SplineAndCones arc_segment_with_cones(float radius, glm::fvec2 start_pos, float start_heading, float end_heading);

            std::deque<Segment> parse_segments_specification(std::string track_specifications_path);
            void update_visible_indices();

            rclcpp::Subscription<ActionMsg>::SharedPtr m_subscriber;
            rclcpp::Publisher<SplineMsg>::SharedPtr m_spline_publisher;
            rclcpp::Publisher<TwistMsg>::SharedPtr m_twist_publisher;
            rclcpp::Publisher<ConeMsg>::SharedPtr m_cone_publisher;

            rclcpp::TimerBase::SharedPtr m_track_timer;
            rclcpp::TimerBase::SharedPtr m_gps_timer;
            rclcpp::TimerBase::SharedPtr m_sim_timer;

            std::deque<Segment> m_all_segments;
            std::vector<glm::fvec2> m_all_left_cones;
            std::vector<glm::fvec2> m_all_right_cones;
            std::vector<glm::fvec2> m_all_spline;

            struct Visibility {
                size_t left_start_idx;
                size_t left_end_idx;
                size_t right_start_idx;
                size_t right_end_idx;
                size_t spline_start_idx;
                size_t spline_end_idx;
            };
            Visibility m_visible_indices; ///< The indices of m_all_left_cones, m_all_right_cones, and m_all_spline that are visible to the car.

            const float m_lookahead;
            const float m_lookahead_squared;

            /// Stores the current state of the car (in Thomas model coordinates)
            std::array<double, 13> m_world_state {-3, 0, 0, 0, 0, 0, 0, 0, -3.0411, 0, 0, 0, 0};

            rclcpp::Time m_time;
            std::mt19937 m_rng;
            std::normal_distribution<float> m_norm_dist {0, 1};
            std::uniform_real_distribution<float> m_uniform_dist {0, 1};

            SegmentType m_last_segment_type = SegmentType::NONE;
            glm::fvec2 m_spline_end_pos = {0, 0};
            float m_spline_end_heading = 0;
            ActionMsg m_last_action_msg;
        };

    }
}

