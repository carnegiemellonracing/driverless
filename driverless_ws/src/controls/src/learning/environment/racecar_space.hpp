#pragma once

#include <iosfwd>
#include <random>
#include <vector>
#include <deque>
#include <glm/common.hpp>
#include <glm/glm.hpp>
#include <map>
#include "environment.hpp"



namespace controls {

    namespace RacecarRL {
        #ifndef RACECAR_STATE_ACTION_PAIRS
        #define RACECAR_STATE_ACTION_PAIRS
            struct RacecarObservation : Observation {
                float x_pos;
                float y_pos;
                float yaw;
                float speed;
            };
    
            struct RacecarAction : Action {
                float torque_sum;
                float swangle;
            };
    
        #endif
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


        struct Point {
            float x;
            float y;
            float z;
        };

        struct ConeSpline {
            std::vector<glm::fvec2> blue_cones;
            std::vector<glm::fvec2> yellow_cones;
        };

        struct TrackState {
            ConeSpline cones;
            std::vector<glm::fvec2> track;
            float twist;
        };

        struct Action {
            float torque_fl;
            float torque_fr;
            float torque_rl;
            float torque_rr;
            float swangle;
        };

        class RacecarObservationSpace : ObservationSpace {
        public:
            /// Constructor reads in the track specification, populates the vector of all segments and cones
            RacecarObservationSpace (std::map<std::string, std::string> config_dict);
            std::array<float, 4> getWorldState();        
            RacecarObservation step(RacecarAction a);   
            bool isTerminalObservation(RacecarObservation o);
            bool isTruncatedObservation(RacecarObservation o);

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
            static constexpr float cone_spacing_straight = 3.0f;
            static constexpr float cone_spacing_arc_inner = 1.5f;
            static constexpr float cone_spacing_arc_outer = 2.5f;

            static constexpr float spline_period = 0.2f;
            static constexpr float gps_period = 0.05f;
            static constexpr float sim_step = 0.01f;
            static constexpr float track_width = 4.0f;



            void publish_track(TrackState &s);
            void publish_twist(TrackState &s);

            std::map<std::string, std::string> m_config_dict;
            std::vector<glm::fvec2> arc_segment(float radius, glm::fvec2 start_pos, float start_heading, float end_heading);
            std::vector<glm::fvec2> straight_segment(glm::fvec2 start, float length, float heading);

            SplineAndCones straight_segment_with_cones(glm::fvec2 start, float length, float heading);
            SplineAndCones arc_segment_with_cones(float radius, glm::fvec2 start_pos, float start_heading, float end_heading);

            std::deque<Segment> parse_segments_specification(std::string track_specifications_path);
            void update_visible_indices();

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

            float m_lookahead;
            float m_lookahead_squared;

            /// Stores the current state of the car (in Thomas model coordinates)
            // std::array<double, 13> m_world_state {0, 0, 0, 0, 0, 0, 0, 0, -3.0411, 0, 0, 0, 0};
            std::array<float, 4> m_world_state {0, 0, M_PI_2, 0};

            std::mt19937 m_rng;
            std::uniform_real_distribution<float> m_uniform_dist {0, 1};

            SegmentType m_last_segment_type = SegmentType::NONE;
            glm::fvec2 m_spline_end_pos = {0, 0};
            glm::fvec2 m_finish_line;
            Visibility m_initial_visible_indices;
            float m_spline_end_heading = 0;

            // frequency of updates
            float m_controller_period = 0.1; // (s)

            // number of calls to the sim
            long long m_steps;

            /// For lap tracking
            std::vector<glm::fvec2> m_start_line;
            std::vector<glm::fvec2> m_end_line;
            std::vector<glm::fvec2> m_raceline_points;
            bool m_seen_start = false;
            bool m_is_loop = false;
            size_t m_lap_count = 1;
            std::ofstream m_log_file;

            /// For cone jittering
            std::normal_distribution<float> m_straight_jitter_gen {0, 0.25f};
            std::normal_distribution<float> m_arc_jitter_gen {0, 0.25f};
            float m_noise_clip = 0.2f;
        };

    }
}

