/**
 * To-do list
 * 1. Add noise to cone positions
 * 2. Add a flag to merge cones that are really close together
 * 3. Start at the start of the track
 * 4. Add timing
 * 3. (When C++ SVM is ready, integrate Husain's code)
 */

/**
 * Notes about track specification
 * Angles are in degrees - positive means clockwise, negative means counterclockwise
 */

#include <iostream>
#include <fstream>
#include <sstream>
#include <types.hpp>
#include <constants.hpp>
#include <cmath>
#include <geometry_msgs/msg/point.hpp>
#include <glm/glm.hpp>
#include <random>
#include <map>

#include <gsl/gsl_odeiv2.h>
#include <gsl/gsl_errno.h>
#include <model/two_track/codegen/minimal_state_function.h>
#include <filesystem>
#include "test_node.hpp"
//#include <utils/general_utils.hpp>

namespace controls {
    namespace tests {
        /// @brief Clamp some heading/angle value to the range [0, 2pi)
        static constexpr float arc_rad_adjusted(float arc_rad)
        {
            while (arc_rad < -M_PI)
            {
                arc_rad += 2.0f * M_PI;
            }
            while (arc_rad >= M_PI)
            {
                arc_rad -= 2.0f * M_PI;
            }
            return arc_rad;
        }

        TestNode::TestNode(std::map<std::string, std::string> config_dict)
            : Node{"test_node"},

              // ROS stuff
              m_subscriber(create_subscription<ActionMsg>(
                  control_action_topic_name, control_action_qos,
                  [this](const ActionMsg::SharedPtr msg)
                  { on_action(*msg); })),

              m_track_timer{create_wall_timer(
                  std::chrono::duration<float, std::milli>(spline_period * 1000),
                  [this]
                  { publish_track(); })},

              m_sim_timer{create_wall_timer(
                  std::chrono::duration<float, std::milli>(sim_step * 1000),
                  [this]
                  { on_sim(); })},

              m_gps_timer{create_wall_timer(
                  std::chrono::duration<float, std::milli>(gps_period * 1000),
                  [this]
                  { publish_twist(); })},

              m_spline_publisher{create_publisher<SplineMsg>(spline_topic_name, spline_qos)},
              m_twist_publisher{create_publisher<TwistMsg>(world_twist_topic_name, world_twist_qos)},
              m_cone_publisher {create_publisher<ConeMsg>(cone_topic_name, spline_qos)},
              
              m_config_dict{config_dict},
              m_all_segments{parse_segments_specification(m_config_dict["root_dir"] + m_config_dict["track_specs"])},


              m_lookahead{std::stof(m_config_dict["look_ahead"])},
              m_lookahead_squared {m_lookahead * m_lookahead},

              m_log_file {m_config_dict["root_dir"] + m_config_dict["track_logs"]}
        {
            std::cout << m_lookahead << std::endl;
            std::cout << m_all_segments.size() << std::endl;
            // m_all_segmentsd = parse_segments_specification(m_config_dict["root_dir"] + m_config_dict["track_specs"]);
            // m_lookahead = std::stof(m_config_dict["look_ahead"]);
            // m_lookahead_squared = m_lookahead * m_lookahead;
            
            glm::fvec2 curr_pos {0.0f, 0.0f};
            float curr_heading = 0.0f;
            for (const auto& seg : m_all_segments) {
                if (seg.type == SegmentType::ARC) {
                    float next_heading = arc_rad_adjusted(curr_heading + seg.heading_change);
                    const auto& [spline, left, right] = arc_segment_with_cones(seg.radius, curr_pos, curr_heading, next_heading);
                    m_all_left_cones.insert(m_all_left_cones.end(), left.begin(), left.end());
                    m_all_right_cones.insert(m_all_right_cones.end(), right.begin(), right.end());
                    m_all_spline.insert(m_all_spline.end(), spline.begin(), spline.end());
                    curr_pos = spline.back();
                    curr_heading = next_heading;
                } else if (seg.type == SegmentType::STRAIGHT) {
                    const auto& [spline, left, right] = straight_segment_with_cones(curr_pos, seg.length, curr_heading);
                    m_all_left_cones.insert(m_all_left_cones.end(), left.begin(), left.end());
                    m_all_right_cones.insert(m_all_right_cones.end(), right.begin(), right.end());
                    m_all_spline.insert(m_all_spline.end(), spline.begin(), spline.end());
                    curr_pos = spline.back();
                }
            }
            m_finish_line = curr_pos;
            // Update visible indexes
            update_visible_indices();

            m_time = get_clock()->now();

            // timing track stuff
            m_start_line.push_back(m_all_left_cones[0]);
            m_start_line.push_back(m_all_right_cones[0]);
            m_end_line.push_back(m_all_left_cones.back());
            m_end_line.push_back(m_all_right_cones.back());
            update_track_time();
        }
        float distanceToLine(glm::fvec2 point, std::vector<glm::fvec2> line) {
            glm::fvec2 s_l = line[0];
            glm::fvec2 s_r = line[1];

            float m = point.x;
            float n = point.y;
            float A = s_r.y - s_l.y;
            float B = s_l.x - s_r.x;
            float C = s_r.x * s_l.y - s_r.y * s_l.x;

            // Perpendicular distance from point (m, n) to the line
            float numerator = std::abs(A * m + B * n + C);
            float denominator = std::sqrt(A * A + B * B);

            return numerator / denominator;
        }
        
        static bool is_within_line(glm::fvec2 point, std::vector<glm::fvec2> line, float err) {
            return distanceToLine(point, line) < err;
        }

        static constexpr float get_squared_distance(glm::fvec2 point1, glm::fvec2 point2) {
            return (point1.x - point2.x) * (point1.x - point2.x) + (point1.y - point2.y) * (point1.y - point2.y);
        }

        static size_t find_closest_point(const std::vector<glm::fvec2>& points, glm::fvec2 position, float lookahead_squared, size_t prev_closest) {
            while (get_squared_distance(points.at(prev_closest), position) >= lookahead_squared) {
                prev_closest = (prev_closest + 1) % points.size();
            }
            return prev_closest;
        }

        static size_t find_furthest_point(const std::vector<glm::fvec2>& points, glm::fvec2 position, float lookahead_squared, size_t prev_furthest, size_t prev_closest) {
            while (get_squared_distance(points.at(prev_furthest), position) < lookahead_squared) {
                prev_furthest = (prev_furthest + 1) % points.size();
                if (prev_furthest == prev_closest) {
                    break;
                }
            }
            return prev_furthest;
        }

        void TestNode::update_visible_indices() {
            auto [left_closest, left_furthest, right_closest, right_furthest, spline_closest, spline_furthest] = m_visible_indices;

            glm::fvec2 curr_pos {m_world_state[0], m_world_state[1]};

            // TODO: make this more modular
            left_closest = find_closest_point(m_all_left_cones, curr_pos, m_lookahead_squared, left_closest);
            right_closest = find_closest_point(m_all_right_cones, curr_pos, m_lookahead_squared, right_closest);
            spline_closest = find_closest_point(m_all_spline, curr_pos, m_lookahead_squared, spline_closest);

            right_furthest = find_furthest_point(m_all_right_cones, curr_pos, m_lookahead_squared, right_furthest, right_closest);
            left_furthest = find_furthest_point(m_all_left_cones, curr_pos, m_lookahead_squared, left_furthest, left_closest);
            spline_furthest = find_furthest_point(m_all_spline, curr_pos, m_lookahead_squared, spline_furthest, spline_closest);
            m_visible_indices = {left_closest, left_furthest, right_closest, right_furthest, spline_closest, spline_furthest};
        }

        void TestNode::update_track_time(){
            glm::fvec2 curr_pos {m_world_state[0], m_world_state[1]};
            bool within_start = false;
            bool within_end = false;
            
            if (!m_is_loop) {
                within_start = is_within_line(curr_pos, m_start_line, 0.5f);
                within_end = is_within_line(curr_pos, m_end_line, 0.5f);
            } else {
                within_start = is_within_line(curr_pos, m_start_line, 0.5f);
            }
            //std::cout << "ONE sx: " << m_start_line[0].x << " sy: " << m_start_line[0].y << "\n";
            //std::cout << "TWO sx: " << m_start_line[1].x << " sy: " << m_start_line[1].x << "\n";

            //std::cout << "ONE sx: " << m_end_line[0].x << " sy: " << m_end_line[0].y << "\n";
            //std::cout << "TWO sx: " << m_end_line[1].x << " sy: " << m_end_line[1].x << "\n";

            m_is_loop = m_is_loop || within_start && within_end;

            if (!m_is_loop) { //track is not a loop
                if (within_start && !m_seen_start) {
                    m_seen_start = true;
                    m_start_time = get_clock()->now();
                } else if (within_end && m_seen_start) {
                    m_end_time = get_clock()->now();
                    rclcpp::Duration elapsed = m_end_time - m_start_time;
                    if (m_log_file) {
                        m_log_file << "Lap:" << m_lap_count << ":" << elapsed.seconds() << std::endl;
                    }
                }
            } else { //track is a loop
                if (within_start && !m_seen_start) { // at the start of loop
                    m_seen_start = true;
                    m_start_time = get_clock()->now();
                } else if (within_start && m_seen_start) { // completed a lap
                    m_end_time = get_clock()->now();
                    rclcpp::Duration elapsed = m_end_time - m_start_time;

                    if (elapsed.seconds() > 2.0f){
                        if (m_log_file) {
                            m_lap_count++;
                            m_log_file << "Lap:" << m_lap_count <<":" << elapsed.seconds() << std::endl;

                            std::cout<< "Lap:" << m_lap_count <<":" << elapsed.seconds() << "\n";
                            std::cout<<"FILE FOUND\n";
                        }
                        else {
                            std::cout<< "Lap:" << m_lap_count <<":" << elapsed.seconds() << "\n";
                        }
                    }
                    m_seen_start = false;
                }
            }
            m_raceline_points.push_back(curr_pos);
        }


        std::vector<glm::fvec2> TestNode::arc_segment(float radius, glm::fvec2 start_pos, float start_heading, float end_heading) {
            std::vector<glm::fvec2> result;

            float arc_rad = end_heading - start_heading;
            float center_heading = arc_rad > 0 ?
                start_heading + M_PI_2 : start_heading - M_PI_2;

            glm::fvec2 center = m_spline_end_pos + radius * glm::fvec2 {glm::cos(center_heading), glm::sin(center_heading)};
            float start_angle = std::atan2(start_pos.y - center.y, start_pos.x - center.x);

            const uint32_t steps = glm::abs(radius * arc_rad / spline_frame_separation);
            const float step_rad = arc_rad / steps;
            for (uint32_t i = 1; i <= steps; i++) {
                float angle = start_angle + i * step_rad;
                result.push_back(center + radius * glm::fvec2 {glm::cos(angle), glm::sin(angle)});
            }
            return result;
        }
        
        TestNode::SplineAndCones TestNode::arc_segment_with_cones(float radius, glm::fvec2 start_pos, float start_heading, float end_heading) {
            std::vector<glm::fvec2> spline, left_cones, right_cones, left_spline, right_spline;

            float arc_rad = arc_rad_adjusted(end_heading - start_heading);
            bool counter_clockwise = arc_rad > 0;
            // Angle from the starting_point to center 
            float center_heading = counter_clockwise ?
                arc_rad_adjusted(start_heading + M_PI_2) : arc_rad_adjusted(start_heading - M_PI_2);

            glm::fvec2 center = start_pos + radius * glm::fvec2 {glm::cos(center_heading), glm::sin(center_heading)};
            // Angle from the center to the starting point
            float start_angle = arc_rad_adjusted(std::atan2(start_pos.y - center.y, start_pos.x - center.x));

            const uint32_t steps = glm::abs(radius * arc_rad / spline_frame_separation);
            
            const float step_rad = arc_rad / steps;

            float left_dist;
            float right_dist;
            float distance_right;
            float distance_left;

            if (counter_clockwise)
            {
                left_dist = radius - track_width / 2;
                right_dist = radius + track_width / 2;
                distance_right = cone_spacing_arc_outer;
                distance_left = cone_spacing_arc_inner;
            }
            else
            {
                left_dist = radius + track_width / 2;
                right_dist = radius - track_width / 2;
                distance_left = cone_spacing_arc_outer;
                distance_right = cone_spacing_arc_inner;
            }
            
            const uint32_t left_steps = glm::abs(left_dist * arc_rad / distance_left);
            const uint32_t right_steps = glm::abs(right_dist * arc_rad / distance_right);            

            for (uint32_t i = 1; i <= std::max(std::max(steps, left_steps), right_steps); i++) {
                if(i <= steps) {
                    float angle = arc_rad_adjusted(start_angle + i * step_rad);
                    glm::fvec2 outgoing_vector = glm::fvec2 {glm::cos(angle), glm::sin(angle)};
                    spline.push_back(center + radius * outgoing_vector);
                }
                if(i <= left_steps) {
                    float angle_left = arc_rad_adjusted(start_angle + i * arc_rad / left_steps);
                    glm::fvec2 outgoing_vector = glm::fvec2 {glm::cos(angle_left), glm::sin(angle_left)};
                    float left_noise = glm::clamp(m_arc_jitter_gen(m_rng), -m_noise_clip, m_noise_clip);
                    left_cones.push_back(center + left_dist * outgoing_vector + left_noise * outgoing_vector);
                }
                if(i <= right_steps) {
                    float angle_right = arc_rad_adjusted(start_angle + i * arc_rad / right_steps);
                    glm::fvec2 outgoing_vector = glm::fvec2 {glm::cos(angle_right), glm::sin(angle_right)};
                    float right_noise = glm::clamp(m_arc_jitter_gen(m_rng), -1 * m_noise_clip, m_noise_clip);
                    right_cones.push_back(center + right_dist * outgoing_vector + right_noise * outgoing_vector);
                    std::cout << "Right dist: " << right_dist << " Right noise: " << right_noise << "\n9";
                }
            }

            return make_tuple(spline, left_cones, right_cones);
        }
    
    
        /**
         * Generate a sequence of positions along a straight line.
         *
         * Given a position, a length, and a heading, generate a sequence of
         * positions along a straight line, starting at the given position,
         * going in the direction of the given heading, and continuing for the
         * given length.
         *
         * @param start The starting position of the line segment.
         * @param length The length of the line segment.
         * @param heading The direction of the line segment, in radians.
         *
         * @return A vector of positions, equally spaced along the line segment.
         */
        std::vector<glm::fvec2> TestNode::straight_segment(glm::fvec2 start, float length, float heading) {
            std::vector<glm::fvec2> result;
            const uint32_t steps = length / spline_frame_separation;
            for (uint32_t i = 1; i <= steps; i++) {
                result.push_back(start + i * spline_frame_separation * glm::fvec2 {glm::cos(heading), glm::sin(heading)});
            }
            return result;
        }

        TestNode::SplineAndCones TestNode::straight_segment_with_cones(glm::fvec2 start, float length, float heading) {
            std::vector<glm::fvec2> spline, left, right;
            constexpr float cone_dist = track_width / 2;


            glm::fvec2 left_diff {glm::cos(heading + M_PI_2), glm::sin(heading + M_PI_2)};
            glm::fvec2 right_diff {glm::cos(heading - M_PI_2), glm::sin(heading - M_PI_2)};
            glm::fvec2 left_start = start + cone_dist * left_diff;
            glm::fvec2 right_start = start + cone_dist * right_diff;
            const uint32_t steps = length / spline_frame_separation;
            const uint32_t lr_steps = length / cone_spacing_straight;

            for (uint32_t i = 1; i <= std::max(lr_steps, steps); i++) {
                if(i <= steps) {
                    glm::fvec2 step = i * spline_frame_separation * glm::fvec2 {glm::cos(heading), glm::sin(heading)};
                    spline.push_back(start + step);
                }
                if(i <= lr_steps) {
                    glm::fvec2 step = i * cone_spacing_straight * glm::fvec2 {glm::cos(heading), glm::sin(heading)};
                    glm::fvec2 perp = glm::fvec2 {glm::sin(heading), -glm::cos(heading)};
                    float left_noise = glm::clamp(m_straight_jitter_gen(m_rng), -1 * m_noise_clip, m_noise_clip);
                    float right_noise = glm::clamp(m_straight_jitter_gen(m_rng), -1 * m_noise_clip, m_noise_clip);
                    left.push_back(left_start + step + left_noise * perp);
                    right.push_back(right_start + step + right_noise * perp);
                }
            }
            return std::make_tuple(spline, left, right);
        }

        float dot(glm::fvec2 u, glm::fvec2 v) {
            return u[0] * v[0] + u[1] * v[1];
        }

        float dist_line(glm::fvec2 start, glm::fvec2 end, glm::fvec2 point) {
            glm::fvec2 line_vec = end - start;
            glm::fvec2 point_vec = point - start;
            glm::fvec2 perp = point_vec - line_vec * (dot(point_vec, line_vec) / dot(line_vec, line_vec));
            return std::sqrt(perp[0] * perp[0] + perp[1] * perp[1]);
        }

        // bool TestNode::detect_cone(float threshold, glm::fvec2 cone_pos, glm::fvec2 robot_pos, float heading, float width, float height) {
        //     glm::fvec2 off_height = height/2 * glm::fvec2 {glm::cos(heading), glm::sin(heading)};
        //     glm::fvec2 off_width = width/2 * glm::fvec2 {glm::sin(heading), -glm::cos(heading)};
        //     glm::fvec2 bounding_box[4];
        //     int idx = 0;
        //     for(int i = -1; i <= 1; i += 2) {
        //         for(int j = -1; j <= 1; j += 2) {
        //             bounding_box[idx++] = robot_pos + i * off_height + j * off_width;
        //         }
        //     }
        //     std::swap(bounding_box[2], bounding_box[3]);
        //     for(int i = 0; i < 2; i++) {
        //         int dist = dist_line(bounding_box[i], bounding_box[3-i], cone_pos);
        //         if(dist <= threshold) return false;
        //     }
        //     return true;
        // }


        int model_func(double t, const double state[], double dstatedt[], void* params) {
            const double yaw = state[2];
            const double x_world_dot = state[3] * cos(yaw) - state[4] * sin(yaw);
            const double y_world_dot = state[3] * sin(yaw) + state[4] * cos(yaw);
            const double yaw_dot = state[5];

            double minimal_state[10];
            memcpy(minimal_state, &state[3], sizeof(double) * 10);

            auto action_msg = *(ActionMsg*)params;
            double action[5] = {
                action_msg.swangle, action_msg.torque_fl, action_msg.torque_fr, action_msg.torque_rl, action_msg.torque_rr
            };

            double minimal_state_dot[10];

            minimal_state_function(minimal_state, action, minimal_state_dot);

            dstatedt[0] = x_world_dot;
            dstatedt[1] = y_world_dot;
            dstatedt[2] = yaw_dot;
            memcpy(&dstatedt[3], minimal_state_dot, sizeof(minimal_state_dot));

            return GSL_SUCCESS;
        }

        void TestNode::on_sim() {
            ActionMsg adj_msg = m_last_action_msg;
            adj_msg.torque_fl *= gear_ratio / 1000.;
            adj_msg.torque_fr *= gear_ratio / 1000.;
            adj_msg.torque_rl *= gear_ratio / 1000.;
            adj_msg.torque_rr *= gear_ratio / 1000.;

            gsl_odeiv2_system system {};
            system.function = model_func;
            system.dimension = 13;
            system.jacobian = nullptr;
            system.params = (void*)&adj_msg;

            gsl_odeiv2_driver* driver = gsl_odeiv2_driver_alloc_y_new(
                &system,
                gsl_odeiv2_step_rkf45,
                1e-4, 1e-4, 1e-4
            );

            double sim_time = m_time.nanoseconds() / 1.0e9;
            m_time = get_clock()->now();

            int result = gsl_odeiv2_driver_apply(driver, &sim_time, m_time.nanoseconds() / 1.0e9, m_world_state.data());
            if (result != GSL_SUCCESS) {
                throw std::runtime_error("GSL driver failed");
            }

            gsl_odeiv2_driver_free(driver);
            update_visible_indices();
            update_track_time();
            // float x_diff = (m_world_state[0] - m_finish_line.x);
            // float y_diff = (m_world_state[1] - m_finish_line.y);
            // if (x_diff * x_diff + y_diff * y_diff < ) {
            //     std::cout << "REACHED FINISH LINE!!!\n";
            //     m_world_state = { -3,
            //                       0,
            //                       0,
            //                       0,
            //                       0,
            //                       0,
            //                       0,
            //                       0,
            //                       -3.0411,
            //                       0,
            //                       0,
            //                       0,
            //                       0 };

            //     m_visible_indices = m_initial_visible_indices;
            // }
        }


        void TestNode::on_action(const interfaces::msg::ControlAction& msg) {
            std::cout << "\nSwangle: " << msg.swangle * (180 / M_PI) << " Torque f: " <<
                msg.torque_fl + msg.torque_fr << " Torque r: " << msg.torque_rl + msg.torque_rr << std::endl;

            m_last_action_msg = msg;
        }
        /**
         * Fills output vector (some field of a ROS message) with the points in input vector starting at start and ending at end
         */
        static void fill_points(std::vector<geometry_msgs::msg::Point> & output, const std::vector<glm::fvec2> &input, size_t start, size_t end,
                                 const std::function<geometry_msgs::msg::Point(const glm::fvec2&)> &gen_point)
        {
            while (start != end)
            {
                output.push_back(gen_point(input.at(start)));
                start++;
                if (start == input.size())
                {
                    start = 0;
                }
            }
        }

        
        void TestNode::publish_track() {
            SplineMsg spline_msg {};
            ConeMsg cone_msg {};

            const glm::fvec2 car_pos = {m_world_state[0], m_world_state[1]};
            const float car_heading = m_world_state[2];

            // transformation from world to car frame
            auto gen_point = [&car_pos, car_heading](const glm::fvec2& point) {
                geometry_msgs::msg::Point p;
                glm::fvec2 rel_point = point - car_pos;
                p.y = rel_point.x * glm::cos(car_heading) + rel_point.y * glm::sin(car_heading);
                p.x = rel_point.x * -glm::sin(car_heading) + rel_point.y * glm::cos(car_heading);
                return p;
            };

            auto [left_closest, left_furthest, right_closest, right_furthest, spline_closest, spline_furthest] = m_visible_indices;

            fill_points(cone_msg.blue_cones, m_all_left_cones, left_closest, left_furthest, gen_point);
            fill_points(cone_msg.yellow_cones, m_all_right_cones, right_closest, right_furthest, gen_point);
            fill_points(spline_msg.frames, m_all_spline, spline_closest, spline_furthest, gen_point);

            // for display only
            for (const glm::fvec2& point : m_all_left_cones) {
                cone_msg.orange_cones.push_back(gen_point(point));
            }

            for (const glm::fvec2& point : m_all_right_cones) {
                cone_msg.unknown_color_cones.push_back(gen_point(point));
            }

            for (const glm::fvec2& point : m_raceline_points) {
                cone_msg.big_orange_cones.push_back(gen_point(point));
            }
    
        
            auto curr_time = get_clock()->now();
            spline_msg.header.stamp = curr_time;
            cone_msg.header.stamp = curr_time;
            spline_msg.orig_data_stamp = curr_time;
            cone_msg.orig_data_stamp = curr_time;

            m_spline_publisher->publish(spline_msg);
            m_cone_publisher->publish(cone_msg);
        }
 

        void TestNode::publish_twist() {
            TwistMsg msg {};

            const float yaw = m_world_state[2];
            const float car_xdot = m_world_state[3];
            const float car_ydot = m_world_state[4];
            const float yawdot = m_world_state[5];

            msg.twist.linear.x = car_xdot * std::cos(yaw) - car_ydot * std::sin(yaw);
            msg.twist.linear.y = car_xdot * std::sin(yaw) + car_ydot * std::cos(yaw);
            msg.twist.linear.z = 0.0;

            msg.twist.angular.x = 0.0;
            msg.twist.angular.y = 0.0;
            msg.twist.angular.z = yawdot;

            msg.header.stamp = get_clock()->now();

            m_twist_publisher->publish(msg);
        }
        std::deque<Segment> TestNode::parse_segments_specification(std::string track_specifications_path)
        {
            if (!std::filesystem::exists(track_specifications_path)) {
                std::cout << "Track spec with the path <" << track_specifications_path << "> does not exist. Did you remember to put it in tests/tracks/ ?\n";
            } else {
                std::cout << "Track spec file exists.\n";
            }                
            std::deque<Segment> segments;
            std::ifstream spec_file(track_specifications_path);

            
            if (spec_file.is_open())
            {
                std::string line;
                while (std::getline(spec_file, line, '\n'))
                {
                    std::istringstream segment_stream(line);
                    char segment_type;
                    segment_stream >> segment_type;
                    
                    Segment segment;

                    if (segment_type == '#') {
                        // this is a comment
                        continue;
                    }
                    else if (segment_type == 's')
                    {
                        float length;
                        segment_stream.ignore(1);
                        segment_stream >> length;
                        
                        segment.type = SegmentType::STRAIGHT;
                        segment.length = length;
                        segments.push_back(segment);
                    }
                    else if (segment_type == 'a')
                    {
                        float radius, heading_change_deg;
                        segment_stream.ignore(1);
                        segment_stream >> radius;

                        segment_stream.ignore(1);
                        segment_stream >> heading_change_deg;
                        
                        segment.type = SegmentType::ARC;
                        segment.radius = radius;
                        segment.heading_change = arc_rad_adjusted(heading_change_deg * M_PI / 180.0f);
                        // std::cout << segment.heading_change << std::endl;
                        segments.push_back(segment);
                    }
                    else
                    {
                        // throw some error here for erroneous file format
                    }
                }
            }

            spec_file.close();
            
            return segments;
        }
    }
}

static constexpr float default_lookahead = 50.0f;

std::string trim(const std::string &str) {
    const size_t first = str.find_first_not_of(" \t");

    if (first == std::string::npos) return "";

    const size_t last = str.find_last_not_of(" \t");
    const size_t length = last - first + 1;

    return str.substr(first, length);
}



int main(int argc, char* argv[]){

    if (argc < 2) {
        std::cout << "Perhaps you didn't pass in the simulation config file path." << std::endl;
        return 1;
    } else {
        std::cout << "Simulation config file passed." << std::endl;
    }
    
    std::string config_file_path = argv[1];
    std::string config_file_base_path = "/home/controls_copy/driverless/driverless_ws/src/controls/tests/sim_configs/";
    std::string config_file_full_path = config_file_base_path + config_file_path;
    
    std::map<std::string, std::string> config_dict;
    
    if (std::filesystem::exists(config_file_full_path)) {
        std::cout << "Simulation config file found.\n";
    } else {
        std::cout << "Simulation config file not found.\n";
        return 1;
    }
    
    std::ifstream conf_file(config_file_full_path);  
    
    if (conf_file.is_open()) {
        std::string line;
        while (std::getline(conf_file, line, '\n'))
        {
            std::cout << line << std::endl;
            if (line.empty() || line[0] == '#') {
                continue;
            } else {
                int delim_position = line.find(':');
                std::string key = trim(line.substr(0, delim_position));
                std::string val = trim(line.substr(delim_position + 1));
                // std::cout << line << " " << key << " " << val <<std::endl;
                config_dict[key] = val;
            }
        }
    }

    // for(const auto & elem : config_dict)
    // {
    //     std::cout << elem.first << " " << elem.second << " " << "\n";
    // }
    // std::string track_spec_full_path = "/home/controls_copy/driverless/driverless_ws/src/controls/tests/" + track_specification;
    // std::cout << "track_spec_full_path: " << track_spec_full_path << std::endl;

    // if (!std::filesystem::exists(config_file_path)) {
    //     std::cout << "File with that config file path does not exist. Did you remember to put it in tests/sim_configs/ ?\n";
    // } else {
    //     std::cout << "Config file exists\n";
    // }

    // if (!std::filesystem::exists(track_lap_log_path)) {
    //     std::cout << "File named with that output name does not exist.\n";
    // } else {
    //     std::cout << "Path for logs exists.\n";
    // }
    rclcpp::init(argc, argv);

    rclcpp::spin(std::make_shared<controls::tests::TestNode>(config_dict));

    rclcpp::shutdown();
    return 0;
}