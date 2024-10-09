#include <iostream>
#include <fstream>
#include <sstream>
#include <types.hpp>
#include <constants.hpp>
#include <cmath>
#include <geometry_msgs/msg/point.hpp>
#include <glm/glm.hpp>
#include <gsl/gsl_odeiv2.h>
#include <gsl/gsl_errno.h>
#include <model/two_track/codegen/minimal_state_function.h>

#include "test_node.hpp"


namespace controls {
    namespace tests {
        /// Add a heading to the current heading, clampiing to within [0, 2pi).
        static constexpr float add_heading(float current, float diff) {
            float result = current + diff;
            if (result < 0.0) {
                result += M_PI_2;
            }
            if (result > 2.0 * M_PI_2) {
                result -= M_PI_2;
            }
            return result;
        }

        TestNode::TestNode(const std::string &track_specification, float lookahead)
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

              m_all_segments{parse_segments_specification(track_specification)},



              m_lookahead{lookahead},
              m_lookahead_squared {lookahead * lookahead}
        {
            glm::fvec2 curr_pos {0.0f, 0.0f};
            float curr_heading = 0.0f;
            for (const auto& seg : m_all_segments) {
                if (seg.type == SegmentType::ARC) {
                    float next_heading = add_heading(curr_heading, seg.heading_change);
                    const auto& [spline, left_cones, cones] = arc_segment_with_cones(seg.radius, curr_pos, curr_heading, next_heading);
                    m_all_left_cones.insert(m_all_left_cones.end(), left_cones.begin(), left_cones.end());
                    m_all_right_cones.insert(m_all_right_cones.end(), cones.begin(), cones.end());
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
            // Update visible indexes
            update_visible_indices();

            m_time = get_clock()->now();
        }

        static constexpr float get_squared_distance(glm::fvec2 point1, glm::fvec2 point2) {
            return (point1.x - point2.x) * (point1.x - point2.x) + (point1.y - point2.y) * (point1.y - point2.y);
        }

        void TestNode::update_visible_indices() {
            auto [left_closest, left_furthest] = m_visible_left_idx;
            auto [right_closest, right_furthest] = m_visible_right_idx;
            glm::fvec2 curr_pos {m_world_state[0], m_world_state[1]};

            while (get_squared_distance(m_all_left_cones.at(left_closest), curr_pos) >= m_lookahead_squared) {
                left_closest = (left_closest + 1) % m_all_left_cones.size();
            }

            while (get_squared_distance(m_all_right_cones.at(right_closest), curr_pos) >= m_lookahead_squared) {
                right_closest = (right_closest + 1) % m_all_right_cones.size();
            }

            while (get_squared_distance(m_all_right_cones.at(right_furthest), curr_pos) < m_lookahead_squared) {
                right_furthest = (right_furthest + 1) % m_all_right_cones.size();
                if (right_furthest == right_closest) {
                    break;
                }
            }
        
            while (get_squared_distance(m_all_left_cones.at(left_furthest), curr_pos) < m_lookahead_squared) {
                left_furthest = (left_furthest + 1) % m_all_left_cones.size();
                if (left_furthest == left_closest) {
                    break;
                }
            }

            m_visible_left_idx = {left_closest, left_furthest};
            m_visible_right_idx = {right_closest, right_furthest};
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
            std::vector<glm::fvec2> spline, left_cones, right_cones;

            float arc_rad = end_heading - start_heading;
            bool counter_clockwise = arc_rad > 0;
            float center_heading = counter_clockwise ?
                start_heading + M_PI_2 : start_heading - M_PI_2;

            glm::fvec2 center = m_spline_end_pos + radius * glm::fvec2 {glm::cos(center_heading), glm::sin(center_heading)};
            // Angle form the center to the starting point
            float start_angle = std::atan2(start_pos.y - center.y, start_pos.x - center.x);

            const uint32_t steps = glm::abs(radius * arc_rad / spline_frame_separation);
            const float step_rad = arc_rad / steps;

            float left_dist, right_dist;
            if (counter_clockwise)
            {
                 left_dist = radius - track_width / 2;
                 right_dist = radius + track_width / 2;
            }
            else
            {
                left_dist = radius + track_width / 2;
                right_dist = radius - track_width / 2;
            }

            for (uint32_t i = 1; i <= steps; i++) {
                float angle = start_angle + i * step_rad;
                glm::fvec2 outgoing_vector = glm::fvec2 {glm::cos(angle), glm::sin(angle)};
                spline.push_back(center + radius * outgoing_vector);
                left_cones.push_back(center + left_dist * outgoing_vector);
                right_cones.push_back(center + right_dist * outgoing_vector);

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
            for (uint32_t i = 1; i <= steps; i++) {
                glm::fvec2 step = i * spline_frame_separation * glm::fvec2 {glm::cos(heading), glm::sin(heading)};
                spline.push_back(start + step);
                left.push_back(left_start + step);
                right.push_back(right_start + step);
            }
            return std::make_tuple(spline, left, right);
        }


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

            const glm::fvec2 car_pos = {m_world_state[0], m_world_state[1]};
            update_visible_indices();
        }


        void TestNode::on_action(const interfaces::msg::ControlAction& msg) {
            std::cout << "\nSwangle: " << msg.swangle * (180 / M_PI) << " Torque f: " <<
                msg.torque_fl + msg.torque_fr << " Torque r: " << msg.torque_rl + msg.torque_rr << std::endl;

            m_last_action_msg = msg;
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

        
            for (const glm::fvec2& point : m_all_spline) {
                spline_msg.frames.push_back(gen_point(point));
            }

            auto [closest_left, furthest_left] = m_visible_left_idx;
            auto [closest_right, furthest_right] = m_visible_right_idx;

            while (closest_left != furthest_left) {
                cone_msg.blue_cones.push_back(gen_point(m_all_left_cones.at(closest_left)));
                closest_left++;
                if (closest_left == m_all_left_cones.size()) {
                    closest_left = 0;
                }
            }

            while (furthest_right != closest_right) {
                cone_msg.yellow_cones.push_back(gen_point(m_all_right_cones.at(furthest_right)));
                furthest_right++;
                if (furthest_right == m_all_right_cones.size()) {
                    furthest_right = 0;
                }
            }

            // for display only
            for (const glm::fvec2& point : m_all_left_cones) {
                cone_msg.orange_cones.push_back(gen_point(point));
            }

            for (const glm::fvec2& point : m_all_right_cones) {
                cone_msg.unknown_color_cones.push_back(gen_point(point));
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
            std::deque<Segment> segments;
            std::ifstream spec_file(track_specifications_path);
            if (spec_file.is_open())
            {
                std::string line;
                while (std::getline(spec_file, line, ','))
                {
                    std::istringstream segment_stream(line);
                    char segment_type;
                    segment_stream >> segment_type;
                    
                    Segment segment;
                    
                    if (segment_type == 's')
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
                        float radius, heading_change;
                        segment_stream.ignore(1);
                        segment_stream >> radius;

                        segment_stream.ignore(1);
                        segment_stream >> heading_change;
                        
                        segment.type = SegmentType::ARC;
                        segment.radius = radius;
                        segment.heading_change = heading_change;
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

static constexpr float default_lookahead = 8.0f;

int main(int argc, char* argv[]){
    // if (argc == 1) {
    //     std::cout << "specify track specification" << std::endl;
    //     return 1;
    // }
    // std::string track_specification = argv[1];
    // float look_ahead = default_lookahead;
    // if (argc == 3) {
    //     look_ahead = std::stof(argv[2]);
    // }
    std::string track_specification = "track1";
    float look_ahead = 10.0f;
        
    rclcpp::init(argc, argv);

    rclcpp::spin(std::make_shared<controls::tests::TestNode>(track_specification, look_ahead));

    rclcpp::shutdown();
    return 0;
}