#include <iostream>
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
        TestNode::TestNode(uint32_t seed)
            : Node{"test_node"},

              m_subscriber (create_subscription<ActionMsg>(
                    control_action_topic_name, control_action_qos,
                    [this] (const ActionMsg::SharedPtr msg) { on_action(*msg); })),

              m_spline_timer {create_wall_timer(
                  std::chrono::duration<float, std::milli>(spline_period * 1000),
                    [this]{ publish_spline(); })},

              m_sim_timer {create_wall_timer(
                  std::chrono::duration<float, std::milli>(sim_step * 1000),
                    [this]{ on_sim(); })},

              m_gps_timer {create_wall_timer(
                        std::chrono::duration<float, std::milli>(gps_period * 1000),
                            [this]{ publish_twist(); })},

              m_spline_publisher {create_publisher<SplineMsg>(spline_topic_name, spline_qos)},
              m_twist_publisher {create_publisher<TwistMsg>(world_twist_topic_name, world_twist_qos)},

              m_rng {seed} {

            next_segment();
            next_segment();

            m_time = get_clock()->now();
        }

        void TestNode::next_segment() {

            std::vector<glm::fvec2> new_seg;
            float end_heading;
            glm::fvec2 end_pos;
            switch (m_last_segment_type) {
                case SegmentType::NONE:
                case SegmentType::ARC: {
                    bool is_straight = m_uniform_dist(m_rng) < straight_after_arc_prob;
                    if (is_straight) {
                        new_seg = straight_segment(m_spline_end_pos, m_uniform_dist(m_rng) * (max_straight - min_straight) + min_straight, m_spline_end_heading);
                        m_last_segment_type = SegmentType::STRAIGHT;
                        end_heading = m_spline_end_heading;
                        end_pos = new_seg.back();

                    } else {
                        float radius = m_uniform_dist(m_rng) * (max_radius - min_radius) + min_radius;
                        float arc_rad = (m_uniform_dist(m_rng) - 0.5f) * (max_arc_rad - min_arc_rad) * 2;
                        end_heading = m_spline_end_heading + arc_rad;

                        new_seg = arc_segment(radius, m_spline_end_pos, m_spline_end_heading, end_heading);
                        m_last_segment_type = SegmentType::ARC;
                        m_spline_end_heading = m_spline_end_heading + arc_rad;

                        end_pos = new_seg.back();
                    }
                    break;
                }

                case SegmentType::STRAIGHT: {
                    float radius = m_uniform_dist(m_rng) * (max_radius - min_radius) + min_radius;
                    float arc_rad = (m_uniform_dist(m_rng) - 0.5f) * (max_arc_rad - min_arc_rad) * 2;
                    end_heading = m_spline_end_heading + arc_rad;

                    new_seg = arc_segment(radius, m_spline_end_pos, m_spline_end_heading, end_heading);
                    m_last_segment_type = SegmentType::ARC;

                    end_pos = new_seg.back();
                    break;
                }
            }

            m_spline_end_heading = end_heading;
            m_spline_end_pos = end_pos;

            m_segments.push_back(new_seg);
            if (m_segments.size() > max_segs) {
                m_segments.erase(m_segments.begin());
            }
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

        std::vector<glm::fvec2> TestNode::straight_segment(glm::fvec2 start, float length, float heading) {
            std::vector<glm::fvec2> result;
            const uint32_t steps = length / spline_frame_separation;
            for (uint32_t i = 1; i <= steps; i++) {
                result.push_back(start + i * spline_frame_separation * glm::fvec2 {glm::cos(heading), glm::sin(heading)});
            }
            return result;
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
            if (distance(car_pos, m_spline_end_pos) < new_seg_dist) {
                next_segment();
            }
        }


        void TestNode::on_action(const interfaces::msg::ControlAction& msg) {
            std::cout << "\nSwangle: " << msg.swangle * (180 / M_PI) << " Torque f: " <<
                msg.torque_fl + msg.torque_fr << " Torque r: " << msg.torque_rl + msg.torque_rr << std::endl;

            m_last_action_msg = msg;
        }

        void TestNode::publish_spline() {
            SplineMsg msg {};

            const glm::fvec2 car_pos = {m_world_state[0], m_world_state[1]};
            const float car_heading = m_world_state[2];

            auto gen_point = [&car_pos, car_heading](const glm::fvec2& point) {
                geometry_msgs::msg::Point p;
                glm::fvec2 rel_point = point - car_pos;
                p.y = rel_point.x * glm::cos(car_heading) + rel_point.y * glm::sin(car_heading);
                p.x = rel_point.x * -glm::sin(car_heading) + rel_point.y * glm::cos(car_heading);
                return p;
            };

            for (const auto& segment : m_segments) {
                for (const glm::fvec2& point : segment) {
                    msg.frames.push_back(gen_point(point));
                }
            }

            msg.orig_data_stamp = get_clock()->now();

            m_spline_publisher->publish(msg);
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
    }
}

int main(int argc, char* argv[]){
    uint32_t seed;
    if (argc == 1) {
        std::cout << "No seed given, using default 0" << std::endl;
        seed = 0;
    } else {
        seed = strtol(argv[1], nullptr, 10);
        std::cout << "Using seed " << seed << std::endl;
    }

    rclcpp::init(argc, argv);

    rclcpp::spin(std::make_shared<controls::tests::TestNode>(seed));

    rclcpp::shutdown();
    return 0;
}