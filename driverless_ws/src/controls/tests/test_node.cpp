#include <iostream>
#include <types.hpp>
#include <constants.hpp>
#include <cmath>
#include <glm/glm.hpp>
#include <gsl/gsl_odeiv2.h>
#include <gsl/gsl_errno.h>
#include <model/two_track/codegen/minimal_state_function.h>

#include "test_node.hpp"


namespace controls {
    namespace tests {
        TestNode::TestNode()
            : Node{"test_node"},

              m_subscriber (create_subscription<ActionMsg>(
                    control_action_topic_name, control_action_qos,
                    [this] (const ActionMsg::SharedPtr msg) { on_action(*msg); })),

              m_spline_publisher {create_publisher<SplineMsg>(spline_topic_name, spline_qos)},

              m_spline_timer {create_wall_timer(
                  std::chrono::duration<float, std::milli>(100),
                    [this]{ publish_spline(); })},

              m_state_publisher {create_publisher<StateMsg>(state_topic_name, state_qos)} {
        }

        SplineMsg sine_spline(float period, float amplitude, float progress, float density) {
            using namespace glm;

            interfaces::msg::SplineFrameList result {};

            fvec2 point {0.0f, 0.0f};
            float total_dist = 0;

            while (total_dist < progress) {
                interfaces::msg::SplineFrame frame {};
                frame.x = point.x;
                frame.y = point.y;
                result.frames.push_back(std::move(frame));

                fvec2 delta = normalize(fvec2(1.0f, amplitude * 2 * M_PI / period * cos(2 * M_PI / period * point.x)))
                            * density;
                total_dist += length(delta);
                point += delta;
            }

            return result;
        }

        SplineMsg spiral_spine(float progress, float density) {
            using namespace glm;

            interfaces::msg::SplineFrameList result {};

            fvec2 point {0.0f, 0.0f};
            float total_dist = 0;
            float theta = 0.0f;
            const float a = 100.0f;

            while (total_dist < progress) {
                interfaces::msg::SplineFrame frame {};
                frame.x = a * theta * cos(theta);
                frame.y = a * theta * sin(theta);
            
                result.frames.push_back(std::move(frame));

                fvec2 ds_dtheta = a * fvec2(cos(theta) - theta * sin(theta), sin(theta) + theta * cos(theta));
                const float delta = density / length(ds_dtheta);
                total_dist += density;
                theta += delta;
            }

            return result;
        }


        SplineMsg line_spline(float progress, float density) {
            using namespace glm;

            SplineMsg result {};

            fvec2 point {0.0f, 0.0f};
            float total_dist = 0;

            while (total_dist < progress) {
                interfaces::msg::SplineFrame frame {};
                frame.x = point.x;
                frame.y = point.y;
                result.frames.push_back(std::move(frame));

                fvec2 delta = fvec2(1, 0) * density;
                total_dist += length(delta);
                point += delta;
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

        void TestNode::on_action(const interfaces::msg::ControlAction& msg) {
            std::cout << "Swangle: " << msg.swangle * (180 / M_PI) << " Torque f: " <<
                msg.torque_fl + msg.torque_fr << " Torque r: " << msg.torque_rl + msg.torque_rr << std::endl << std::endl;

            ActionMsg adj_msg = msg;
            adj_msg.torque_fl /= 1000.;
            adj_msg.torque_fr /= 1000.;
            adj_msg.torque_rl /= 1000.;
            adj_msg.torque_rr /= 1000.;

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

            int result = gsl_odeiv2_driver_apply(driver, &m_time, m_time + controller_period, m_world_state.data());
            if (result != GSL_SUCCESS) {
                throw std::runtime_error("GSL driver failed");
            }

            gsl_odeiv2_driver_free(driver);
            publish_state();
        }

        void TestNode::publish_spline() {
            // std::cout << "Publishing spline" << std::endl << std::endl;
            const auto spline = sine_spline(30, 5, 200, 0.5);
            // const auto spline = spiral_spine(200, 0.5);
            // const auto spline = line_spline(100, 0.5);
            m_spline_publisher->publish(spline);
        }

        void TestNode::publish_state() {
            std::cout << "Publishing state" << std::endl;
            for (float dim : m_world_state) {
                std::cout << dim << " ";
            }
            std::cout << std::endl << std::endl;
            std::cout << "Time: " << m_time << std::endl;

            StateMsg msg {};
            msg.x = m_world_state[0];
            msg.y = m_world_state[1];
            msg.yaw = m_world_state[2];
            msg.xcar_dot = m_world_state[3];
            msg.ycar_dot = m_world_state[4];
            msg.yaw_dot = m_world_state[5];
            msg.downforce = m_world_state[6];
            msg.moment_y = m_world_state[7];
            msg.whl_speed_f = m_world_state[9] + m_world_state[10];
            msg.whl_speed_r = m_world_state[11] + m_world_state[12];

            m_state_publisher->publish(msg);
        }
    }
}

int main(int argc, char* argv[]){
    rclcpp::init(argc, argv);

    rclcpp::spin(std::make_shared<controls::tests::TestNode>());

    rclcpp::shutdown();
    return 0;
}