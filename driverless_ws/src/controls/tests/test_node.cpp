#include <iostream>
#include <types.hpp>
#include <constants.hpp>
#include <interfaces/msg/control_action.hpp>
#include <interfaces/msg/spline_frame_list.hpp>
#include <interfaces/msg/spline_list.hpp>
#include <cmath>
#include <rclcpp/rclcpp.hpp>
#include <glm/glm.hpp>
#include <std_msgs/msg/header.hpp>

#include "test_node.hpp"


namespace controls {
    namespace tests {
        TestNode::TestNode()
            : Node{"test_node"},

              m_subscriber (create_subscription<interfaces::msg::ControlAction>(
                    control_action_topic_name, control_action_qos,
                    [this] (const interfaces::msg::ControlAction::SharedPtr msg) { print_message(*msg); })),

              m_spline_publisher {create_publisher<interfaces::msg::SplineFrameList>("spline", spline_qos)},

              m_spline_timer {create_wall_timer(
                  std::chrono::duration<float, std::milli>(1000),
                    [this]{ publish_spline(); })} {

        }

        interfaces::msg::SplineFrameList
        sine_spline(float period, float amplitude, float progress, float density) {
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

            result.header = std_msgs::msg::Header {};
            return result;
        }

        interfaces::msg::SplineFrameList
        line_spline(float progress, float density) {
            using namespace glm;

            interfaces::msg::SplineFrameList result {};

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

            result.header = std_msgs::msg::Header {};
            return result;
        }

        void TestNode::print_message(const interfaces::msg::ControlAction& msg) {
            std::cout << "Swangle: " << msg.swangle << " Torque f: " <<
                msg.torque_fl + msg.torque_fr << " Torque r: " << msg.torque_rl + msg.torque_rr << std::endl << std::endl;;
        }

        void TestNode::publish_spline() {
            std::cout << "Publishing spline" << std::endl << std::endl;
//            const auto spline = sine_spline(1, 0.3, 3, 0.05);
            const auto spline = line_spline(100, 0.5);
            m_spline_publisher->publish(spline);
        }
    }
}

int main(int argc, char* argv[]){
    rclcpp::init(argc, argv);

    rclcpp::spin(std::make_shared<controls::tests::TestNode>());

    rclcpp::shutdown();
    return 0;
}