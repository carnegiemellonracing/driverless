//
// Created by anthony on 8/30/24.
//

#include <iostream>
#include <types.hpp>
#include <constants.hpp>
#include <cmath>
#include <geometry_msgs/msg/point.hpp>
#include <rclcpp/time.hpp>
#include <rclcpp/duration.hpp>

#include "echo_node.hpp"

static const std::string old_spline_topic_name = "old_spline";

namespace controls {
    namespace test {
        EchoNode::EchoNode()
            : Node {"echo_node"},
              m_old_spline_subscriber {create_subscription<SplineMsg>(
                    old_spline_topic_name, spline_qos,
                    [this] (const SplineMsg::SharedPtr msg) { echo(*msg); })},
              m_spline_publisher {create_publisher<SplineMsg>(spline_topic_name, spline_qos)} {
                // m_curr_time = get_clock()->now().seconds();
              }

        void EchoNode::echo(const SplineMsg& msg) {
            if (!m_orig_time) {
                m_orig_time = rclcpp::Time(msg.orig_data_stamp).seconds();
                m_curr_time = get_clock()->now().seconds() - 0.1;
            }
            SplineMsg new_msg = msg;
            new_msg.orig_data_stamp = rclcpp::Time(msg.orig_data_stamp) + rclcpp::Duration::from_seconds(m_curr_time - m_orig_time.value());
            m_spline_publisher->publish(new_msg);
        }
    }
}

// int main(int argc, char* argv[]) {
//     if (argc != 2) {
//         std::cout << "Usage: echo_node <start time in seconds of rosbag>" << std::endl;
//         return 1;
//     } else {
//         double orig_time = strtod(argv[1], nullptr);
//         rclcpp::init(argc, argv);
//         rclcpp::spin(std::make_shared<controls::test::EchoNode>(orig_time));
//         rclcpp::shutdown();
//         return 0;
//     }
// }


int main(int argc, char* argv[]) {
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<controls::test::EchoNode>());
    rclcpp::shutdown();
    return 0;
}