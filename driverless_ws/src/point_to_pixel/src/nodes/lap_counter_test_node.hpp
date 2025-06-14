#pragma once

// Project Headers
#include "../cones/cones.hpp"

// ROS2 imports
#include "rclcpp/rclcpp.hpp"
#include "interfaces/msg/cone_array.hpp"

class LapCounterTestNode : public rclcpp::Node
{
public:
    LapCounterTestNode();

private:

    rclcpp::Subscription<interfaces::msg::ConeArray>::SharedPtr cone_sub;

    cones::LapCounter lap_counter;

    void cone_callback(const interfaces::msg::ConeArray::SharedPtr msg);

};