#include "lap_counter_test_node.hpp"

LapCounterTestNode::LapCounterTestNode() : rclcpp::Node("lap_counter_test_node") {
    cone_sub = create_subscription<interfaces::msg::ConeArray>("/perc_cones", 10, [this](const interfaces::msg::ConeArray::SharedPtr msg) {cone_callback(msg);});

    lap_counter = cones::LapCounter();
}

void LapCounterTestNode::cone_callback(const interfaces::msg::ConeArray::SharedPtr msg) {
    bool lap_detected = lap_counter.update(!(msg->orange_cones).empty());
    RCLCPP_INFO(get_logger(), "cone_prob = %f", lap_counter.get_cone_prob());

    if (lap_detected) {
        RCLCPP_INFO(get_logger(), "\nLap Detected");
        RCLCPP_INFO(get_logger(), "Lap Count: %d\n", lap_counter.num_laps);
    }
}

int main(int argc, char **argv)
{
    rclcpp::init(argc, argv);
    rclcpp::Node::SharedPtr node = std::make_shared<LapCounterTestNode>();
    rclcpp::spin(node);
    rclcpp::shutdown();
    return 0;
}
