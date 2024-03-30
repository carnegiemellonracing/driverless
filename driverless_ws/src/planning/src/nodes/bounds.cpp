#include "rclcpp/rclcpp.hpp"
#include "../planning_codebase/raceline/raceline.hpp"

using std::placeholders::_1;

class BoundsNode : public rclcpp::Node {

public:
    BoundsNode()
        : Node("bounds")
    {
        subscription_cones = this->create_subscription<interfaces::msg::ConeArray>("/perc_cones", 10, std::bind(&MidpointNode::cones_callback, this, _1));
        publisher_rcl_pt = this->create_publisher<interfaces::msg::TrackBounds>("/track_bounds", 10);
        generator = MidpointGenerator(10);


    }
};