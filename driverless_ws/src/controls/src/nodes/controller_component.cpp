#include "rclcpp/rclcpp.hpp"
#include "interfaces/msg/control_action.hpp"

namespace controller {

    class ControllerComponent : public rclcpp::Node
    {
    public:
        ControllerComponent()
            : Node("controller_component")
        {
            // Create a subscriber to listen to the "controls action" topic
            subscription_ = create_subscription<interfaces::msg::ControlAction>(
                "control_action", 10,
                std::bind(&ControllerComponent::controlsActionCallback, this, std::placeholders::_1));
        }

        ControllerComponent(const rclcpp::NodeOptions &options)
            : Node("controller_component")
        {
            // Create a subscriber to listen to the "controls action" topic
            subscription_ = create_subscription<interfaces::msg::ControlAction>(
                "control_action", 10,
                std::bind(&ControllerComponent::controlsActionCallback, this, std::placeholders::_1));
        }

    private:
        void controlsActionCallback(const interfaces::msg::ControlAction::SharedPtr msg)
        {
            // Handle the received controls action message here
            RCLCPP_INFO(get_logger(), "Received controls action message");
            
            // fill me in

        }

        rclcpp::Subscription<interfaces::msg::ControlAction>::SharedPtr subscription_;
    };
}


#include "rclcpp_components/register_node_macro.hpp"

// Register the component with class_loader.
// This acts as a sort of entry point, allowing the component to be discoverable when its library
// is being loaded into a running process.
RCLCPP_COMPONENTS_REGISTER_NODE(controller::ControllerComponent)
