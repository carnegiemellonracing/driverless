#include <mppi/mppi.hpp>
#include <mutex
#include <rclcpp/rclcpp.hpp>
#include <types.hpp>
#include <constants.hpp>
#include <interfaces/msg/control_action.hpp>

#include "controller.hpp"


namespace controls {
    namespace nodes {

            ControllerNode::ControllerNode(std::unique_ptr<state::StateEstimator> state_estimator, std::unique_ptr<Controller> controller)
                : Node {controller_node_name},
                  m_state_estimator(std::move(state_estimator)),
                  m_controller(std::move(controller)),
                  m_controller_timer(
                      create_wall_timer(
                          controller_period,
                          [this] { controller_callback(); })),
                  m_action_publisher(
                      create_publisher<interfaces::msg::ControlAction>(
                          control_action_topic_name,
                          control_action_qos)) { }

            void ControllerNode::publish_action(const Action& action) const {
                std::cout << "Action publishing not implmented. Continuing..." << std::endl;
            }

            void ControllerNode::controller_callback() const {
                const State curv_state = m_state_estimator->get_curv_state();
                const Action action = m_controller->generate_action(curv_state);

                publish_action(action);
            }

    };
}


int main(int argc, char *argv[]) {
    using namespace controls;

    std::unique_ptr<state::StateEstimator> state_estimator = std::make_unique<state::StateEstimator>();
    std::unique_ptr<Controller> controller = std::make_unique<mppi::MppiController>();

    rclcpp::init(argc, argv);

    const auto node = std::make_shared<nodes::ControllerNode>(state_estimator, controller);
    rclcpp::spin(node);

    rclcpp::shutdown();
    return 0;
}