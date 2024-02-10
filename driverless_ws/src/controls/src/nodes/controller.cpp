#include <mppi/mppi.hpp>
#include <mutex>
#include <rclcpp/rclcpp.hpp>
#include <types.hpp>
#include <constants.hpp>
#include <interfaces/msg/control_action.hpp>
#include <state/state_estimator.hpp>

#include "controller.hpp"

namespace controls {
    namespace nodes {

            ControllerNode::ControllerNode(
                std::unique_ptr<state::StateEstimator> state_estimator,
                std::unique_ptr<mppi::MppiController> mppi_controller)
                : Node {controller_node_name},
                  m_state_estimator(std::move(state_estimator)),
                  m_mppi_controller(std::move(mppi_controller)),
                  m_action_timer(
                      create_wall_timer(
                          std::chrono::duration<float, std::milli>(controller_period_ms),
                          [this] { publish_action_callback(); })),
                  m_action_publisher(
                      create_publisher<interfaces::msg::ControlAction>(
                          control_action_topic_name,
                          control_action_qos)) { }

            void ControllerNode::publish_action_callback() {
                std::lock_guard<std::mutex> action_read_guard {action_read_mut};

                publish_action(*action_read);
            }

            void ControllerNode::spline_callback(const SplineMsg& spline_msg) {
                m_state_estimator->on_spline(spline_msg);
                run_mppi();
            }

            void ControllerNode::slam_callback(const SlamMsg& slam_msg) {
                m_state_estimator->on_slam(slam_msg);
                run_mppi();
            }

            void ControllerNode::publish_action(const Action& action) const {
                std::cout << "Action publishing not implmented. Continuing..." << std::endl;
            }

            void ControllerNode::run_mppi() {
                {
                    std::lock_guard<std::mutex> action_write_guard {action_write_mut};

                    *action_write = m_mppi_controller->generate_action();
                }

                swap_action_buffers();
            }

            void ControllerNode::swap_action_buffers() {
                std::lock(action_read_mut, action_write_mut);  // lock both using a deadlock avoidance scheme
                std::lock_guard<std::mutex> action_read_guard {action_read_mut, std::adopt_lock};
                std::lock_guard<std::mutex> action_write_guard {action_write_mut, std::adopt_lock};

                std::swap(action_read, action_write);
            }

    }
}


int main(int argc, char *argv[]) {
    using namespace controls;

    std::unique_ptr<state::StateEstimator> state_estimator = state::StateEstimator::create();
    std::unique_ptr<mppi::MppiController> controller = mppi::MppiController::create();

    rclcpp::init(argc, argv);

    const auto node = std::make_shared<nodes::ControllerNode>(
        std::move(state_estimator),
        std::move(controller)
    );

    rclcpp::spin(node);

    rclcpp::shutdown();
    return 0;
}