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

                  m_state_estimator {std::move(state_estimator)},
                  m_mppi_controller {std::move(mppi_controller)},

                  m_action_timer {
                      create_wall_timer(
                          std::chrono::duration<float, std::milli>(controller_period_ms),
                          [this] { publish_action_callback(); })
                  },

                  m_action_publisher {
                      create_publisher<interfaces::msg::ControlAction>(
                          control_action_topic_name,
                          control_action_qos)
                  },

                  m_spline_subscription {
                      create_subscription<SplineMsg>(
                          spline_topic_name, spline_qos,
                          [this] (const SplineMsg::SharedPtr msg) { spline_callback(*msg); })
                  },

                  m_slam_subscription {
                      create_subscription<SlamMsg>(
                          spline_topic_name, spline_qos,
                          [this] (const SlamMsg::SharedPtr msg) { slam_callback(*msg); })
                  },

                  m_action_read {std::make_unique<Action>()},
                  m_action_write {std::make_unique<Action>()} { }

            void ControllerNode::publish_action_callback() {
                std::lock_guard<std::mutex> action_read_guard {action_read_mut};

                publish_action(*m_action_read);
            }

            void ControllerNode::spline_callback(const SplineMsg& spline_msg) {
                std::cout << "Received spline" << std::endl;

                m_state_estimator->on_spline(spline_msg);
                run_mppi();
            }

            void ControllerNode::slam_callback(const SlamMsg& slam_msg) {
                std::cout << "Received slam" << std::endl;

                m_state_estimator->on_slam(slam_msg);
                run_mppi();
            }

            void ControllerNode::publish_action(const Action& action) const {
                interfaces::msg::ControlAction msg;
                msg.swangle = action[action_swangle_idx];
                msg.torque_f = action[action_torque_f_idx];
                msg.torque_r = action[action_torque_r_idx];

                std::cout << "Publishing message: { swangle: " << msg.swangle << ", torque_f: "
                          << msg.torque_f << ", torque_r: " << msg.torque_r << " }" << std::endl;

                m_action_publisher->publish(msg);
            }

            void ControllerNode::run_mppi() {
                std::cout << "-------- RUN MPPI -------" << std::endl;

                std::cout << "locking action_write..." << std::endl;
                {
                    std::lock_guard<std::mutex> action_write_guard {action_write_mut};
                    std::cout << "done.\n" << std::endl;

                    std::cout << "generating action..." << std::endl;
                    *m_action_write = m_mppi_controller->generate_action();
                    std::cout << "done.\n" << std::endl;
                }

                std::cout << "swapping action buffers..." << std::endl;
                swap_action_buffers();
                std::cout << "done.\n" << std::endl;
            }

            void ControllerNode::swap_action_buffers() {
                std::lock(action_read_mut, action_write_mut);  // lock both using a deadlock avoidance scheme
                std::lock_guard<std::mutex> action_read_guard {action_read_mut, std::adopt_lock};
                std::lock_guard<std::mutex> action_write_guard {action_write_mut, std::adopt_lock};

                std::swap(m_action_read, m_action_write);
            }

    }
}


int main(int argc, char *argv[]) {
    using namespace controls;

    std::unique_ptr<state::StateEstimator> state_estimator = state::StateEstimator::create();
    std::unique_ptr<mppi::MppiController> controller = mppi::MppiController::create();

    rclcpp::init(argc, argv);
    std::cout << "rclcpp initialized" << std::endl;

    const auto node = std::make_shared<nodes::ControllerNode>(
        std::move(state_estimator),
        std::move(controller)
    );

    std::cout << "controller node created" << std::endl;

    rclcpp::spin(node);

    rclcpp::shutdown();
    return 0;
}