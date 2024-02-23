#include <mppi/mppi.hpp>
#include <mutex>
#include <rclcpp/rclcpp.hpp>
#include <types.hpp>
#include <constants.hpp>
#include <interfaces/msg/control_action.hpp>
#include <state/state_estimator.hpp>

#ifdef PUBLISH_STATES
#include <display/display.hpp>
#endif

#include "controller.hpp"

namespace controls {
    namespace nodes {

            ControllerNode::ControllerNode(
                std::shared_ptr<state::StateEstimator> state_estimator,
                std::shared_ptr<mppi::MppiController> mppi_controller)
                : Node {controller_node_name},

                  m_state_estimator {std::move(state_estimator)},
                  m_mppi_controller {std::move(mppi_controller)},

                  m_action_timer {
                      create_wall_timer(
                          std::chrono::duration<float>(1),
                          [this] { publish_action_callback(); })
                  },

                  m_action_publisher {
                      create_publisher<interfaces::msg::ControlAction>(
                          control_action_topic_name,
                          control_action_qos)
                  },

                  m_action_read {std::make_unique<Action>()},
                  m_action_write {std::make_unique<Action>()} {

                rclcpp::CallbackGroup::SharedPtr state_estimation_callback_group {
                    create_callback_group(rclcpp::CallbackGroupType::MutuallyExclusive)};

                rclcpp::SubscriptionOptions options {};
                options.callback_group = state_estimation_callback_group;

                m_spline_subscription = create_subscription<SplineMsg>(
                    spline_topic_name, spline_qos,
                    [this] (const SplineMsg::SharedPtr msg) { spline_callback(*msg); },
                    options
                );

                m_state_subscription  = create_subscription<StateMsg>(
                    state_topic_name, state_qos,
                    [this] (const StateMsg::SharedPtr msg) { state_callback(*msg); },
                    options
                );
            }

            void ControllerNode::publish_action_callback() {
                std::lock_guard<std::mutex> action_read_guard {action_read_mut};

                publish_action(*m_action_read);
            }

            void ControllerNode::spline_callback(const SplineMsg& spline_msg) {
                std::cout << "Received spline" << std::endl;

                {
                    std::lock_guard<std::mutex> guard {state_mut};

                    m_state_estimator->on_spline(spline_msg);
                    notify_state_dirty();
                }
            }

            void ControllerNode::state_callback(const StateMsg& state_msg) {
                std::cout << "Received slam" << std::endl;

                {
                    std::lock_guard<std::mutex> guard {state_mut};

                    m_state_estimator->on_state(state_msg);
                    notify_state_dirty();
                }
            }

            void ControllerNode::publish_action(const Action& action) const {
                interfaces::msg::ControlAction msg;
                msg.swangle = action[action_swangle_idx];
                msg.torque_fl = action[action_torque_f_idx] / 2;
                msg.torque_fr = action[action_torque_f_idx] / 2;
                msg.torque_rl = action[action_torque_r_idx] / 2;
                msg.torque_rr = action[action_torque_r_idx] / 2;

                m_action_publisher->publish(msg);
            }

            std::thread ControllerNode::launch_mppi() {
                return {[this] {
                    std::unique_lock<std::mutex> state_lock {state_mut};

                    while (true) {
                        state_lock.lock();
                        state_cond_var.wait();
                        m_state_estimator->sync_to_device();
                        state_lock.unlock();

                        std::cout << "-------- MPPI -------" << std::endl;

                        Action action = m_mppi_controller->generate_action();
                        {
                            std::lock_guard<std::mutex> action_guard {action_write_mut};
                            *m_action_write = action;
                        }

                        std::cout << "swapping action buffers" << std::endl;
                        swap_action_buffers();

                        std::cout << "---------------------" << std::endl;
                    }
                }}
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

    std::shared_ptr<state::StateEstimator> state_estimator = state::StateEstimator::create();
    std::shared_ptr<mppi::MppiController> controller = mppi::MppiController::create();

    rclcpp::init(argc, argv);
    std::cout << "rclcpp initialized" << std::endl;

    const auto node = std::make_shared<nodes::ControllerNode>(state_estimator, controller);
    std::cout << "controller node created" << std::endl;

    std::mutex thread_died_mut;
    std::condition_variable thread_died_cond;
    bool thread_died = false;

    std::thread node_thread {[&] {
        rclcpp::executors::MultiThreadedExecutor exec;
        exec.add_node(node);
        exec.spin();

        {
            std::lock_guard<std::mutex> guard {thread_died_mut};

            std::cout << "Node terminated. Exiting..." << std::endl;
            thread_died = true;
            thread_died_cond.notify_all();
        }
    }};

    std::cout << "controller node thread launched" << std::endl;


#ifdef PUBLISH_STATES
    display::Display display {controller, state_estimator};
    std::cout << "display created" << std::endl;

    std::thread display_thread {[&] {
        display.run();

        {
            std::lock_guard<std::mutex> guard {thread_died_mut};

            std::cout << "Display thread closed. Exiting.." << std::endl;
            thread_died = true;
            thread_died_cond.notify_all();
        }
    }};
    std::cout << "display thread launched" << std::endl;
#endif

    {
        std::unique_lock<std::mutex> guard {thread_died_mut};
        if (!thread_died) {
            thread_died_cond.wait(guard);
        }
    }

    std::cout << "Shutting down" << std::endl;
    rclcpp::shutdown();
    return 0;
}