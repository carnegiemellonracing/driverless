#include <mppi/mppi.hpp>
#include <mutex>
#include <rclcpp/rclcpp.hpp>
#include <types.hpp>
#include <constants.hpp>
#include <interfaces/msg/control_action.hpp>
#include <state/state_estimator.hpp>

#ifdef DISPLAY
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
                          std::chrono::duration<float>(controller_publish_period),
                          [this] { publish_action_callback(); })
                  },

                  m_action_publisher {
                      create_publisher<interfaces::msg::ControlAction>(
                          control_action_topic_name,
                          control_action_qos)
                  },

                  m_action_read {std::make_unique<Action>()},
                  m_action_write {std::make_unique<Action>()} {

                // create a callback group that prevents state and spline callbacks from being executed concurrently
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

               // no pose subscription, since everything in car frame for now. will change when fast mode is implemented

                m_world_quat_subscription = create_subscription<QuatMsg>(
                    world_quat_topic_name, world_quat_qos,
                    [this] (const QuatMsg::SharedPtr msg) { world_quat_callback(*msg); },
                    options
                );

                m_world_twist_subscription = create_subscription<TwistMsg>(
                    world_twist_topic_name, world_twist_qos,
                    [this] (const TwistMsg::SharedPtr msg) { world_twist_callback(*msg); },
                    options
                );

                // start mppi :D
                // this won't immediately begin publishing, since it waits for the first dirty state
                launch_mppi().detach();
            }

            void ControllerNode::publish_action_callback() {
                std::lock_guard<std::mutex> action_read_guard {m_action_read_mut};

                publish_action(*m_action_read);
            }

            void ControllerNode::spline_callback(const SplineMsg& spline_msg) {
                std::cout << "Received spline" << std::endl;

                {
                    std::lock_guard<std::mutex> guard {m_state_mut};
                    m_state_estimator->on_spline(spline_msg);
                }

                notify_state_dirty();
            }

            void ControllerNode::state_callback(const StateMsg& state_msg) {
                std::cout << "Received state" << std::endl;

                {
                    std::lock_guard<std::mutex> guard {m_state_mut};
                    m_state_estimator->on_state(state_msg);
                }

                notify_state_dirty();
            }

            void ControllerNode::world_twist_callback(const TwistMsg &twist_msg) {
                std::cout << "Received twist" << std::endl;

                {
                    std::lock_guard<std::mutex> guard {m_state_mut};
                    m_state_estimator->on_world_twist(twist_msg);
                }

                notify_state_dirty();
            }

            void ControllerNode::world_quat_callback(const QuatMsg &quat_msg) {
                std::cout << "Received quat" << std::endl;

                {
                    std::lock_guard<std::mutex> guard {m_state_mut};
                    m_state_estimator->on_world_quat(quat_msg);
                }

                notify_state_dirty();
            }

            void ControllerNode::world_pose_callback(const PoseMsg &pose_msg) {
                std::cout << "Received pose" << std::endl;

                {
                    std::lock_guard<std::mutex> guard {m_state_mut};
                    m_state_estimator->on_world_pose(pose_msg);
                }

                notify_state_dirty();
            }

            void ControllerNode::publish_action(const Action& action) const {
                interfaces::msg::ControlAction msg;
                msg.swangle = action[action_swangle_idx];
                msg.torque_fl = action[action_torque_idx] / 4;
                msg.torque_fr = action[action_torque_idx] / 4;
                msg.torque_rl = action[action_torque_idx] / 4;
                msg.torque_rr = action[action_torque_idx] / 4;

                m_action_publisher->publish(msg);
            }

            std::thread ControllerNode::launch_mppi() {
                return std::thread {[this] {
                    while (true) {
                        std::unique_lock<std::mutex> state_lock {m_state_mut};

                        m_state_cond_var.wait(state_lock);  // wait to be dirtied
                        while (!m_state_estimator->is_ready()) {
                            m_state_cond_var.wait(state_lock);
                        }

                        // record time to estimate speed
                        auto start_time = std::chrono::high_resolution_clock::now();
                        std::cout << "-------- MPPI -------" << std::endl;

                        // send state to device (i.e. cuda globals)
                        // (also serves to lock state since nothing else updates gpu state)
                        std::cout << "syncing state to device" << std::endl;
                        m_state_estimator->sync_to_device();

                        // we don't need the host state anymore, so release the lock and let state callbacks proceed
                        state_lock.unlock();

                        // run mppi, and write action to the write buffer
                        Action action = m_mppi_controller->generate_action();
                        {
                            std::lock_guard<std::mutex> action_guard {m_action_write_mut};
                            *m_action_write = action;
                        }

                        // swap the read and write buffers so publish action read this action
                        std::cout << "swapping action buffers" << std::endl;
                        swap_action_buffers();

                        // calculate and print time elapsed
                        auto time_elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(
                            std::chrono::high_resolution_clock::now() - start_time
                        );
                        std::cout << "time elapsed: " << time_elapsed.count() << std::endl;

                        std::cout << "---------------------" << std::endl;
                    }
                }};
            }

            void ControllerNode::notify_state_dirty() {
                m_state_cond_var.notify_all();
            }

            void ControllerNode::swap_action_buffers() {
                std::lock(m_action_read_mut, m_action_write_mut);  // lock both using a deadlock avoidance scheme
                std::lock_guard<std::mutex> action_read_guard {m_action_read_mut, std::adopt_lock};
                std::lock_guard<std::mutex> action_write_guard {m_action_write_mut, std::adopt_lock};

                std::swap(m_action_read, m_action_write);
            }

    }
}


int main(int argc, char *argv[]) {
    using namespace controls;
    
    std::mutex mppi_mutex;
    std::mutex state_mutex;

    // create resources
    std::shared_ptr<state::StateEstimator> state_estimator = state::StateEstimator::create(state_mutex);
    std::shared_ptr<mppi::MppiController> controller = mppi::MppiController::create(mppi_mutex);

    rclcpp::init(argc, argv);
    std::cout << "rclcpp initialized" << std::endl;

    // instantiate node
    const auto node = std::make_shared<nodes::ControllerNode>(state_estimator, controller);
    std::cout << "controller node created" << std::endl;

    // create a condition variable to notify main thread when either display or node dies
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


#ifdef DISPLAY
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

    // wait for a thread to die
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