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

                  m_action_publisher {
                //TODO: <ActionMsg>
                      create_publisher<interfaces::msg::ControlAction>(
                          control_action_topic_name,
                          control_action_qos
                        )
                  },

                  m_info_publisher {
                      create_publisher<InfoMsg>(
                            controller_info_topic_name,
                            controller_info_qos
                        )
                  } {

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

                m_world_twist_subscription = create_subscription<TwistMsg>(
                    world_twist_topic_name, world_twist_qos,
                    [this] (const TwistMsg::SharedPtr msg) { world_twist_callback(*msg); },
                    options
                );

                m_world_pose_subscription = create_subscription<PoseMsg>(
                    world_pose_topic_name, world_pose_qos,
                    [this] (const PoseMsg::SharedPtr msg) { world_pose_callback(*msg); },
                    options
                );
                //TODO: m_state_mut never gets initialized? I guess default construction is alright;

                // start mppi :D
                // this won't immediately begin publishing, since it waits for the first dirty state
                launch_mppi().detach();
            }

            void ControllerNode::spline_callback(const SplineMsg& spline_msg) {
                RCLCPP_DEBUG(get_logger(), "Received spline");

                {
                    std::lock_guard<std::mutex> guard {m_state_mut};
                    m_state_estimator->on_spline(spline_msg);
                }

                notify_state_dirty();
            }

            void ControllerNode::world_twist_callback(const TwistMsg &twist_msg) {
                RCLCPP_DEBUG(get_logger(), "Received twist");


                {
                    std::lock_guard<std::mutex> guard {m_state_mut};
                    m_state_estimator->on_twist(twist_msg, twist_msg.header.stamp);
                }

                notify_state_dirty();
            }

            void ControllerNode::world_pose_callback(const PoseMsg &pose_msg) {
                RCLCPP_DEBUG(get_logger(), "Received pose");


                {
                    std::lock_guard<std::mutex> guard {m_state_mut};
                    m_state_estimator->on_pose(pose_msg);
                }

                notify_state_dirty();
            }

            void ControllerNode::publish_action(const Action& action) {
                const auto msg = action_to_msg(action);
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
                        RCLCPP_DEBUG(get_logger(), "mppi iteration beginning");

                        // save for info publishing later, since might be changed during iteration
                        State proj_curr_state = m_state_estimator->get_projected_state();
                        rclcpp::Time orig_spline_data_stamp = m_state_estimator->get_orig_spline_data_stamp();

                        // send state to device (i.e. cuda globals)
                        // (also serves to lock state since nothing else updates gpu state)
                        RCLCPP_DEBUG(get_logger(), "syncing state to device");
                        m_state_estimator->sync_to_device(get_clock()->now());


                        // we don't need the host state anymore, so release the lock and let state callbacks proceed
                        state_lock.unlock();

                        // run mppi, and write action to the write buffer
                        Action action = m_mppi_controller->generate_action();
                        publish_action(action);

                        m_state_estimator->record_control_action(action, get_clock()->now());

                        // calculate and print time elapsed
                        auto time_elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(
                            std::chrono::high_resolution_clock::now() - start_time
                        );

                        // can't use high res clock since need to be aligned with other nodes
                        auto total_time_elapsed = (get_clock()->now().nanoseconds() - orig_spline_data_stamp.nanoseconds()) / 1000000;

                        interfaces::msg::ControllerInfo info {};
                        info.action = action_to_msg(action);
                        info.proj_state = state_to_msg(proj_curr_state);
                        info.latency_ms = time_elapsed.count();
                        info.total_latency_ms = total_time_elapsed;

                        std::stringstream ss;
                        publish_and_print_info(ss, info);
                        std::string info_str = ss.str();

                        m_info_publisher->publish(info);
                        std::cout << clear_term_sequence << info_str << std::flush;
                        RCLCPP_DEBUG_STREAM(get_logger(), "mppi step complete. info:\n" << info_str);
                    }
                }};
            }

            void ControllerNode::notify_state_dirty() {
                m_state_cond_var.notify_all();
            }

            ActionMsg ControllerNode::action_to_msg(const Action &action) {
                interfaces::msg::ControlAction msg;

                //TODO: why not current time?
                msg.orig_data_stamp = m_state_estimator->get_orig_spline_data_stamp();

                msg.swangle = action[action_swangle_idx];
                msg.torque_fl = action[action_torque_idx] / 4;
                msg.torque_fr = action[action_torque_idx] / 4;
                msg.torque_rl = action[action_torque_idx] / 4;
                msg.torque_rr = action[action_torque_idx] / 4;

                switch (torque_mode) {
                    case TorqueMode::AWD:
                        msg.torque_fl = action[action_torque_idx] / 4;
                        msg.torque_fr = action[action_torque_idx] / 4;
                        msg.torque_rl = action[action_torque_idx] / 4;
                        msg.torque_rr = action[action_torque_idx] / 4;
                        break;

                    case TorqueMode::FWD:
                        msg.torque_fl = action[action_torque_idx] / 2;
                        msg.torque_fr = action[action_torque_idx] / 2;
                        msg.torque_rl = 0;
                        msg.torque_rr = 0;
                        break;

                    case TorqueMode::RWD:
                        msg.torque_fl = 0;
                        msg.torque_fr = 0;
                        msg.torque_rl = action[action_torque_idx] / 2;
                        msg.torque_rr = action[action_torque_idx] / 2;
                        break;

                    default:
                        throw std::runtime_error("Invalid torque mode");
                }

                return msg;
            }

            StateMsg ControllerNode::state_to_msg(const State &state) {
                StateMsg msg;
                msg.x = state[state_x_idx];
                msg.y = state[state_y_idx];
                msg.yaw = state[state_yaw_idx];
                msg.speed = state[state_speed_idx];

                return msg;
            }

            void ControllerNode::publish_and_print_info(std::ostream &stream, interfaces::msg::ControllerInfo info) {
                stream
                << "Action:\n"
                << "  swangle (rad): " << info.action.swangle << "\n"
                << "  torque_fl (Nm): " << info.action.torque_fl << "\n"
                << "  torque_fr (Nm): " << info.action.torque_fr << "\n"
                << "  torque_rl (Nm): " << info.action.torque_rl << "\n"
                << "  torque_rr (Nm): " << info.action.torque_rr << "\n"
                << "Projected State:\n"
                << "  x (m): " << info.proj_state.x << "\n"
                << "  y (m): " << info.proj_state.y << "\n"
                << "  yaw (rad): " << info.proj_state.yaw << "\n"
                << "  speed (m/s): " << info.proj_state.speed << "\n"
                << "MPPI Step Latency (ms): " << info.latency_ms << "\n"
                << "Total Latency (ms): " << info.total_latency_ms << "\n"
                << std::endl;
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

    rclcpp::Logger logger = node->get_logger();
    LoggerFunc logger_func = [logger](const std::string& msg) {
        RCLCPP_DEBUG(logger, msg.c_str());
    };

    //TODO why is this not a nullptr
    state_estimator->set_logger(logger_func);
    controller->set_logger(logger_func);

    // create a condition variable to notify main thread when either display or node dies
    std::mutex thread_died_mut;
    std::condition_variable thread_died_cond;
    bool thread_died = false;

    std::thread node_thread {[&] {
        rclcpp::executors::MultiThreadedExecutor exec;
        exec.add_node(node);
        exec.spin();

        {
            //TODO: what does guard do here
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