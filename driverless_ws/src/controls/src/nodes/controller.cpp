#include <mppi/mppi.hpp>
#include <mutex>
#include <rclcpp/rclcpp.hpp>
#include <types.hpp>
#include <constants.hpp>
#include <interfaces/msg/control_action.hpp>
#include <state/state_estimator.hpp>
#include <can/cmr_can.h>

static uint16_t swangle_to_adc(float swangle)
{

    int modulus = 4096;
    float swangle_in_degrees = swangle * 180 / (float)M_PI;
    int zero_adc = 3159;
    int min_adc = 2010;
    int max_adc = modulus + 212;
    float min_deg = -21.04;
    float max_deg = 23.6;
    float adc_deg_ratio = ((float)(max_adc - min_adc)) / ((max_deg - min_deg));
    int desired_adc = (int)(swangle_in_degrees * adc_deg_ratio) + zero_adc;
    uint16_t desired_adc_modded = (uint16_t)(desired_adc % modulus);
    return desired_adc_modded;
}

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
                  },

                  m_spline_republisher {
                    create_publisher<SplineMsg>(
                        republished_spline_topic_name,
                        spline_qos
                    )
                  }
                  
                  
                  {
                if constexpr (send_to_can) {
                    int can_init_result = initializeCan();
                    if (can_init_result < 0)
                    {
                        std::cout << "Can failed to initialize with error " << can_init_result << std::endl;
                        throw std::runtime_error("Failed to initialize can");
                    }
                }

                // create a callback group that prevents state and spline callbacks from being executed concurrently
                rclcpp::CallbackGroup::SharedPtr state_estimation_callback_group {
                    create_callback_group(rclcpp::CallbackGroupType::MutuallyExclusive)};

                rclcpp::SubscriptionOptions options {};
                options.callback_group = state_estimation_callback_group;

                const char* desired_spline_topic_name;
                if constexpr (testing_on_rosbag) {
                    desired_spline_topic_name = republished_spline_topic_name;
                } else {
                    desired_spline_topic_name = spline_topic_name;
                }
    

                m_spline_subscription = create_subscription<SplineMsg>(
                    desired_spline_topic_name, spline_qos,
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

                m_rosbag_action_subscriber = create_subscription<ActionMsg> (
                    control_action_topic_name, control_action_qos,
                    [this] (const ActionMsg::SharedPtr msg) { rosbag_action_callback(*msg); },
                    options
                );

                launch_can().detach();
                // start mppi :D
                // this won't immediately begin publishing, since it waits for the first dirty state
                launch_mppi().detach();
            }


            static Action msg_to_action(const ActionMsg& msg) {
                Action action;
                float torque = msg.torque_fl + msg.torque_fr + msg.torque_rl + msg.torque_rr;
                action[action_torque_idx] = torque;
                action[action_swangle_idx] = msg.swangle;
                return action;
            }

            void ControllerNode::rosbag_action_callback(const ActionMsg& action_msg) {
                std::lock_guard<std::mutex> guard {m_state_mut};
                if constexpr (testing_on_rosbag) {
                    rclcpp::Time action_publish_time = action_msg.header.stamp;
                    Action action = msg_to_action(action_msg);
                    m_state_estimator->record_control_action(action, action_publish_time);
                }
            }

            void ControllerNode::spline_callback(const SplineMsg& spline_msg) {
                RCLCPP_DEBUG(get_logger(), "Received spline");

                {
                    std::lock_guard<std::mutex> guard {m_state_mut};
                    m_state_estimator->on_spline(spline_msg);
                }
                m_last_spline_msg = spline_msg;

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

            static void send_finished_ignore_error()
            {
                std::cout << "I just got terminated lol\n";
                sendFinishedCommand();
            }

            ControllerNode::ActionSignal ControllerNode::action_to_signal(Action action)
            {
                ActionSignal action_signal;
                switch (torque_mode) {
                    case TorqueMode::AWD: {
                        action_signal.front_torque_mNm = static_cast<int16_t>(action[action_torque_idx] * 500.0f);
                        action_signal.back_torque_mNm = static_cast<int16_t>(action[action_torque_idx] * 500.0f);
                        break;
                    }
                    case TorqueMode::FWD: {
                        action_signal.front_torque_mNm = static_cast<int16_t>(action[action_torque_idx] * 1000.0f);
                        action_signal.back_torque_mNm = 0;
                        break;
                    }
                    case TorqueMode::RWD: {
                        action_signal.front_torque_mNm = 0;
                        action_signal.back_torque_mNm = static_cast<int16_t>(action[action_torque_idx] * 1000.0f);
                        break;
                    }
                }
                action_signal.front_torque_mNm = abs(action_signal.front_torque_mNm);
                action_signal.back_torque_mNm = abs(action_signal.back_torque_mNm);
                if (action[action_torque_idx] < 0) {
                    action_signal.velocity_rpm = 0;
                } else {
                    action_signal.velocity_rpm = can_max_velocity_rpm;
                }

                action_signal.rack_displacement_adc = swangle_to_adc(action[action_swangle_idx]);
                return action_signal; 
            }

            std::thread ControllerNode::launch_can() {
                return std::thread {
                [this]
                {
                    while (rclcpp::ok)
                    {
                        std::this_thread::sleep_for(std::chrono::milliseconds(aim_signal_period_ms));
                        auto current_time = std::chrono::high_resolution_clock::now();
                        // std::cout << "Time: " << std::chrono::duration_cast<std::chrono::milliseconds>(current_time.time_since_epoch()).count() % 100 << std::endl;

                        // if (std::chrono::duration_cast<std::chrono::milliseconds>(current_time.time_since_epoch()).count() % 100 < 5) {
                        auto start = std::chrono::steady_clock::now();
                        ActionSignal last_action_signal = m_last_action_signal;
                        if constexpr (send_to_can)
                        {
                            sendPIDConstants(2.5, 0.0);
                            // FYI, velocity_rpm is determined from the speed threshold
                            sendControlAction(last_action_signal.front_torque_mNm, last_action_signal.back_torque_mNm, last_action_signal.velocity_rpm, last_action_signal.rack_displacement_adc);
                            RCLCPP_DEBUG(get_logger(), "Sending action signal %d, %d, %u, %u\n", last_action_signal.front_torque_mNm, last_action_signal.back_torque_mNm, last_action_signal.velocity_rpm, last_action_signal.rack_displacement_adc);
                        }
                        auto end = std::chrono::steady_clock::now();
                        RCLCPP_DEBUG(get_logger(), "sendControlAction took %ld ms", std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count());
                    }
                    std::cout << "I just got terminated in another way lol\n";
                    send_finished_ignore_error();
                }};
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

                        rclcpp::Time projection_time = get_clock()->now();
                        rclcpp::Time time_to_project_till;
                        if constexpr (testing_on_rosbag)
                        {
                            time_to_project_till = m_last_spline_msg.header.stamp;
                        }
                        else
                        {
                            time_to_project_till = projection_time;
                        }



                        m_state_estimator->sync_to_device(time_to_project_till);

                        if constexpr (!testing_on_rosbag) {
                            SplineMsg spline_msg = m_last_spline_msg;
                            spline_msg.header.stamp = time_to_project_till;
                            m_spline_republisher->publish(spline_msg);
                        }


                        // we don't need the host state anymore, so release the lock and let state callbacks proceed
                        state_lock.unlock();

                        // run mppi, and write action to the write buffer
                        Action action = m_mppi_controller->generate_action();

                        auto action_publish_time = get_clock()->now();
                        ActionMsg action_msg = action_to_msg(action);
                        action_msg.header.stamp = action_publish_time;
                        if constexpr (!testing_on_rosbag) {
                            m_action_publisher->publish(action_msg);
                            m_state_estimator->record_control_action(action, action_publish_time);
                        }
                        
                        m_last_action_signal = action_to_signal(action);


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