#include <chrono>
#include <mppi/mppi.hpp>
#include <mutex>
#include <rclcpp/rclcpp.hpp>
#include <types.hpp>
#include <constants.hpp>
#include <interfaces/msg/control_action.hpp>
#include <state/state_estimator.hpp>
#include <iostream>
#include <fstream>
#include <can/cmr_can.h> // Now it's part of git
#include <chrono>

#ifdef DISPLAY
#include <display/display.hpp>
#endif

#include "controller.hpp"
#include <sstream>
#include <utils/general_utils.hpp>
#include <utils/ros_utils.hpp>


// This is to fit into the ROS API
void send_finished_ignore_error() {
    std::cout << "I just got terminated lol\n";
    sendFinishedCommand();
}

namespace controls {
    namespace nodes {
        ControllerNode::ControllerNode(
            std::shared_ptr<state::StateEstimator> state_estimator,
            std::shared_ptr<mppi::MppiController> mppi_controller)
            : Node{controller_node_name},

              m_state_estimator{std::move(state_estimator)},
              m_mppi_controller{std::move(mppi_controller)},

              m_action_publisher{
                  // TODO: <ActionMsg>
                  create_publisher<interfaces::msg::ControlAction>(
                      control_action_topic_name,
                      control_action_qos)},

              m_info_publisher{
                  create_publisher<InfoMsg>(
                      controller_info_topic_name,
                      controller_info_qos)
              },

              m_data_trajectory_log {"mppi_inputs.txt", std::ios::out},
              m_p_value {0.1}
        {
            // create a callback group that prevents state and spline callbacks from being executed concurrently
            rclcpp::CallbackGroup::SharedPtr state_estimation_callback_group{
                create_callback_group(rclcpp::CallbackGroupType::MutuallyExclusive)};

            rclcpp::SubscriptionOptions options{};
            options.callback_group = state_estimation_callback_group;


                m_cone_subscription = create_subscription<ConeMsg>(
                    cone_topic_name, spline_qos, //was cone_qos but that didn't exist, publisher uses spline_qos
                    [this] (const ConeMsg::SharedPtr msg) { cone_callback(*msg); },
                    options
                );

                m_world_twist_subscription = create_subscription<TwistMsg>(
                    world_twist_topic_name, world_twist_qos,
                    [this] (const TwistMsg::SharedPtr msg) { world_twist_callback(*msg); },
                    options
                );

            m_world_pose_subscription = create_subscription<PoseMsg>(
                world_pose_topic_name, world_pose_qos,
                [this](const PoseMsg::SharedPtr msg)
                { world_pose_callback(*msg); },
                options);

            m_pid_subscription = create_subscription<PIDMsg>(
                pid_topic_name, pid_qos,
                [this](const PIDMsg::SharedPtr msg)
                { pid_callback(*msg); },
                options);  
                
            
            // TODO: m_state_mut never gets initialized? I guess default construction is alright;
            if constexpr (send_to_can) {
                int can_init_result = initializeCan();
                if (can_init_result < 0)
                {
                    std::cout << "Can failed to initialize with error " << can_init_result << std::endl;
                    throw std::runtime_error("Failed to initialize can");
                }
                sendPIDConstants(default_p, default_feedforward);
            }


            launch_aim_communication().detach();
        }


#ifdef DATA
            std::string vector_to_parseable_string(const std::vector<glm::fvec2> &vec) {
                std::stringstream ss;
                for (int i = 0; i < vec.size() - 1; i++)
                {
                    ss << vec[i].x << " " << vec[i].y << ",";
                }
                ss << vec[vec.size() - 1].x << " " << vec[vec.size() - 1].y;

                return ss.str();
            }
#endif
            State ControllerNode::get_state_under_strategy() {
                switch (projection_mode) {
                    case StateProjectionMode::MODEL_MULTISET: {
                        std::optional<State> proj_curr_state_opt = m_state_estimator->project_state(get_clock()->now());
                        if (proj_curr_state_opt.has_value()) {
                            return proj_curr_state_opt.value();
                        } else {
                            RCLCPP_WARN(get_logger(), "Failed to project state, using naive speed only");
                            return {0.0f, 0.0f, M_PI_2, m_last_speed};
                        }
                        break;
                    }
                    case StateProjectionMode::NAIVE_SPEED_ONLY:
                        return {0.0f, 0.0f, M_PI_2, m_last_speed};
                        break;
                    case StateProjectionMode::POSITIONLLA_YAW_SPEED:
                        return {}; //TODO: implement the binary set lol
                        break;
                    default:
                        RCLCPP_WARN(get_logger(), "unknown state projection mode");
                        return {};
                }
            }


            void ControllerNode::cone_callback(const ConeMsg& cone_msg) {

                m_mppi_controller->set_follow_midline_only(follow_midline_only);
                m_state_estimator->set_follow_midline_only(follow_midline_only);

                                
                std::stringstream ss;
                ss << "Received cones: " << std::endl;
                ss << "Length of blue cones: " << cone_msg.blue_cones.size() << std::endl;
                ss << "Length of yellow cones: " << cone_msg.yellow_cones.size() << std::endl;
                ss << "Length of orange cones: " << cone_msg.orange_cones.size() << std::endl;
                ss << "Length of unknown color cones: " << cone_msg.unknown_color_cones.size() << std::endl;
                ss << "Length of big orange cones: " << cone_msg.big_orange_cones.size() << std::endl;
                RCLCPP_DEBUG(get_logger(), ss.str().c_str());


                // save for info publishing later, since might be changed during iteration
                rclcpp::Time cone_arrival_time = cone_msg.header.stamp;

                // record time to estimate speed of the entire pipeline
                auto start_time = std::chrono::high_resolution_clock::now();

                // Process cones and report timing
                Action action {};
                State proj_curr_state {};
                std::chrono::_V2::system_clock::time_point cone_process_start, cone_process_end, project_start, project_end, sync_start, sync_end, gen_action_start, gen_action_end;

                try {
                    {
                        std::lock_guard<std::mutex> guard {m_state_mut};
                        cone_process_start = std::chrono::high_resolution_clock::now();
                        float svm_time = m_state_estimator->on_cone(cone_msg);
                        cone_process_end = std::chrono::high_resolution_clock::now();
                        m_last_cone_process_time = std::chrono::duration_cast<std::chrono::milliseconds>(cone_process_end - cone_process_start).count();
                        m_last_svm_time = svm_time;

                        RCLCPP_DEBUG(get_logger(), "mppi iteration beginning");


                        // send state to device (i.e. cuda globals)
                        // (also serves to lock state since nothing else updates gpu state)
                        RCLCPP_DEBUG(get_logger(), "syncing state to device");
                        project_start = std::chrono::high_resolution_clock::now();
                        proj_curr_state = get_state_under_strategy();
                        project_end = std::chrono::high_resolution_clock::now();

                        // * Let state callbacks proceed, so unlock the state mutex
                    }

                    // send state to device
                    sync_start = std::chrono::high_resolution_clock::now();
                    m_state_estimator->render_and_sync(proj_curr_state);
                    sync_end = std::chrono::high_resolution_clock::now();

                    // we don't need the host state anymore, so release the lock and let state callbacks proceed

                    // run mppi, and write action to the write buffer
                    gen_action_start = std::chrono::high_resolution_clock::now();
                    action = m_mppi_controller->generate_action();
                    gen_action_end = std::chrono::high_resolution_clock::now();

                } catch (const ControllerError& e) {
                    RCLCPP_ERROR(get_logger(), "Error generating action: %s, taking next averaged action instead", e.what());
                    gen_action_start = std::chrono::high_resolution_clock::now();
                    action = m_mppi_controller->get_next_averaged_action();
                    gen_action_end = std::chrono::high_resolution_clock::now();
                }

                publish_action(action);
                m_last_action_signal = action_to_signal(action);
                std::string error_str;

#ifdef DATA
                std::stringstream parameters_ss;
                parameters_ss << "Swangle range: " << 19 * M_PI / 180 * 2 << "\nThrottle range: " << saturating_motor_torque * 2 << "\n";
                RCLCPP_WARN_ONCE(get_logger(), parameters_ss.str().c_str());

                std::vector<Action> percentage_diff_trajectory = m_mppi_controller->m_percentage_diff_trajectory;
                std ::vector<Action> averaged_trajectory = m_mppi_controller->m_averaged_trajectory;
                std::vector<Action> last_action_trajectory = m_mppi_controller->m_last_action_trajectory_logging;
                auto diff_statistics = m_mppi_controller->m_diff_statistics;

                // write the spline
                std::vector<glm::fvec2> frames = m_state_estimator->get_spline_frames();
                std::vector<glm::fvec2> left_cones = m_state_estimator->get_left_cone_points();
                std::vector<glm::fvec2> right_cones = m_state_estimator->get_right_cone_points();

                // create debugging string
                ss.clear();
                ss << "Spline (MPPI Input): \n";
                ss << points_to_string(frames) << "\n";
                ss << "Left cones (MPPI Input): \n";
                ss << points_to_string(left_cones) << "\n";
                ss << "Right cones (MPPI Input): \n";
                ss << points_to_string(right_cones) << "\n";
                ss << "Last action_trajectory (MPPI Input/Guess): \n";
                for (const auto &action : last_action_trajectory)
                {
                    ss << "[" << action[0] << ", " << action[1] << "], ";
                }
                ss << "\nAveraged Trajectory (MPPI Output): \n";
                for (const auto &action : averaged_trajectory)
                {
                    ss << "[" << action[0] << ", " << action[1] << "], ";
                }
                ss << "\nPercentage Difference between Guess and Result: \n";
                for (const auto &action : percentage_diff_trajectory)
                {
                    ss << "[" << action[0] << ", " << action[1] << "], ";
                }
                ss << "\nMean Swangle Error: " << diff_statistics.mean_swangle << std::endl;
                ss << "Mean Torque Error: " << diff_statistics.mean_throttle << std::endl;
                ss << "Max Swangle Error: " << diff_statistics.max_swangle << std::endl;
                ss << "Max Torque Error: " << diff_statistics.max_throttle << std::endl;
                error_str = ss.str();

                // writing to logging file for sampling exploration
                // write the state
                for (int i = 0; i < state_dims - 1; i++)
                {
                    m_data_trajectory_log << proj_curr_state[i] << ",";
                }
                m_data_trajectory_log << proj_curr_state[state_dims - 1] << "|";
                m_data_trajectory_log << vector_to_parseable_string(frames) << "|";
                m_data_trajectory_log << vector_to_parseable_string(left_cones) << "|";
                m_data_trajectory_log << vector_to_parseable_string(right_cones) << "|";
                // write the guess trajectory
                for (int i = 0; i < last_action_trajectory.size() - 1; i++)
                {
                    m_data_trajectory_log << last_action_trajectory[i][0] << " " << last_action_trajectory[i][1] << ",";
                }
                m_data_trajectory_log << last_action_trajectory[last_action_trajectory.size() - 1][0] << " " << last_action_trajectory[last_action_trajectory.size() - 1][1] << "\n";

#endif

                // this can happen concurrently with another state estimator callback, but the internal lock
                // protects it.
                m_state_estimator->record_control_action(action, get_clock()->now());

                // calculate and print time elapsed
                auto finish_time = std::chrono::high_resolution_clock::now();
                auto time_elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(
                    finish_time - start_time);

                // can't use high res clock since need to be aligned with other nodes
                auto total_time_elapsed = (get_clock()->now().nanoseconds() - cone_arrival_time.nanoseconds()) / 1000000;

                interfaces::msg::ControllerInfo info{};
                info.action = action_to_msg(action);
                info.proj_state = state_to_msg(proj_curr_state);
                info.projection_latency_ms = (std::chrono::duration_cast<std::chrono::milliseconds>(project_end - project_start)).count(); // duration_vec[0].count();
                info.render_latency_ms = (std::chrono::duration_cast<std::chrono::milliseconds>(sync_end - sync_start)).count();
                info.mppi_latency_ms = (std::chrono::duration_cast<std::chrono::milliseconds>(gen_action_end - gen_action_start)).count();
                info.latency_ms = time_elapsed.count();
                info.total_latency_ms = total_time_elapsed;

                publish_and_print_info(info, error_str);
            }

            void ControllerNode::world_twist_callback(const TwistMsg &twist_msg) {
                RCLCPP_DEBUG(get_logger(), "Received twist");


                {
                    std::lock_guard<std::mutex> guard {m_state_mut};
                    m_state_estimator->on_twist(twist_msg, twist_msg.header.stamp);
                    m_last_speed = twist_msg_to_speed(twist_msg);
                }

            }

            void ControllerNode::world_pose_callback(const PoseMsg &pose_msg) {
                RCLCPP_DEBUG(get_logger(), "Received pose");


                {
                    std::lock_guard<std::mutex> guard {m_state_mut};
                    m_state_estimator->on_pose(pose_msg);
                }

            }

            void ControllerNode::pid_callback(const PIDMsg& pid_msg) {
                m_p_value = pid_msg.x;
                if constexpr (send_to_can) {
                    sendPIDConstants(m_p_value, 0);
                }
                RCLCPP_WARN(get_logger(), "send Kp %f", m_p_value);
            }


            void ControllerNode::publish_action(const Action& action) {
                const auto msg = action_to_msg(action);
                m_action_publisher->publish(msg);
            }

            static uint8_t swangle_to_rackdisplacement(float swangle) {
                return 0;
            }



            ControllerNode::ActionSignal ControllerNode::action_to_signal(Action action) {
                ActionSignal action_signal;

                action_signal.front_torque_mNm = static_cast<int16_t>(action[action_torque_idx] * 500.0f);
                action_signal.back_torque_mNm = static_cast<int16_t>(action[action_torque_idx] * 500.0f);
                action_signal.velocity_rpm = can_max_velocity_rpm;
                action_signal.rack_displacement_adc = swangle_to_adc(action[action_swangle_idx]);
                return action_signal;   
            }



            ActionMsg ControllerNode::action_to_msg(const Action &action) {
                interfaces::msg::ControlAction msg;

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

            void ControllerNode::publish_and_print_info(interfaces::msg::ControllerInfo info, const std::string& additional_info) {
                m_info_publisher->publish(info);
                std::stringstream ss;

                ss
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
                    // << "Cone Processing Latency (ms)" << m_last_cone_process_time << "\n"
                    << "SVM Latency (ms): " << m_last_cone_process_time << "\n"
                    << "State Projection Latency (ms): " << info.projection_latency_ms << "\n"
                    << "OpenGL Render Latency (ms): " << info.render_latency_ms << "\n"
                    << "MPPI Step Latency (ms): " << info.mppi_latency_ms << "\n"
                    << "Controls Latency (ms): " << info.latency_ms << "\n"
                    << "Total Latency (ms): " << info.total_latency_ms << "\n"
                    << additional_info
                    << std::endl;

                std::string info_str = ss.str();

                std::cout << clear_term_sequence << info_str << std::flush;
                RCLCPP_INFO_STREAM(get_logger(), "mppi step complete. info:\n"
                                                     << info_str);
            }

            std::thread ControllerNode::launch_aim_communication()
            {
                return std::thread{
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
                                if constexpr (send_to_can) {
                                    // FYI, velocity_rpm is determined from the speed threshold
                                    sendControlAction(last_action_signal.front_torque_mNm, last_action_signal.back_torque_mNm, last_action_signal.velocity_rpm, last_action_signal.rack_displacement_adc);
                                }
                                auto end = std::chrono::steady_clock::now();
                                RCLCPP_DEBUG(get_logger(), "sendControlAction took %ld ms", std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count());
                            // }
                            // }
                        }
                        std::cout << "I just got terminated in another way lol\n";
                        send_finished_ignore_error();
                    }};
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

    // rclcpp::on_shutdown(send_finished_ignore_error); // Need to figure out how to gracefully exit the aim communication threadssl

    // instantiate node
    const auto node = std::make_shared<nodes::ControllerNode>(state_estimator, controller);
    std::cout << "controller node created" << std::endl;

    rclcpp::Logger logger = node->get_logger();
    LoggerFunc logger_func = [logger](const std::string& msg) {
        RCLCPP_DEBUG(logger, msg.c_str());
    };

    //TODO why is this not a nullptr
    state_estimator->set_logger(logger_func);
    state_estimator->set_logger_obj(logger);
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