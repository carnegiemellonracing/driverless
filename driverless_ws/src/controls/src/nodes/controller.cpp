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
#include <filesystem>
#include <map>

#ifdef DISPLAY
#include <display/display.hpp>
#endif

#include "controller.hpp"
#include <sstream>
#include <utils/general_utils.hpp>
#include <utils/ros_utils.hpp>
#include <state/naive_state_tracker.hpp>


// This is to fit into the ROS API
void send_finished_ignore_error() {
    std::cout << "I just got terminated lol\n";
    sendFinishedCommand();
}

namespace controls {
    // global runtime variables
    float approx_propogation_delay;
    bool follow_midline_only;
    bool testing_on_rosbag;
    bool ingest_midline;
    bool send_to_can;
    bool display_on;
    StateProjectionMode state_projection_mode;
    bool publish_spline;
    bool log_state_projection_history;
    bool no_midline_controller;




    namespace nodes {
        ControllerNode::ControllerNode(
            std::shared_ptr<state::StateEstimator> state_estimator,
            std::shared_ptr<mppi::MppiController> mppi_controller)
            : Node{controller_node_name},

              m_state_estimator{std::move(state_estimator)},
              m_mppi_controller{std::move(mppi_controller)},

              m_perc_cones_republisher{
                  create_publisher<ConeMsg>(
                      republished_perc_cones_topic_name,
                      republished_perc_cones_qos)},

              m_action_publisher{
                  // TODO: <ActionMsg>
                  create_publisher<interfaces::msg::ControlAction>(
                      control_action_topic_name,
                      control_action_qos)},

              m_info_publisher{
                  create_publisher<InfoMsg>(
                      controller_info_topic_name,
                      controller_info_qos)},

              m_spline_publisher{
                create_publisher<SplineMsg>(
                    spline_topic_name,
                    spline_qos
                )
              },

              m_data_trajectory_log{"mppi_inputs.txt", std::ios::out},
              m_p_value{0.1}
        {
            // create a callback group that prevents state and spline callbacks from being executed concurrently
            rclcpp::CallbackGroup::SharedPtr main_control_loop_callback_group{
                create_callback_group(rclcpp::CallbackGroupType::MutuallyExclusive)};
            
            rclcpp::CallbackGroup::SharedPtr auxiliary_state_callback_group{
                create_callback_group(rclcpp::CallbackGroupType::MutuallyExclusive)};

            rclcpp::SubscriptionOptions main_control_loop_options{};
            main_control_loop_options.callback_group = main_control_loop_callback_group;


            rclcpp::SubscriptionOptions auxiliary_state_options{};
            auxiliary_state_options.callback_group = auxiliary_state_callback_group;

            const char* desired_cone_topic_name;
            if (testing_on_rosbag) {
                desired_cone_topic_name = republished_perc_cones_topic_name;
            } else {
                desired_cone_topic_name = cone_topic_name;
            }


                m_cone_subscription = create_subscription<ConeMsg>(
                    desired_cone_topic_name, spline_qos, //was cone_qos but that didn't exist, publisher uses spline_qos
                    [this] (const ConeMsg::SharedPtr msg) { cone_callback(*msg); },
                    main_control_loop_options
                );

                m_world_twist_subscription = create_subscription<TwistMsg>(
                    world_twist_topic_name, world_twist_qos,
                    [this] (const TwistMsg::SharedPtr msg) { world_twist_callback(*msg); },
                    auxiliary_state_options
                );

            m_world_pose_subscription = create_subscription<PoseMsg>(
                world_pose_topic_name, world_pose_qos,
                [this](const PoseMsg::SharedPtr msg)
                { world_pose_callback(*msg); },
                auxiliary_state_options);

            m_pid_subscription = create_subscription<PIDMsg>(
                pid_topic_name, pid_qos,
                [this](const PIDMsg::SharedPtr msg)
                { pid_callback(*msg); },
                auxiliary_state_options);  

            m_imu_accel_subscription = create_subscription<IMUAccelerationMsg>(
                imu_accel_topic_name, imu_accel_qos,
                [this](const IMUAccelerationMsg::SharedPtr msg)
                { imu_accel_callback(*msg); },
                auxiliary_state_options);

            m_world_quat_subscription = create_subscription<QuatMsg>(
                world_quat_topic_name, world_quat_qos,
                [this](const QuatMsg::SharedPtr msg)
                { world_quat_callback(*msg); },
                auxiliary_state_options);

            m_position_lla_subscription = create_subscription<PositionLLAMsg>(
                world_position_lla_topic_name, world_pose_qos,
                [this](const PositionLLAMsg::SharedPtr msg)
                { world_position_lla_callback(*msg); },
                auxiliary_state_options);

            if (testing_on_rosbag) {
                m_action_subscription = create_subscription<ActionMsg>(
                control_action_topic_name, control_action_qos,
                [this](const ActionMsg::SharedPtr msg)
                { rosbag_action_callback(*msg); },
                auxiliary_state_options);
            }

            
            // TODO: m_state_mut never gets initialized? I guess default construction is alright;
            if (send_to_can) {
                int can_init_result = initializeCan();
                if (can_init_result < 0)
                {
                    std::cout << "Can failed to initialize with error " << can_init_result << std::endl;
                    throw std::runtime_error("Failed to initialize can");
                }
                sendPIDConstants(default_p, default_feedforward);
            }

            if constexpr (testing_on_breezway) {
                m_last_imu_acceleration_time = get_clock()->now();
                m_last_x_velocity = 0.0f;
                m_last_y_velocity = 0.0f;
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
            std::string progress_bar(float current, float min, float max, int width) {
                if (current < min || current > max) {
                    return "OUT OF BOUNDS";
                }
                float progress = (current - min)/(max - min);
                int pos = static_cast<int>(progress * width);
                std::string bar;
                if (pos < width / 2)
                {
                    bar = "[" + std::string(std::max(0, pos), ' ') + "\033[31m" + std::string(width / 2 - pos, '|') + "\033[0m" + "|" + std::string(width / 2, ' ') + "]";
                }
                else if (pos > width / 2)
                {
                    bar = "[" + std::string(width / 2, ' ') + "|" + "\033[32m" + std::string(std::max(0, pos - width / 2), '|') + "\033[0m" + std::string(width - pos, ' ') + "]";
                }
                else
                {
                    bar = "[" + std::string(width / 2, ' ') + "|" + std::string(width / 2, ' ') + "]";
                }
                return bar;
            }
            std::string swangle_bar(float current, float min, float max, int width) {
                if (current < min || current > max)
                {
                    return "OUT OF BOUNDS";
                }
                float progress = (current - min)/(max - min);
                int pos = static_cast<int>(progress * width);
                std::string bar;
                if (pos < width / 2)
                {
                    bar = "[" + std::string(std::max(0, pos), ' ') + "\033[31m+\033[0m" + std::string(width / 2 - pos-1, ' ') + "|" + std::string(width / 2, ' ') + "]";
                }
                else if (pos > width / 2){
                    bar = "[" + std::string(width / 2, ' ') + "|" + std::string(std::max(0, pos - width / 2), ' ') + "\033[32m+\033[0m" + std::string(std::max(0, width - pos - 1), ' ') + "]";
                }
                else{
                    bar = "[" + std::string(width / 2, ' ') + "\033[33m+\033[0m" + std::string(width / 2, ' ') + "]";
                }

                return bar;
            }
            State ControllerNode::get_state_under_strategy(rclcpp::Time current_time) {
                switch (state_projection_mode) {
                    case StateProjectionMode::MODEL_MULTISET: {
                        std::optional<State> proj_curr_state_opt = m_state_estimator->project_state(current_time);
                        if (proj_curr_state_opt.has_value()) {
                            return proj_curr_state_opt.value();
                        } else {
                            RCLCPP_WARN(get_logger(), "Failed to project state, using naive speed only");
                            return {0.0f, 0.0f, M_PI_2, m_last_speed};
                        }
                        break;
                    }
                    case StateProjectionMode::NAIVE_SPEED_ONLY:
                        if (!testing_on_rosbag) { // TODO: Move this if statement to before get_state_under_strategy is called, allowing current_time to be siphoned off the republished perc cone message instead of the normal one
                            m_state_estimator->project_state(current_time);
                        }
                        return {0.0f, 0.0f, M_PI_2, m_last_speed};
                        break;
                    case StateProjectionMode::POSITIONLLA_YAW_SPEED: {
                        m_state_estimator->project_state(current_time);
                        std::optional<PositionAndYaw> position_and_yaw_opt = m_naive_state_tracker.get_relative_position_and_yaw();
                        if (position_and_yaw_opt.has_value()) {
                            return {position_and_yaw_opt.value().first.first, position_and_yaw_opt.value().first.second, position_and_yaw_opt.value().second, m_last_speed};
                        } else {
                            RCLCPP_WARN(get_logger(), "Failed to get position and yaw, using naive speed only");
                            return {0.0f, 0.0f, M_PI_2, m_last_speed};
                        }
                        break;
                    }
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
                // ss << "Length of unknown color cones: " << cone_msg.unknown_color_cones.size() << std::endl;
                // ss << "Length of big orange cones: " << cone_msg.big_orange_cones.size() << std::endl;
                RCLCPP_DEBUG(get_logger(), ss.str().c_str());


                // save for info publishing later, since might be changed during iteration
                rclcpp::Time lidar_points_seen_time = cone_msg.header.stamp;

                rclcpp::Time cone_callback_time = get_clock()->now();
                rclcpp::Time time_to_project_till;
                if (testing_on_rosbag)
                {
                    time_to_project_till = cone_msg.controller_receive_time;
                }
                else
                {
                    time_to_project_till = cone_callback_time;
                }

                // record time to estimate speed of the entire pipeline
                auto start_time = std::chrono::high_resolution_clock::now();

                // Process cones and report timing
                Action action {};
                State proj_curr_state {};
                std::chrono::_V2::system_clock::time_point cone_process_start, cone_process_end, project_start, project_end, sync_start, sync_end, gen_action_start, gen_action_end;

                try {
                    {
                        std::lock_guard<std::mutex> guard {m_state_mut};
                        if (cone_msg.blue_cones.size() == 0 && cone_msg.yellow_cones.size() == 0) {
                            throw ControllerError("No blue or yellow cones to process");
                        }
                        cone_process_start = std::chrono::high_resolution_clock::now();
                        float svm_time = m_state_estimator->on_cone(cone_msg, m_spline_publisher);
                        m_naive_state_tracker.record_cone_seen_time(cone_msg.header.stamp);
                        cone_process_end = std::chrono::high_resolution_clock::now();
                        m_last_cone_process_time = std::chrono::duration_cast<std::chrono::milliseconds>(cone_process_end - cone_process_start).count();
                        m_last_svm_time = svm_time;

                        RCLCPP_DEBUG(get_logger(), "mppi iteration beginning");


                        // send state to device (i.e. cuda globals)
                        // (also serves to lock state since nothing else updates gpu state)
                        RCLCPP_DEBUG(get_logger(), "syncing state to device");
                        project_start = std::chrono::high_resolution_clock::now();


                        proj_curr_state = get_state_under_strategy(time_to_project_till);
                        project_end = std::chrono::high_resolution_clock::now();
                        if (!testing_on_rosbag && republish_perc_cones) {
                            ConeMsg new_perc_cone_msg = cone_msg;
                            new_perc_cone_msg.controller_receive_time = time_to_project_till;
                            m_perc_cones_republisher->publish(new_perc_cone_msg);
                        }

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

                auto action_time = get_clock()->now(); 
                if (!testing_on_rosbag) {
                    publish_action(action, action_time);
                    m_state_estimator->record_control_action(action, action_time);
                }


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

                // calculate and print time elapsed
                auto finish_time = std::chrono::high_resolution_clock::now();
                auto time_elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(
                    finish_time - start_time);

                // can't use high res clock since need to be aligned with other nodes
                RCLCPP_WARN(get_logger(), "Current time %ld", get_clock()->now().nanoseconds());
                RCLCPP_WARN(get_logger(), "Cone arrival time %ld", lidar_points_seen_time.nanoseconds());

                rclcpp::Duration total_time_elapsed (0, 0);
                if (testing_on_rosbag) {
                    // The first measures how long this function took, the second measures how long it took from lidar receiving points to this function getting called.
                    total_time_elapsed = (get_clock()->now() - cone_callback_time) - (rclcpp::Time(cone_msg.controller_receive_time) - lidar_points_seen_time);
                } else {
                    total_time_elapsed = get_clock()->now() - lidar_points_seen_time;
                }

                interfaces::msg::ControllerInfo info{};
                info.header.stamp = action_time;
                info.action = action_to_msg(action);
                info.proj_state = state_to_msg(proj_curr_state);
                info.projection_latency_ms = (std::chrono::duration_cast<std::chrono::milliseconds>(project_end - project_start)).count(); // duration_vec[0].count();
                info.render_latency_ms = (std::chrono::duration_cast<std::chrono::milliseconds>(sync_end - sync_start)).count();
                info.mppi_latency_ms = (std::chrono::duration_cast<std::chrono::milliseconds>(gen_action_end - gen_action_start)).count();
                info.latency_ms = time_elapsed.count();
                info.total_latency_ms = total_time_elapsed.seconds() * 1000 + total_time_elapsed.nanoseconds() / 1000000;


                publish_and_print_info(info, error_str);

            }

            void ControllerNode::world_twist_callback(const TwistMsg &twist_msg) {
                RCLCPP_DEBUG(get_logger(), "Received twist");

                if constexpr (!testing_on_breezway) {

                {
                    std::lock_guard<std::mutex> guard {m_state_mut};
                    m_state_estimator->on_twist(twist_msg, twist_msg.header.stamp);
                    m_last_speed = twist_msg_to_speed(twist_msg);
                }

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
                if (send_to_can) {
                    sendPIDConstants(m_p_value, 0);
                }
                RCLCPP_WARN(get_logger(), "send Kp %f", m_p_value);
            }

            void ControllerNode::imu_accel_callback(const IMUAccelerationMsg& imu_accel_msg) {
                if constexpr (testing_on_breezway) {
                    float time_since_last_imu_acceleration_s = get_clock()->now().seconds() - m_last_imu_acceleration_time.seconds();
                    // GPS and controller/dv frame are different
                    m_last_x_velocity = m_last_x_velocity - imu_accel_msg.vector.y * time_since_last_imu_acceleration_s;
                    m_last_y_velocity = m_last_y_velocity + imu_accel_msg.vector.x * time_since_last_imu_acceleration_s;
                    m_last_imu_acceleration_time = get_clock()->now();
                    m_last_speed = std::sqrt(m_last_x_velocity * m_last_x_velocity + m_last_y_velocity * m_last_y_velocity);
                }
            }
            
            void ControllerNode::world_quat_callback(const QuatMsg& quat_msg) {
                if (state_projection_mode == StateProjectionMode::POSITIONLLA_YAW_SPEED) {
                    m_naive_state_tracker.record_quaternion(quat_msg);
                }
                m_state_estimator->on_quat(quat_msg);
            }

            void ControllerNode::world_position_lla_callback(const PositionLLAMsg& position_lla_msg) {
                if (state_projection_mode == StateProjectionMode::POSITIONLLA_YAW_SPEED) {
                    m_naive_state_tracker.record_positionlla(position_lla_msg);
                }
                m_state_estimator->on_position_lla(position_lla_msg);
            }

            void ControllerNode::publish_action(const Action& action, rclcpp::Time current_time) {
                auto msg = action_to_msg(action);
                msg.header.stamp = current_time;
                m_action_publisher->publish(msg);
            }


            static Action msg_to_action(const ActionMsg& msg) {
                Action action;
                action[action_swangle_idx] = msg.swangle;
                action[action_torque_idx] = msg.torque_fl + msg.torque_fr + msg.torque_rl + msg.torque_rr;
                return action;
            }

            void ControllerNode::rosbag_action_callback(const ActionMsg& rosbag_action_msg) {
                std::lock_guard<std::mutex> guard {m_state_mut};
                m_state_estimator->record_control_action(msg_to_action(rosbag_action_msg), rosbag_action_msg.header.stamp);
            }

            static uint8_t swangle_to_rackdisplacement(float swangle) {
                return 0;
            }




            ControllerNode::ActionSignal ControllerNode::action_to_signal(Action action) {
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
                if (!testing_on_rosbag) {
                    m_info_publisher->publish(info);

                }
                
                std::stringstream ss;

                ss
                    << "Action:\n"
                    << "  swangle (rad): " << info.action.swangle << "\n"
                    << swangle_bar(info.action.swangle,min_swangle, max_swangle,40) << "\n"
                    << progress_bar(info.action.torque_fl, min_torque, max_torque, 40) << "\n"
                    << progress_bar(info.action.torque_fr, min_torque, max_torque, 40) << "\n"
                    << progress_bar(info.action.torque_rl, min_torque, max_torque, 40) << "\n"
                    << progress_bar(info.action.torque_rr, min_torque, max_torque, 40) << "\n"
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
                    << get_clock()->get_clock_type()
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
                                if (send_to_can) {
                                      sendPIDConstants(default_p, default_feedforward);

                                    // FYI, velocity_rpm is determined from the speed threshold
                                    sendControlAction(last_action_signal.front_torque_mNm, last_action_signal.back_torque_mNm, last_action_signal.velocity_rpm, last_action_signal.rack_displacement_adc);
                                    RCLCPP_DEBUG(get_logger(), "Sending action signal %d, %d, %u, %u\n", last_action_signal.front_torque_mNm, last_action_signal.back_torque_mNm, last_action_signal.velocity_rpm, last_action_signal.rack_displacement_adc);
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

std::string trim(const std::string &str)
{
    const size_t first = str.find_first_not_of(" \t");

    if (first == std::string::npos)
        return "";

    const size_t last = str.find_last_not_of(" \t");
    const size_t length = last - first + 1;

    return str.substr(first, length);
}


static int process_config_file(std::string config_file_path) {
    std::string config_file_base_path = std::string{getenv("DRIVERLESS")} + "/driverless_ws/src/controls/src/nodes/configs/";
    std::string config_file_full_path = config_file_base_path + config_file_path;

    std::map<std::string, std::string> config_dict;

    std::cout << "Looking for " << config_file_full_path << std::endl;
    if (std::filesystem::exists(config_file_full_path))
    {
        std::cout << "Config file found.\n";
    }
    else
    {
        std::cout << "Config file not found.\n";
        return 1;
    }

    std::ifstream conf_file(config_file_full_path);

    if (conf_file.is_open())
    {
        std::string line;
        while (std::getline(conf_file, line, '\n'))
        {
            std::cout << line << std::endl;
            if (line.empty() || line[0] == '#')
            {
                continue;
            }
            else
            {
                int delim_position = line.find(':');
                std::string key = trim(line.substr(0, delim_position));
                std::string val = trim(line.substr(delim_position + 1));
                // std::cout << line << " " << key << " " << val <<std::endl;
                config_dict[key] = val;
            }
        }
    }

    approx_propogation_delay = std::stof(config_dict["approx_propogation_delay"]);
    follow_midline_only = config_dict["follow_midline_only"] == "true" ? true : false;

    testing_on_rosbag = config_dict["testing_on_rosbag"] == "true" ? true : false;
    ingest_midline = config_dict["ingest_midline"] == "true" ? true : false;
    send_to_can = config_dict["send_to_can"] == "true" ? true : false;
    display_on = config_dict["display_on"] == "true" ? true : false;
    no_midline_controller = config_dict["no_midline_controller"] == "true" ? true : false;
    if (config_dict["state_projection_mode"] == "model_multiset") {
        state_projection_mode = StateProjectionMode::MODEL_MULTISET;
    } else if (config_dict["state_projection_mode"] == "naive_speed_only") {
        state_projection_mode = StateProjectionMode::NAIVE_SPEED_ONLY;
    } else if (config_dict["state_projection_mode"] == "positionlla_yaw_speed") {
    }
    
    publish_spline = config_dict["publish_spline"] == "true" ? true : false;
    log_state_projection_history = config_dict["log_state_projection_history"] == "true" ? true : false;
    return 0;
}

}



int main(int argc, char *argv[]) {
    using namespace controls;
    std::string default_config_path = "controls_default_config";
    std::string config_file_path;
    if (argc < 2)
    {
        std::cout << "Perhaps you didn't pass in the config file name, using default" << std::endl;
        config_file_path = default_config_path;
    }
    else
    {
        std::cout << "Config file passed." << std::endl;
        config_file_path = argv[1];
    }
    if (process_config_file(config_file_path) != 0) {
        return 1;
    }

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
            if (!display_on) {
                return;
            }
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