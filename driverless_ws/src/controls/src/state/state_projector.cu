#include <state/state_projector.cuh>
#include <utils/general_utils.hpp>
#include <utils/cuda_macros.cuh>
#include <sstream>
#include <fstream>

namespace controls {
    namespace state {

        StateProjector::StateProjector() : m_logger_obj {rclcpp::get_logger("")} {
            std::string log_location = getenv("ROS_LOG_DIR") + std::string{"/state_projection_history.txt"};
            std::fstream overwrite_fs{log_location};
            overwrite_fs << "\n";
        }
            
        std::string StateProjector::history_to_string() const {
            std::stringstream ss;
            ss << "---BEGIN HISTORY---" << std::endl;
            for (const Record& record : m_history_since_pose) {
                ss << "[1]Time: " << record.time.nanoseconds();
                switch (record.type) {
                    case Record::Type::Action:
                        ss << "Action: " << record.action[0] << ", " << record.action[1] << std::endl;
                        break;

                    case Record::Type::Speed:
                        ss << "Speed: " << record.speed << std::endl;
                        break;

                    case Record::Type::Pose:
                        ss << "Pose: " << record.pose.x << ", " << record.pose.y << ", " << record.pose.yaw << std::endl;
                        break;

                    case Record::Type::PositionLLA:
                        ss << "PositionLLAX: " << record.position_lla.x << std::endl;
                        ss << "[1]Time: " << record.time.nanoseconds();
                        ss << "PositionLLAY: " << record.position_lla.y << std::endl;
                        break;

                    case Record::Type::Yaw:
                        ss << "Yaw: " << record.yaw << std::endl;
                        break;

                    default:
                        throw new std::runtime_error("bruh. invalid record type bruh. (in print history)");
                }
            }
            ss << "---END HISTORY---" << std::endl;
            return ss.str();
        }

        void StateProjector::record_action(Action action, rclcpp::Time time) {
            RCLCPP_DEBUG_STREAM(m_logger_obj, "Recording action " << action[0] << ", " << action[1] << " at time " << time.nanoseconds());

            if (m_pose_record.has_value()) {
                if (time >= m_pose_record.value().time) {
                m_history_since_pose.insert(Record {
                    .action = action,
                    .time = time,
                    .type = Record::Type::Action
                });
            } else {
                    RCLCPP_WARN(m_logger_obj, "Attempted to record an action before the latest pose's time. Ignoring.");
                }
            } else {
                RCLCPP_WARN(m_logger_obj, "Attempted to record an action before the first pose. Ignoring.");
            }
        }

        void StateProjector::record_speed(float speed, rclcpp::Time time) {
            // std::cout << "Recording speed " << speed << " at time " << time.nanoseconds() << std::endl;

            if (m_pose_record.has_value() && time < m_pose_record.value().time) {
                if (time > m_init_speed.time) {
                    m_init_speed = Record {
                        .speed = speed,
                        .time = time,
                        .type = Record::Type::Speed
                    };
                }
            } else {
                m_history_since_pose.insert(Record {
                    .speed = speed,
                    .time = time,
                    .type = Record::Type::Speed
                });
            }

        }

        void StateProjector::record_pose(float x, float y, float yaw, rclcpp::Time time) {
            // std::cout << "Recording pose " << x << ", " << y << ", " << yaw << " at time " << time.nanoseconds() << std::endl;

            m_pose_record = Record {
                .pose = {
                    .x = x,
                    .y = y,
                    .yaw = yaw
                },
                .time = time,
                .type = Record::Type::Pose
            };

            auto record_iter = m_history_since_pose.begin();
            for (; record_iter != m_history_since_pose.end(); ++record_iter) {
                if (record_iter->time > time) {
                    break;
                }

                switch (record_iter->type) {
                    case Record::Type::Action:
                        m_init_action = *record_iter;
                        break;

                    case Record::Type::Speed:
                        m_init_speed = *record_iter;
                        break;

                    default:
                        break;
                }
            }

            m_history_since_pose.erase(m_history_since_pose.begin(), record_iter);

        }

        void StateProjector::record_position_lla(float x, float y, rclcpp::Time time) {
            m_history_since_pose.insert(Record {
                .position_lla = {
                    .x = x,
                    .y = y
                },
                .time = time,
                .type = Record::Type::PositionLLA
            });
        }

        void StateProjector::record_yaw(float yaw, rclcpp::Time time) {
            m_history_since_pose.insert(Record {
                .yaw = yaw,
                .time = time,
                .type = Record::Type::Yaw
            });
        }

        void StateProjector::record_swangle(float swangle, rclcpp::Time time) {
            if (m_pose_record.has_value() && time < m_pose_record.value().time) {
                if (time > m_init_swangle.time) {
                    m_init_swangle = Record {
                        .swangle = swangle,
                        .time = time,
                        .type = Record::Type::Swangle
                    };
                }
            }
            else {
                m_history_since_pose.insert(Record {
                    .swangle = swangle,
                    .time = time,
                    .type = Record::Type::Swangle
                });
            }

        }

        static void record_state(size_t time_ns, const State& state, std::stringstream& output_stream) {
            if (log_state_projection_history) {
                output_stream << "[0]Time: " << time_ns << "ns|||";
                output_stream << "Predicted X: " << state[state_x_idx] << "|||";
                output_stream << std::endl;

                output_stream << "[0]Time: " << time_ns << "ns|||";
                output_stream << "Predicted Y: " << state[state_y_idx] << "|||";
                output_stream << std::endl;

                output_stream << "[0]Time: " << time_ns << "ns|||";
                output_stream << "Predicted Yaw: " << state[state_yaw_idx] << "|||";
                output_stream << std::endl;

                output_stream << "[0]Time: " << time_ns << "ns|||";
                output_stream << "Predicted Speed: " << state[state_speed_idx] << "|||";
                output_stream << std::endl;
            }
        }

        std::optional<State> StateProjector::project(const rclcpp::Time& time, LoggerFunc logger_func) const {
            paranoid_assert(m_pose_record.has_value() && "State projector has not recieved first pose");

            std::stringstream predicted_ss;
            State state;
            state[state_x_idx] = m_pose_record.value().pose.x;
            state[state_y_idx] = m_pose_record.value().pose.y;
            state[state_yaw_idx] = m_pose_record.value().pose.yaw;
            state[state_speed_idx] = m_init_speed.speed;
            state[state_actual_swangle_idx] = m_init_swangle.swangle;
            record_state(m_pose_record.value().time.nanoseconds(), state, predicted_ss);

            const auto first_time = m_history_since_pose.empty() ? time : m_history_since_pose.begin()->time;
            const float delta_time = (first_time.nanoseconds() - m_pose_record.value().time.nanoseconds()) / 1e9f;
            // std::cout << "delta time: " << delta_time << std::endl;
            if (delta_time <= 0) {
                RCLCPP_WARN(m_logger_obj, "RUH ROH. Delta time for propogation delay simulation was negative.   : (");
                return std::nullopt;
            }
            // simulates up to first_time
            ONLINE_DYNAMICS_FUNC(state.data(), m_init_action.action.data(), state.data(), delta_time);
            record_state(first_time.nanoseconds(), state, predicted_ss);

            rclcpp::Time sim_time = first_time;
            Action last_action = m_init_action.action;
            for (auto record_iter = m_history_since_pose.begin(); record_iter != m_history_since_pose.end(); ++record_iter) {
                // checks if we're on last record
                const auto next_time = std::next(record_iter) == m_history_since_pose.end() ? time : std::next(record_iter)->time;

                const float delta_time = (next_time - sim_time).nanoseconds() / 1e9f;
                if (delta_time < 0) {
                    RCLCPP_WARN(m_logger_obj, "RUH ROH. Delta time for propogation delay simulation within the while loop  was negative.   : (");
                    return std::nullopt;
                }

                switch (record_iter->type) {
                    case Record::Type::Action:
                        ONLINE_DYNAMICS_FUNC(state.data(), record_iter->action.data(), state.data(), delta_time);
                        last_action = record_iter->action;
                        break;

                    case Record::Type::Speed:
                        char logger_buf[70];
                        snprintf(logger_buf, 70, "Predicted speed: %f\nActual speed: %f", state[state_speed_idx], record_iter->speed);
                        //std::cout << logger_buf << std::endl;
                        state[state_speed_idx] = record_iter->speed;
                        ONLINE_DYNAMICS_FUNC(state.data(), last_action.data(), state.data(), delta_time);
                        break;

// #ifdef STEERING_MODEL
//                     case Record::Type::Swangle:
//                         state[state_actual_swangle_idx] = record_iter->swangle;
//                         ONLINE_DYNAMICS_FUNC(state.data(), last_action.data(), state.data(), delta_time);
// #endif
                    default:
                        break;
                }

                record_state(next_time.nanoseconds(), state, predicted_ss);

                sim_time = next_time;
            }
            if (log_state_projection_history) {
                std::string log_location = getenv("ROS_LOG_DIR") + std::string{"/state_projection_history.txt"};
                std::cout << "Logging to: " << log_location << std::endl;
                std::fstream fs {log_location, std::ios::out | std::ios::app};
                fs << history_to_string();
                fs << "---- BEGIN PREDICTION ----" << std::endl;
                fs << predicted_ss.str();
                fs << "---- END PREDICTION ----" << std::endl;
            }

            return std::optional<State> {state};
        }

        bool StateProjector::is_ready() const {
            return m_pose_record.has_value();
        }
    }
}