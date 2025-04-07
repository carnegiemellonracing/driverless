#include <state/state_projector.cuh>
#include <utils/general_utils.hpp>
#include <model/slipless/model.cuh>


namespace controls {
    namespace state {

        StateProjector::StateProjector() : m_logger_obj {rclcpp::get_logger("")} {}
            
        void StateProjector::print_history() const {
            std::cout << "---- BEGIN HISTORY ---\n";
            for (const Record& record : m_history_since_pose) {
                switch (record.type) {
                    case Record::Type::Action:
                        std::cout << "Action: " << record.action[0] << ", " << record.action[1] << std::endl;
                        break;

                    case Record::Type::Speed:
                        std::cout << "Speed: " << record.speed << std::endl;
                        break;

                    case Record::Type::Pose:
                        std::cout << "Pose: " << record.pose.x << ", " << record.pose.y << ", " << record.pose.yaw << std::endl;
                        break;

                    default:
                        throw new std::runtime_error("bruh. invalid record type bruh. (in print history)");
                }
            }
            std::cout << "---END HISTORY---" << std::endl;
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

            // print_history();
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
                        throw new std::runtime_error("bruh. invalid record type bruh. (in record pose)");
                }
            }

            m_history_since_pose.erase(m_history_since_pose.begin(), record_iter);

            // print_history();
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


        std::optional<State> StateProjector::project(const rclcpp::Time& time, LoggerFunc logger_func) const {
            paranoid_assert(m_pose_record.has_value() && "State projector has not recieved first pose");
            // std::cout << "Projecting to " << time.nanoseconds() << std::endl;

            State state;
            state[state_x_idx] = m_pose_record.value().pose.x;
            state[state_y_idx] = m_pose_record.value().pose.y;
            state[state_yaw_idx] = m_pose_record.value().pose.yaw;
            state[state_speed_idx] = m_init_speed.speed;

            const auto first_time = m_history_since_pose.empty() ? time : m_history_since_pose.begin()->time;
            const float delta_time = (first_time.nanoseconds() - m_pose_record.value().time.nanoseconds()) / 1e9f;
            // std::cout << "delta time: " << delta_time << std::endl;
            if (delta_time <= 0) {
                RCLCPP_WARN(m_logger_obj, "RUH ROH. Delta time for propogation delay simulation was negative.   : (");
                return std::nullopt;
            }
            // simulates up to first_time
            model::slipless::dynamics(state.data(), m_init_action.action.data(), state.data(), delta_time);

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
                        model::slipless::dynamics(state.data(), record_iter->action.data(), state.data(), delta_time);
                        last_action = record_iter->action;
                        break;

                    case Record::Type::Speed:
                        char logger_buf[70];
                        snprintf(logger_buf, 70, "Predicted speed: %f\nActual speed: %f", state[state_speed_idx], record_iter->speed);
                        //std::cout << logger_buf << std::endl;
                        state[state_speed_idx] = record_iter->speed;
                        model::slipless::dynamics(state.data(), last_action.data(), state.data(), delta_time);
                        break;

                    default:
                        throw new std::runtime_error("bruh. invalid record type bruh. (in simulation)");
                }

                sim_time = next_time;
            }

            return std::optional<State> {state};
        }

        bool StateProjector::is_ready() const {
            return m_pose_record.has_value();
        }
    }
}