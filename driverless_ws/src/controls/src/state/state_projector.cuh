#pragma once

#include <optional>
#include <types.hpp>
#include <rclcpp/rclcpp.hpp>
#include <utils/general_utils.hpp>

namespace controls {
    namespace state {
        class StateProjector {
        public:
            StateProjector();
            /**
             * @brief Record an action into the history
             * @param[in] action The action to be recorded
             * @param[in] time The time at which the action was taken by the actuators.
             */
            void record_action(Action action, rclcpp::Time time);

            /**
             * @brief Record a speed into the history
             * @param[in] speed The speed to be recorded
             * @param[in] time The time at which the speed was measured. Should have no latency.
             */
            void record_speed(float speed, rclcpp::Time time);

            /**
             * @brief Record a pose into the history. Although pose data itself is not measured, this is called when
             * the spline is updated, and a pose of (0,0,0) is inferred from the spline.
             * @param x The x coordinate of the pose
             * @param y The y coordinate of the pose
             * @param yaw The yaw of the vehicle w.r.t. the inertial coordinate frame
             * @param time The time at which the vehicle had the pose. Since the pose is inferred from the spline, this
             * should be when the LIDAR points first came in.
             */
            void record_pose(float x, float y, float yaw, rclcpp::Time time);

            void record_position_lla(float x, float y, rclcpp::Time time);

            void record_yaw(float yaw, rclcpp::Time time);

            /**
             * @brief "main" projection function. Projects the state of the car at a given time, from the most
             * recent pose record and the history of actions and speeds since that pose record.
             * @param time The time at which the state is to be projected
             * @param logger The logger function to be used
             * @return The projected state of the car at the given time
             */
            std::optional<State> project(const rclcpp::Time& time, LoggerFunc logger) const;
            /**
             * @brief Whether the StateProjector is ready to project a state. This is true if there is a pose record,
             * which is every time since the first spline is received.
             * @return True if the StateProjector is ready to project a state, false otherwise.
             */
            bool is_ready() const;

            void output_history_to_file(const char* filename) const;

        private:
            /// Historical record type
            struct Record {
                enum class Type {
                    Action,
                    Speed,
                    Pose,
                    PositionLLA,
                    Yaw
                };

                union {
                    Action action;
                    float speed;

                    struct {
                        float x;
                        float y;
                        float yaw;
                    } pose;

                    struct {
                        float x;
                        float y;
                    } position_lla;

                    float yaw;
                };

                rclcpp::Time time;
                Type type;
            };

            /// Prints the elements of m_history_since_pose, for debugging purposes.
            void print_history() const;

            /// @note m_init_action and m_init_speed should occur <= m_pose_record, if m_pose_record exists

            /// "Default" action to be used until a recorded action is available
            Record m_init_action { .action {}, .time = rclcpp::Time(0UL, default_clock_type), .type = Record::Type::Action};
            /// "Default" speed to be used until a recorded speed is available
            /// @note direction of velocity is inferred from swangle
            Record m_init_speed { .speed = 0, .time = rclcpp::Time(0UL, default_clock_type), .type = Record::Type::Speed};
            /// most recent and only pose (new pose implies a new coord. frame, throw away data in old coord. frame)
            /// only nullopt before first pose received
            std::optional<Record> m_pose_record = std::nullopt;

            /// Helper binary operator for sorting records by time, needed for the multiset.
            struct CompareRecordTimes {
                bool operator() (const Record& a, const Record& b) const {
                    return a.time < b.time;
                }
            };
            /**
             * Contains all action and speed records since the last pose record.
             * Multiset is used like a self-sorting array.
             *
             * Invariants of m_history_since_pose:
             * should only contain Action and Speed records
             * time stamps of each record should be strictly after m_pose_record
             */
            std::multiset<Record, CompareRecordTimes> m_history_since_pose {};
            rclcpp::Logger m_logger_obj;

        };
    }
}