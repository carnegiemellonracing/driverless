#pragma once

#include <types.hpp>
#include <optional>

namespace controls {
    namespace state {


        class NaiveStateTracker {
            public:
                void record_positionlla(const PositionLLAMsg& position_lla_msg);
                void record_quaternion(const QuatMsg& quat_msg);
                void record_cone_seen_time(const rclcpp::Time& cone_timestamp);
                // void clear_data_on_cone(const rclcpp::Time& cone_timestamp);
                std::optional<PositionAndYaw> get_relative_position_and_yaw();

            private:
                using EpochSeconds = double;
                std::multimap<EpochSeconds, XYPosition> m_xy_positions;
                std::mutex m_xy_positions_mutex;
                std::multimap<EpochSeconds, float> m_yaws;
                std::mutex m_yaws_mutex;
                rclcpp::Time m_last_cone_seen_time;
        };
    }
}