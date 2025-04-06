#include <utils/ros_utils.hpp>

namespace controls {
    namespace state {

        void NaiveStateTracker::record_positionlla(const PositionLLAMsg& position_lla_msg) {
            std::lock_guard<std::mutex> guard {m_xy_positions_mutex};
            m_xy_positions.emplace(position_lla_msg.header.stamp.seconds(), position_lla_msg_to_xy(position_lla_msg));
        }

        void NaiveStateTracker::record_quaternion(const QuatMsg& quat_msg) {
            std::lock_guard<std::mutex> guard {m_yaws_mutex};
            m_yaws.insert({quat_msg.header.stamp.seconds(), quat_msg_to_yaw(quat_msg.yaw)});
        }


        std::optional<PositionAndYaw> get_relative_position_and_yaw(rclcpp::Time cone_seen_time) {
            std::lock_guard<std::mutex> guard {m_xy_positions_mutex};
            std::lock_guard<std::mutex> guard {m_yaws_mutex};


            double cone_seen_time_seconds = cone_seen_time.seconds();
            auto xy_start_iterator = m_xy_positions.lower_bound(cone_seen_time_seconds);
            auto yaw_start_iterator = m_yaws.lower_bound(cone_seen_time_seconds);
            if (xy_start_iterator == m_xy_positions.end() || yaw_start_iterator == m_yaws.end()) {
                return std::nullopt;
            }

            const XYPosition start_position = xy_start_iterator->second;
            const float start_yaw = yaw_start_iterator->second;

            const XYPosition end_position = (m_xy_positions.crbegin())->second;
            const float end_yaw = (m_yaws.crbegin())->second;

            const XYPosition position_diff {end_position.first - start_position.first, end_position.second - start_position.second};
            const float yaw_diff = end_yaw - start_yaw;
            return {std::make_pair(position_diff, yaw_diff)};
        }        

    }
}