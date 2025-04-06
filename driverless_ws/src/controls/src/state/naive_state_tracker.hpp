#include <types.hpp>
#include <optional>

namespace controls {
    namespace state {


        class NaiveStateTracker {
            public:
                using PositionAndYaw = std::pair<XYPosition, float>;
                void record_positionlla(const PositionLLAMsg& position_lla_msg);
                void record_quaternion(const QuatMsg& quat_msg);
                // void clear_data_on_cone(const rclcpp::Time& cone_timestamp);
                std::optional<PositionAndYaw> get_relative_position_and_yaw(rclcpp::Time cone_seen_time);

            private:
                using EpochSeconds = double;
                std::multimap<EpochSeconds, XYPosition> m_xy_positions;
                std::mutex m_xy_positions_mutex;
                std::multimap<EpochSeconds, float> m_yaws;
                std::mutex m_yaws_mutex;
        };
    }
}