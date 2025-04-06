#include <types.hpp>

namespace controls {
    namespace state {


        class NaiveStateTracker {
            public:
                void record_positionlla(const PositionLLAMsg& position_lla_msg);
                void record_quaternion(const QuatMsg& quat_msg);
                get_relative_position_and_yaw(rclcpp::Time )


            private:
                using EpochSeconds = double;
                std::multimap<EpochSeconds, XYPosition> m_xy_positions;
                std::multimap<EpochSeconds, float> m_yaws;
        };
    }
}