#include <utils/ros_utils.hpp>

namespace controls {
    namespace state {

        void NaiveStateTracker::record_positionlla(const PositionLLAMsg& position_lla_msg) {
            m_xy_positions.emplace(position_lla_msg.header.stamp.seconds(), position_lla_msg_to_xy(position_lla_msg));
        }

        void NaiveStateTracker::record_quaternion(const QuatMsg& quat_msg) {
            m_yaws.insert({quat_msg.header.stamp.seconds(), quat_msg_to_yaw(quat_msg.yaw)});
        }

        

    }
}