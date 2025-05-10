#pragma once

#include <rclcpp/rclcpp.hpp>
#include <types.hpp>
#include <utils/general_utils.hpp>
#include <cmath>
#include <queue>
namespace controls {

    inline float twist_msg_to_speed(const TwistMsg& twist_msg) {
        return std::sqrt(
        twist_msg.twist.linear.x * twist_msg.twist.linear.x
        + twist_msg.twist.linear.y * twist_msg.twist.linear.y);
    }

    inline float quat_msg_to_yaw(const QuatMsg &quat_msg) {
        double qw = quat_msg.quaternion.w;
        double qx = quat_msg.quaternion.x;
        double qy = quat_msg.quaternion.y;
        double qz = quat_msg.quaternion.z;
        return atan2(2 * (qz * qw + qx * qy), -1 + 2 * (qw * qw + qx * qx));
    }

    inline XYPosition position_lla_msg_to_xy(const PositionLLAMsg& position_lla_msg) {
        /* Doesn't depend on imu axes. These are global coordinates in ENU frame */
        double latitude = position_lla_msg.vector.x;
        double longitude = position_lla_msg.vector.y;

        double LAT_DEG_TO_METERS = 111111;

        /* Intuition: Find the radius of the circle at current latitude
        * Convert the change in longitude to radians
        * Calculate the distance in meters
        *
        * The range should be the earth's radius: 6378137 meters.
        * The radius of the circle at current latitude: 6378137 * cos(latitude_rads)
        * To get the longitude, we need to convert change in longitude to radians
        * - longitude_rads = longitude * (M_PI / 180.0)
        *
        * Observe: 111320 = 6378137 * M_PI / 180.0
        */

        /* Represents the radius used to multiply angle in radians */
        double LON_DEG_TO_METERS = 111319.5 * cos(degrees_to_radians(latitude));

        return {LON_DEG_TO_METERS * longitude, LAT_DEG_TO_METERS * latitude};
    }


    template <typename T> // ConeMsg, float (swangle), etc.
    class PropagationSimulator {
        public:
            PropagationSimulator(float delay_ms) : m_delay_ms{delay_ms} {}
            void push(T element, rclcpp::Time time) {
                m_queue.emplace(element, time);
            }

            std::optional<T> maybe_pop(rclcpp::Time current_time) {
                if (m_queue.size() > 0) {
                    if (current_time.nanoseconds() - m_queue.front().second.nanoseconds() > m_delay_ms * 1e6) {
                        auto ret = m_queue.front();
                        m_queue.pop();
                        return ret.first;
                    }
                }
                return std::nullopt;
            }

        private:
            float m_delay_ms;
            using QueueElement = std::pair<T, rclcpp::Time>;
            std::queue<QueueElement> m_queue;
    };

}