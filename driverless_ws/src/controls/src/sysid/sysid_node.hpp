#include <rclcpp/rclcpp.hpp>
#include <constants.hpp>
#include <types.hpp>
#include <interfaces/msg/control_action.hpp>

namespace controls {
    namespace sysid {
        class SysIdNode : public rclcpp::Node
        {
            public:
                SysIdNode(int selected_test);
                void slow_accel_deccel();
                void fast_accel_deccel();
                void full_clockwise_ramp();
                void full_anticlockwise_ramp();
                void half_clockwise_ramp();
                void half_anticlockwise_ramp();
            private:

                void get_next_message();
                void on_twist(const TwistMsg& msg);

                rclcpp::TimerBase::SharedPtr m_timer;
                rclcpp::Publisher<ActionMsg>::SharedPtr m_action_publisher; ///< Publishes control action for actuators
                rclcpp::Subscription<TwistMsg>::SharedPtr m_twist_subscription; ///< Subscribes to twist messages
                int m_selected_test;
                int m_counter;
                bool full_swing = false;
                ActionMsg m_next_action_msg;
                void publish_action();
        };
    }
}