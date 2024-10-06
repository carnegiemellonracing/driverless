#include <sysid/sysid_node.hpp>
#include <types.hpp>


namespace controls {
    namespace sysid {
        constexpr float max_wheel_request = saturating_motor_torque / 2;

        static ActionMsg slow_straight_msg {};
        static ActionMsg med_straight_msg {};
        static ActionMsg slow_clockwise_msg {};
        static ActionMsg fast_straight_msg {};
        static ActionMsg med_clockwise_msg {};
        static ActionMsg fast_clockwise_msg {};


        SysIdNode::SysIdNode(int selected_test) : Node("sysid_node"), m_selected_test{selected_test} {

            m_next_action_msg = slow_straight_msg;

            m_timer = create_wall_timer(
                std::chrono::duration<float, std::milli>(controller_period * 1000), 
                [this] {publish_action();});                    

            m_twist_subscription = create_subscription<TwistMsg>(
                world_twist_topic_name, world_twist_qos,
                [this] (const TwistMsg::SharedPtr msg) {
                    on_twist(*msg);
                }
            );

            m_action_publisher = create_publisher<ActionMsg>(
                control_action_topic_name, control_action_qos
            );

        }

        void SysIdNode::on_twist(const TwistMsg& msg) {
            if (!full_swing) {
                float speed = msg.twist.linear.x * msg.twist.linear.x
                + msg.twist.linear.y * msg.twist.linear.y;
                if (speed > 0.5f) {
                    full_swing = true;
                    get_next_message();
                }
            }
        }

        void SysIdNode::get_next_message() {

        }

        // Accelerate from 0 to max in 3 seconds, hold for 2 seconds, then decelerate from max to 0 in 3 seconds
        void SysIdNode::slow_accel_deccel() {
            // modifies m_next_action_msg
            if (m_counter < 30) {
                m_next_action_msg.torque_fl = max_wheel_request * (m_counter / 30.0f);
                m_next_action_msg.torque_fr = max_wheel_request * (m_counter / 30.0f);
            } else if (m_counter < 50) {
                m_next_action_msg.torque_fl = max_wheel_request;
                m_next_action_msg.torque_fr = max_wheel_request;
            } else if (m_counter < 80) {
                m_next_action_msg.torque_fl = max_wheel_request * ((50 - m_counter) / 30.0f);
                m_next_action_msg.torque_fr = max_wheel_request * ((50 - m_counter) / 30.0f);
            } else {
                m_next_action_msg.torque_fl = -max_wheel_request;
                m_next_action_msg.torque_fr = -max_wheel_request;
            }   
        }

        // Set to max for 4 seconds, then set to min
        void SysIdNode::fast_accel_deccel() {
            // modifies m_next_action_msg
            if (m_counter < 40) {
                m_next_action_msg.torque_fl = max_wheel_request;
                m_next_action_msg.torque_fr = max_wheel_request;
            } else{
                m_next_action_msg.torque_fl = -max_wheel_request;
                m_next_action_msg.torque_fr = -max_wheel_request;
            }  
        }

        // 4 seconds at slow, 4 seconds at med, 4 seconds at fast
        void SysIdNode::full_clockwise_ramp() {
            m_next_action_msg.swangle = saturating_swangle;
            if (m_counter < 40) {
                m_next_action_msg.torque_fl = max_wheel_request / 4;
                m_next_action_msg.torque_fr = max_wheel_request / 4;
            } else if (m_counter < 80) {
                m_next_action_msg.torque_fl = max_wheel_request / 2;
                m_next_action_msg.torque_fr = max_wheel_request / 2;
            } else {
                m_next_action_msg.torque_fl = max_wheel_request;
                m_next_action_msg.torque_fr = max_wheel_request;
            }
        }

        // 4 seconds at slow, 4 seconds at med, 4 seconds at fast
        void SysIdNode::full_anticlockwise_ramp() {
            m_next_action_msg.swangle = -saturating_swangle;
            if (m_counter < 40) {
                m_next_action_msg.torque_fl = max_wheel_request / 4;
                m_next_action_msg.torque_fr = max_wheel_request / 4;
            } else if (m_counter < 80) {
                m_next_action_msg.torque_fl = max_wheel_request / 2;
                m_next_action_msg.torque_fr = max_wheel_request / 2;
            } else {
                m_next_action_msg.torque_fl = max_wheel_request;
                m_next_action_msg.torque_fr = max_wheel_request;
            }
        }

        // 4 seconds at slow, 4 seconds at med, 4 seconds at fast
        void SysIdNode::half_clockwise_ramp() {
            m_next_action_msg.swangle = saturating_swangle;
            if (m_counter < 40) {
                m_next_action_msg.torque_fl = max_wheel_request / 4;
                m_next_action_msg.torque_fr = max_wheel_request / 4;
            } else if (m_counter < 80) {
                m_next_action_msg.torque_fl = max_wheel_request / 2;
                m_next_action_msg.torque_fr = max_wheel_request / 2;
            } else {
                m_next_action_msg.torque_fl = max_wheel_request;
                m_next_action_msg.torque_fr = max_wheel_request;
            } 
        }

        // 4 seconds at slow, 4 seconds at med, 4 seconds at fast
        void SysIdNode::half_anticlockwise_ramp() {
            m_next_action_msg.swangle = -saturating_swangle;
            if (m_counter < 40) {
                m_next_action_msg.torque_fl = max_wheel_request / 4;
                m_next_action_msg.torque_fr = max_wheel_request / 4;
            } else if (m_counter < 80) {
                m_next_action_msg.torque_fl = max_wheel_request / 2;
                m_next_action_msg.torque_fr = max_wheel_request / 2;
            } else {
                m_next_action_msg.torque_fl = max_wheel_request;
                m_next_action_msg.torque_fr = max_wheel_request;
            } 
        }

        using UpdateFunction = void(SysIdNode::*)();
        UpdateFunction actions[] {
            &SysIdNode::slow_accel_deccel,
            &SysIdNode::fast_accel_deccel,
            &SysIdNode::full_clockwise_ramp,
            &SysIdNode::full_anticlockwise_ramp,
            &SysIdNode::half_clockwise_ramp,
            &SysIdNode::half_anticlockwise_ramp
        };

        void SysIdNode::publish_action() {
            m_next_action_msg.orig_data_stamp = get_clock()->now();
            m_action_publisher->publish(m_next_action_msg);
            RCLCPP_INFO(get_logger(), "Publishing - Swangle: %f, Torques: %f, %f, %f, %f, Time: %u", 
                m_next_action_msg.swangle, m_next_action_msg.torque_fl, m_next_action_msg.torque_fr, 
                m_next_action_msg.torque_rl, m_next_action_msg.torque_rr, m_next_action_msg.orig_data_stamp.nanosec);
            std::invoke(actions[m_selected_test % 6], this);
            m_counter++;
        }

    }
}

int main(int argc, char* argv[]) {
    using namespace controls;
    using namespace controls::sysid;
    int test_number = 0;
    if (argc < 2) {
        std::cout << "Using test number 0 by default" << std::endl;
    } else {
        test_number = std::stoi(argv[1]);
    }

    slow_straight_msg.swangle = 0.0f;
    slow_straight_msg.torque_fl = max_wheel_request / 4;
    slow_straight_msg.torque_fr = max_wheel_request / 4;

    med_straight_msg.swangle = 0.0f;
    med_straight_msg.torque_fl = max_wheel_request / 2;
    med_straight_msg.torque_fr = max_wheel_request / 2;

    fast_straight_msg.swangle = 0.0f;
    fast_straight_msg.torque_fl = max_wheel_request;
    fast_straight_msg.torque_fr = max_wheel_request;

    fast_clockwise_msg.swangle = saturating_swangle;
    fast_clockwise_msg.torque_fl = max_wheel_request;
    fast_clockwise_msg.torque_fr = max_wheel_request;




    rclcpp::init(0, nullptr);
    rclcpp::spin(std::make_shared<SysIdNode>(test_number));
    
    display::Display display{};
    std::cout << "display created" << std::endl;

    std::thread display_thread {[&] {
        display.run();

        {
            std::lock_guard<std::mutex> guard {thread_died_mut};

            std::cout << "Display thread closed. Exiting.." << std::endl;
            thread_died = true;
            thread_died_cond.notify_all();
        }
    }};
    std::cout << "display thread launched" << std::endl;   
    rclcpp::shutdown();
}