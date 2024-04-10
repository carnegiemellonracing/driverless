#include <rclcpp/rclcpp.hpp>
#include <types.hpp>

namespace controls {
    namespace tests {

        class NullGpsSpoofer : public rclcpp::Node {
        private: 
            static constexpr uint32_t publish_millis = 100;

            void publish_quat () const {
                QuatMsg msg {};
                msg.quaternion.w = 1.0f;
                m_quat_publisher->publish(msg);
            }

            void publish_twist () const {
                TwistMsg msg {};
                msg.twist.linear.x = 1.7f;
                m_twist_publisher->publish(msg);
            }

            rclcpp::Publisher<QuatMsg>::SharedPtr m_quat_publisher;
            rclcpp::Publisher<TwistMsg>::SharedPtr m_twist_publisher;
            rclcpp::TimerBase::SharedPtr m_timer;

        public:
            NullGpsSpoofer () 
                : Node{"null_gps_spoofer"},
                  m_quat_publisher {create_publisher<QuatMsg>(world_quat_topic_name, world_quat_qos)},
                  m_twist_publisher {create_publisher<TwistMsg>(world_twist_topic_name, world_twist_qos)},
                  m_timer {create_wall_timer(
                  std::chrono::duration<float, std::milli>(100),
                    [this]{ publish_quat(); publish_twist(); })} { }

        };
    }
}

int main(int argc, char* argv[]){
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<controls::tests::NullGpsSpoofer>());
    rclcpp::shutdown();
    return 0;
}