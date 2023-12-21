#include <interface/interface.hpp>
#include <mppi/mppi.hpp>
#include <mutex>
#include <rclcpp/rclcpp.hpp>
#include <types.hpp>
#include <constants.hpp>
#include <condition_variable>
#include <interfaces/msg/control_action.hpp>


namespace controls {
    namespace nodes {
        class ControllerNode : public rclcpp::Node {
        public:
            ControllerNode (std::unique_ptr<interface::Environment> environment, std::unique_ptr<Controller> controller)
                : Node {controller_node_name},
                  m_environment(std::move(environment)),
                  m_controller(std::move(controller)),
                  m_controller_timer(
                      create_wall_timer(
                          controller_period,
                          [this] { controller_callback(); })),
                  m_action_publisher(
                      create_publisher<interfaces::msg::ControlAction>(
                          control_action_topic_name,
                          control_action_qos)) { }

        private:
            void publish_action(const Action &action) const;

            void controller_callback() {
                Action action;
                {
                    std::lock_guard<std::mutex> lock {m_environment->mutex};

                    const State curv_state = m_environment->get_curv_state();
                    action = m_controller->generate_action(curv_state);
                }

                publish_action(action);
            }

            std::unique_ptr<interface::Environment> m_environment;
            std::unique_ptr<Controller> m_controller;
            rclcpp::TimerBase::SharedPtr m_controller_timer;
            rclcpp::Publisher<interfaces::msg::ControlAction>::SharedPtr m_action_publisher;
        };
    }
}


int main(int argc, char *argv[]) {
    using namespace controls;

    std::unique_ptr<interface::Environment> environment;
    std::unique_ptr<Controller> controller;
    if (default_device == Device::Cpu) {
        environment = std::make_unique<interface::CpuEnvironment>();
        controller = std::make_unique<mppi::CpuMppiController>();
    } else {
#ifdef CONTROLS_NO_CUDA
        throw std::runtime_error("Cuda enabled without cuda compilation");
#else
        // TODO: implement cuda
#endif
    }

    rclcpp::init(argc, argv);

    const auto node = std::make_shared<nodes::ControllerNode>(environment, controller);
    rclcpp::spin(node);

    rclcpp::shutdown();
    return 0;
}