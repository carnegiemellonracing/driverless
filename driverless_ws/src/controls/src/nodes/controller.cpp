#include <interface/types.hpp>
#include <interface/environment.hpp>
#include <mppi/mppi.hpp>
#include <mutex>
#include <condition_variable>
#include <rclcpp/rclcpp.hpp>
#include <atomic>

namespace controls {
    class controller_node : public rclcpp::Node {
    public:
        constexpr static const char *node_name = "controller";

        controller_node ()
            : Node {node_name} { }

    private:

        [[noreturn]]
        void mppi_loop() {
            while (true) {

            }
        }

        void on_spline(const interface::spline_msg &msg);
        void on_slam(const interface::slam_msg &msg);
        void on_gps(const interface::gps_msg &msg);

        interface::environment m_environment;
        std::mutex m_environment_mutex;
        std::condition_variable m_environment_notifier;
        std::atomic<bool> m_environment_dirty {false};

        mppi::mppi_controller m_mppi_controller;
    };
}


int main(int argc, char *argv[]) {
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<controls::controller_node>());
    rclcpp::shutdown();
    return 0;
}