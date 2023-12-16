#include <interface/interface.hpp>
#include <mppi/mppi.hpp>
#include <mutex>
#include <rclcpp/rclcpp.hpp>
#include <common/types.hpp>
#include <atomic>
#include <condition_variable>

namespace controls {

    /**
     * @note It is undefined for the destructor to be called while the node is spun up
     *       (most likely, mppi will exit uncleanly)
     */
    class controller_node : public rclcpp::Node {
    public:
        constexpr static const char *node_name = "controller";

        controller_node ()
            : Node {node_name} { }

        void start_mppi() {
            assert (!m_mppi_loop_thread.joinable());

            m_mppi_loop_should_exit = false;
            m_mppi_loop_thread = std::thread {mppi_loop};
        }

        void stop_mppi() {
            assert (m_mppi_loop_thread.joinable());

            m_mppi_loop_should_exit = true;
            m_mppi_loop_thread.join();
        }

    private:
        void publish_action(const action &action) const;

        void mppi_loop() {
            while (!m_mppi_loop_should_exit) {
                {
                    // wait for the environment to be updated, then copy it

                    std::unique_lock<std::mutex> lock {m_environment_updated_mutex};

                    while (!m_environment_dirty) {
                        m_environment_updated_notifier.wait(lock);
                    }

                    m_environment_frozen = std::make_unique<interface::environment>(*m_environment_updated);  // copy
                }

                const state current_state = m_environment_frozen->get_state();
                const action mppi_action = m_mppi_controller->generate_action(current_state);

                publish_action(mppi_action);
            }
        }

        void on_shutdown() {
            if (m_mppi_loop_thread.joinable()) {
                stop_mppi();
            }
        }

        std::unique_ptr<interface::environment> m_environment_frozen;

        std::unique_ptr<interface::environment> m_environment_updated;
        std::mutex m_environment_updated_mutex;
        std::condition_variable m_environment_updated_notifier;
        std::atomic<bool> m_environment_dirty {false};

        mppi::mppi_controller m_mppi_controller;
        std::thread m_mppi_loop_thread;
        std::atomic<bool> m_mppi_loop_should_exit {true};
    };
}


int main(int argc, char *argv[]) {
    rclcpp::init(argc, argv);

    const auto node = std::make_shared<controls::controller_node>();
    rclcpp::spin(node);
    node->start_mppi();

    rclcpp::shutdown();
    return 0;
}