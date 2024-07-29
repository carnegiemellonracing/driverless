#include <mppi/mppi.hpp>
#include <mutex>
#include <rclcpp/rclcpp.hpp>
#include <types.hpp>
#include <constants.hpp>
#include <interfaces/msg/control_action.hpp>
#include <state/state_estimator.hpp>

#ifdef DISPLAY
#include <display/display.hpp>
#endif

#include "controller.hpp"
#include "controller_component.cpp"

int main(int argc, char *argv[]) {
    using namespace controls;
    
    std::mutex mppi_mutex;
    std::mutex state_mutex;

    const rclcpp::NodeOptions options;


    // create resources
    std::shared_ptr<state::StateEstimator> state_estimator = state::StateEstimator::create(state_mutex);
    std::shared_ptr<mppi::MppiController> controller = mppi::MppiController::create(mppi_mutex);

    rclcpp::init(argc, argv);
    std::cout << "rclcpp initialized" << std::endl;
    
    const auto node2 = std::make_shared<controller::ControllerComponent>();
    std::cout << "dummy node created" << std::endl;

    // instantiate node
    const auto node = std::make_shared<controls::nodes::ControllerNode>(state_estimator, controller, options);
    std::cout << "controller node created" << std::endl;


    rclcpp::Logger logger = node->get_logger();
    LoggerFunc logger_func = [logger](const std::string& msg) {
        RCLCPP_DEBUG(logger, msg.c_str());
    };

    state_estimator->set_logger(logger_func);
    controller->set_logger(logger_func);

    // create a condition variable to notify main thread when either display or node dies
    std::mutex thread_died_mut;
    std::condition_variable thread_died_cond;
    bool thread_died = false;

    // auto node2 = std::make_shared<controller::ControllerComponent>();

    std::thread node_thread {[&] {
        rclcpp::executors::MultiThreadedExecutor exec;
        exec.add_node(node);
        exec.add_node(node2);
        exec.spin();

        {
            std::lock_guard<std::mutex> guard {thread_died_mut};

            std::cout << "Node terminated. Exiting..." << std::endl;
            thread_died = true;
            thread_died_cond.notify_all();
        }
    }};

    std::cout << "controller node thread launched" << std::endl;


#ifdef DISPLAY
    display::Display display {controller, state_estimator};
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
#endif

    // wait for a thread to die
    {
        std::unique_lock<std::mutex> guard {thread_died_mut};
        if (!thread_died) {
            thread_died_cond.wait(guard);
        }
    }

    std::cout << "Shutting down" << std::endl;
    rclcpp::shutdown();
    return 0;
}