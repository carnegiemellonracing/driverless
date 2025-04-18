#include <mppi/mppi.hpp>
#include <mutex>
#include <types.hpp>
#include <constants.hpp>
#include <interfaces/msg/control_action.hpp>
#include <state/state_estimator.hpp>

#ifdef DISPLAY
#include <display/display.hpp>
#endif

#include "offline_controller.hpp"
#include <fstream>

namespace controls
{
    OfflineController::OfflineController(
        std::shared_ptr<state::StateEstimator> state_estimator,
        std::shared_ptr<mppi::MppiController> mppi_controller,
        const std::string& input_file)
        : Node("offline_controller"),
          m_state_estimator{std::move(state_estimator)},
          m_mppi_controller{std::move(mppi_controller)}
    {
        std::cout << "Processing input file: " << input_file << " with " << num_samples << " samples" << std::endl;
        process_file(input_file, std::to_string(num_samples) + "_offline_output_" + input_file);
        std::cout << "Job done!\n";
    }

    static std::vector<std::string> split(const std::string& str, char delimiter) {
        std::vector<std::string> tokens;
        std::istringstream iss(str);
        std::string token;

        while (std::getline(iss, token, delimiter)) {
            tokens.push_back(token);
        }

        return tokens;
    }

    static std::vector<geometry_msgs::msg::Point,
                       std::allocator<geometry_msgs::msg::Point>> string_to_point_array(
        const std::vector<std::string>& fields)
    {
        std::vector<geometry_msgs::msg::Point,
                    std::allocator<geometry_msgs::msg::Point>> points;
        for (const auto& field : fields) {
            geometry_msgs::msg::Point point;
            float x;
            float y;
            std::istringstream(field) >> x >> y;
            point.x = x;
            point.y = y;
            points.push_back(point);
        }
        return points;
    }

        void OfflineController::process_file(const std::string &input_file, const std::string &output_file)
    {
        // Process the input file line by line
        std::ifstream file {input_file};
        std::ofstream output {output_file};

        std::string line;

        while (std::getline(file, line)) {
            std::vector<std::string> parameters = split(line, '|');
            std::vector<std::string> state_fields = split(parameters[0], ',');
            if (state_fields.size() != state_dims) {
                throw std::runtime_error("Invalid state dimensions in offline input");
            }
            State state;
            for (size_t i = 0; i < state_dims; i++) {
                state[i] = std::stof(state_fields[i]);
            }
            SplineMsg spline_msg;
            spline_msg.frames = string_to_point_array(split(parameters[1], ','));
            ConeMsg cone_msg;
            cone_msg.blue_cones = string_to_point_array(split(parameters[2], ','));
            cone_msg.yellow_cones = string_to_point_array(split(parameters[3], ','));

            std::vector<Action> last_action_trajectory;
            std::vector<std::string> last_action_fields = split(parameters[4], ',');
            for (size_t i = 0; i < last_action_fields.size(); i++) {
                Action action;
                std::istringstream(last_action_fields[i]) >> action[0] >> action[1];
                last_action_trajectory.push_back(action);
            }

            // parse line into spline, state and best guess messages
            // construct spline and twist "messages" then pass them to state estimator

            m_state_estimator->on_spline(spline_msg);
            m_state_estimator->on_cone(cone_msg);

            m_state_estimator->render_and_sync(state);

            // send state to device (i.e. cuda globals)
            // (also serves to lock state since nothing else updates gpu state)
            m_mppi_controller->hardcode_last_action_trajectory(last_action_trajectory);

            // run mppi, and write action to the write buffer
            m_mppi_controller->generate_action();

            // parameters_ss << "Swangle range: " << 19 * M_PI / 180 * 2 << "\nThrottle range: " << saturating_motor_torque * 2 << "\n";

            std::vector<Action> averaged_trajectory = m_mppi_controller->m_averaged_trajectory;
            for (Action action : averaged_trajectory) {
                output << action[0] << ";" << action[1] << ",";
            }
            output << "\n";
        }
    }
}

int main(int argc, char *argv[])
{
    using namespace controls;
    if (argc != 2) {
        std::cout << "Usage: offline_controller <input_file>" << std::endl;
        return 1;
    }


    std::mutex mppi_mutex;
    std::mutex state_mutex;

    // create resources
    std::shared_ptr<state::StateEstimator> state_estimator = state::StateEstimator::create(state_mutex);
    std::shared_ptr<mppi::MppiController> controller = mppi::MppiController::create(mppi_mutex);

    // create a condition variable to notify main thread when either display or node dies
    rclcpp::init(argc, argv);
    auto node = std::make_shared<OfflineController>(state_estimator, controller, argv[1]);
    rclcpp::spin(node);
    rclcpp::shutdown();
    return 0;
}