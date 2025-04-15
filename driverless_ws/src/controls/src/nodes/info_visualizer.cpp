#include <nodes/info_visualizer.hpp>
#include <constants.hpp>

namespace controls {
    InfoVisualizer::InfoVisualizer() : rclcpp::Node("info_visualizer")
    {
        m_info_subscription = create_subscription<InfoMsg>(
                controller_info_topic_name, controller_info_qos, // was cone_qos but that didn't exist, publisher uses spline_qos
                [this](const InfoMsg::SharedPtr msg)
                { output_info(*msg); });
    }

    static std::string swangle_bar(float current, float min, float max, int width)
    {
        if (current < min || current > max)
        {
            return "OUT OF BOUNDS";
        }
        float progress = (current - min) / (max - min);
        int pos = static_cast<int>(progress * width);
        std::string bar;
        if (pos < width / 2)
        {
            bar = "[" + std::string(std::max(0, pos), ' ') + "\033[31m+\033[0m" + std::string(width / 2 - pos - 1, ' ') + "|" + std::string(width / 2, ' ') + "]";
        }
        else if (pos > width / 2)
        {
            bar = "[" + std::string(width / 2, ' ') + "|" + std::string(std::max(0, pos - width / 2), ' ') + "\033[32m+\033[0m" + std::string(width - pos - 1, ' ') + "]";
        }
        else
        {
            bar = "[" + std::string(width / 2, ' ') + "\033[33m+\033[0m" + std::string(width / 2, ' ') + "]";
        }

        return bar;
    }

            static std::string progress_bar(float current, float min, float max, int width) {
                if (current < min || current > max) {
                    return "OUT OF BOUNDS";
                }
                float progress = (current - min)/(max - min);
                int pos = static_cast<int>(progress * width);
                std::string bar;
                if (pos < width / 2)
                {
                    bar = "[" + std::string(std::max(0, pos), ' ') + "\033[31m" + std::string(width / 2 - pos, '|') + "\033[0m" + "|" + std::string(width / 2, ' ') + "]";
                }
                else if (pos > width / 2)
                {
                    bar = "[" + std::string(width / 2, ' ') + "|" + "\033[32m" + std::string(std::max(0, pos - width / 2), '|') + "\033[0m" + std::string(width - pos, ' ') + "]";
                }
                else
                {
                    bar = "[" + std::string(width / 2, ' ') + "|" + std::string(width / 2, ' ') + "]";
                }
                return bar;
            }

    void InfoVisualizer::output_info(InfoMsg info)
    {
        std::stringstream ss;

        ss
            << "Info Time: " << info.header.stamp.sec << "." << info.header.stamp.nanosec << "\n"
            << "Action:\n"
            << "  swangle (rad): " << info.action.swangle << "\n"
            << swangle_bar(info.action.swangle, min_swangle_rad, max_swangle_rad, 40) << "\n"
            << progress_bar(info.action.torque_fl, min_torque, max_torque, 40) << "\n"
            << progress_bar(info.action.torque_fr, min_torque, max_torque, 40) << "\n"
            << progress_bar(info.action.torque_rl, min_torque, max_torque, 40) << "\n"
            << progress_bar(info.action.torque_rr, min_torque, max_torque, 40) << "\n"
            << "  torque_fl (Nm): " << info.action.torque_fl << "\n"
            << "  torque_fr (Nm): " << info.action.torque_fr << "\n"
            << "  torque_rl (Nm): " << info.action.torque_rl << "\n"
            << "  torque_rr (Nm): " << info.action.torque_rr << "\n"
            << "Projected State:\n"
            << "  x (m): " << info.proj_state.x << "\n"
            << "  y (m): " << info.proj_state.y << "\n"
            << "  yaw (rad): " << info.proj_state.yaw << "\n"
            << "  speed (m/s): " << info.proj_state.speed << "\n"
            // << "Cone Processing Latency (ms)" << m_last_cone_process_time << "\n"
            << "State Projection Latency (ms): " << info.projection_latency_ms << "\n"
            << "OpenGL Render Latency (ms): " << info.render_latency_ms << "\n"
            << "MPPI Step Latency (ms): " << info.mppi_latency_ms << "\n"
            << "Controls Latency (ms): " << info.latency_ms << "\n"
            << "Total Latency (ms): " << info.total_latency_ms << "\n"
            << std::endl;

        std::string info_str = ss.str();

        std::cout << clear_term_sequence << info_str << std::flush;
        RCLCPP_INFO_STREAM(get_logger(), "received controller info. info:\n"
                                            << info_str);
    }



}

int main(int argc, char* argv[]) {
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<controls::InfoVisualizer>());
    rclcpp::shutdown();
    return 0;
}


