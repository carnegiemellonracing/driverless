#pragma once

#include <rclcpp/rclcpp.hpp>
#include <rclcpp/time.hpp>
#include <optional>

namespace controls {
    namespace test {
        class EchoNode : public rclcpp::Node {
        public:
            EchoNode();

        private:
            std::optional<double> m_orig_time;
            // double m_orig_time;
            double m_curr_time;
            rclcpp::Subscription<SplineMsg>::SharedPtr m_old_spline_subscriber;
            rclcpp::Publisher<SplineMsg>::SharedPtr m_spline_publisher;

            void echo(const SplineMsg& msg);

        };
    }
}


