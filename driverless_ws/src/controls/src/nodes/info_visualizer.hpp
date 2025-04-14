#pragma once
#include <ros_types_and_constants.hpp>


namespace controls {
    class InfoVisualizer : public rclcpp::Node {
        public:
            InfoVisualizer();
        private:
            rclcpp::Subscription<InfoMsg>::SharedPtr m_info_subscription; ///< Subscribes to intertial twist
            void output_info(InfoMsg info);

    };
}