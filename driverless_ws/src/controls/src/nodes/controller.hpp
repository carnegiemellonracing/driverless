#pragma once

#include <state/state_estimator.hpp>


namespace controls {
    namespace nodes {

        class ControllerNode : public rclcpp::Node {
        public:
            ControllerNode (
                std::unique_ptr<state::StateEstimator> state_estimator,
                std::unique_ptr<mppi::MppiController> mppi_controller
            );

        private:

            void publish_action_callback();
            void spline_callback(const SplineMsg& spline_msg);
            void slam_callback(const SlamMsg& slam_msg);

            /**
             * \brief Publish action to actuators
             * \param action action to publish
             */
            void publish_action(const Action& action) const;

            void run_mppi();
            void swap_action_buffers();

            std::unique_ptr<state::StateEstimator> m_state_estimator;
            std::unique_ptr<mppi::MppiController> m_mppi_controller;
            rclcpp::TimerBase::SharedPtr m_action_timer;
            rclcpp::Publisher<ActionMsg>::SharedPtr m_action_publisher;
            rclcpp::Subscription<SplineMsg>::SharedPtr m_spline_subscription;
            rclcpp::Subscription<SlamMsg>::SharedPtr m_slam_subscription;

            std::unique_ptr<Action> m_action_read;
            std::unique_ptr<Action> m_action_write;
            std::mutex action_read_mut;
            std::mutex action_write_mut;
        };

    }
}