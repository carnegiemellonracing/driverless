#pragma once

#include <state/state_estimator.hpp>
#include <condition_variable>


namespace controls {
    namespace nodes {

        class ControllerNode : public rclcpp::Node {
        public:
            ControllerNode (
                std::shared_ptr<state::StateEstimator> state_estimator,
                std::shared_ptr<mppi::MppiController> mppi_controller
            );

        private:

            void publish_action_callback();
            void spline_callback(const SplineMsg& spline_msg);
            void state_callback(const StateMsg& state_msg);

            /**
             * \brief Publish action to actuators
             * \param action action to publish
             */
            void publish_action(const Action& action) const;

            std::thread launch_mppi();

            void notify_state_dirty();
            void swap_action_buffers();

            std::shared_ptr<state::StateEstimator> m_state_estimator;
            std::shared_ptr<mppi::MppiController> m_mppi_controller;
            rclcpp::TimerBase::SharedPtr m_action_timer;
            rclcpp::Publisher<ActionMsg>::SharedPtr m_action_publisher;
            rclcpp::Subscription<SplineMsg>::SharedPtr m_spline_subscription;
            rclcpp::Subscription<StateMsg>::SharedPtr m_state_subscription;

            std::unique_ptr<Action> m_action_read;
            std::unique_ptr<Action> m_action_write;
            std::mutex m_action_read_mut;
            std::mutex m_action_write_mut;

            std::mutex m_state_mut;
            std::condition_variable m_state_cond_var;

            std::atomic<bool> received_first_spline;
        };

    }
}
