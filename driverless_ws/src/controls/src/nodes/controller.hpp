/**
 * The main controller process.
 *
 * --- OVERVIEW ---
 * Running `main` starts up to two tasks:
 *    1. Spinning the controller node
 *    2. Starting the OpenGL display, if DISPLAY is defined
 *
 * Each of these is started asynchronously, and the process terminates after either tasks exits.
 */

#pragma once

#include <state/state_estimator.hpp>
#include <condition_variable>


namespace controls {
    namespace nodes {

        /**
         * Controller node!
         *
         *  - Initialization: Construction creates subscribers, publishers, and timers, and launches MPPI in a new thread.
         *
         *  - Subscribers:
         *    - Spline: path planning spline. On receive, the state estimator is updated, and the curvilinear state
         *      recalculated. The MPPI thread is notified that the state is dirty and to refire if idle.
         *    - State: ditto.
         *
         *    These callbacks are mutually exclusive.
         *
         *  - Publishers:
         *    - Control Action: every `controller_period`, the most recently calculated action is published.
         *      The action is double buffered, to minimize the delay that MPPI will have on action publishing. The
         *      timer callback is parallel with any other callbacks, the while the consistency of the publishing isn't
         *      guaranteed, it won't be delayed by MPPI or state updates.
         */
        class ControllerNode : public rclcpp::Node {
        public:

            /**
             * Construct the controller node. Launches MPPI in a new (detached) thread.
             *
             * @param state_estimator Shared pointer to a state estimator
             * @param mppi_controller Shared pointer to an mppi controller
             */
            ControllerNode (
                std::shared_ptr<state::StateEstimator> state_estimator,
                std::shared_ptr<mppi::MppiController> mppi_controller
            );

            struct ActionSignal
            {
                int16_t front_torque_mNm = 0;
                int16_t back_torque_mNm = 0;
                uint16_t velocity_rpm = 0;
                uint16_t rack_displacement_adc = 0;
            };

        private:

            /**
             * Callback for spline subscription. Forwards message to `StateEstimator::on_spline`, and notifies MPPI
             * thread of the dirty state.
             *
             * @param spline_msg Received spline message
             */
            void spline_callback(const SplineMsg& spline_msg);

            /**
             * Callback for world twist subscription. Forwards message to `StateEstimator::on_world_twist`, and notifies MPPI
             * thread of the dirty state. Likely from GPS.
             *
             * @param twist_msg Received twist message
             */
            void world_twist_callback(const TwistMsg& twist_msg);

            /**
             * Callback for world pose subscription. Forwards message to `StateEstimator::on_world_pose`, and notifies MPPI
             * thread of the dirty state. Likely from GPS.
             *
             * @param pose_msg Received pose message
             */
            void world_pose_callback(const PoseMsg& pose_msg);

            /**
             * Publish an action
             *
             * @param action Action to publish
             */
            void publish_action(const Action& action);

            void rosbag_action_callback(const ActionMsg& action_msg);

            /**
             * Launch MPPI thread, which loops the following routine persitently:
             *  - Wait to be notified that the state is dirty
             *  - Run MPPI
             *  - Swap action buffers
             *
             * @return the launched thread
             */
            std::thread launch_mppi();
            ActionSignal action_to_signal(Action action);
            ActionSignal m_last_action_signal;
            std::thread launch_can();

            /** Notify MPPI thread that the state is dirty, and to refire if idle */
            void notify_state_dirty();

            ActionMsg action_to_msg(const Action& action);

            static StateMsg state_to_msg(const State& state);
            static void publish_and_print_info(std::ostream& stream, InfoMsg info);


            /** State estimator instance */
            std::shared_ptr<state::StateEstimator> m_state_estimator;

            /** Controller instance */
            std::shared_ptr<mppi::MppiController> m_mppi_controller;

            rclcpp::Publisher<ActionMsg>::SharedPtr m_action_publisher;
            rclcpp::Publisher<InfoMsg>::SharedPtr m_info_publisher;
            rclcpp::Publisher<SplineMsg>::SharedPtr m_spline_republisher;
            rclcpp::Subscription<SplineMsg>::SharedPtr m_spline_subscription;
            rclcpp::Subscription<TwistMsg>::SharedPtr m_world_twist_subscription;
            rclcpp::Subscription<QuatMsg>::SharedPtr m_world_quat_subscription;
            rclcpp::Subscription<PoseMsg>::SharedPtr m_world_pose_subscription;
            rclcpp::Subscription<ActionMsg>::SharedPtr m_rosbag_action_subscriber;

            /**
             * Mutex protecting `m_state_estimator`. This needs to be acquired when forwarding callbacks or waiting
             * on `m_state_cond_var`
             */
            std::mutex m_state_mut;

            /**
             * Condition variable for notifying state dirty-ing. MPPI waits on this variable while state and spline
             * callbacks notify it. `m_state_mut` must be acquired before waiting on this.
             */
            std::condition_variable m_state_cond_var;
            SplineMsg m_last_spline_msg;
        };
    }
}
