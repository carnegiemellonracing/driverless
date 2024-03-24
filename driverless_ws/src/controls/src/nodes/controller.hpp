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


        private:

            /**
             * Callback for control action timer. Allowed to execute in parallel to other callbacks, and called every
             * `controller_period`
             */
            void publish_action_callback();

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
             * Callback for world quaternion subscription. Forwards message to `StateEstimator::on_world_quat`, and notifies MPPI
             * thread of the dirty state. Likely from GPS.
             *
             * @param quat_msg Received quaternion message
             */
            void world_quat_callback(const QuatMsg& quat_msg);

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
            void publish_action(const Action& action) const;

            /**
             * Launch MPPI thread, which loops the following routine persitently:
             *  - Wait to be notified that the state is dirty
             *  - Run MPPI
             *  - Swap action buffers
             *
             * @return the launched thread
             */
            std::thread launch_mppi();

            /** Notify MPPI thread that the state is dirty, and to refire if idle */
            void notify_state_dirty();

            /** Swap reading action buffer and writing action buffer */
            void swap_action_buffers();


            /** State estimator instance */
            std::shared_ptr<state::StateEstimator> m_state_estimator;

            /** Controller instance */
            std::shared_ptr<mppi::MppiController> m_mppi_controller;

            /** Timer to trigger action publishing */
            rclcpp::TimerBase::SharedPtr m_action_timer;

            rclcpp::Publisher<ActionMsg>::SharedPtr m_action_publisher;
            rclcpp::Subscription<SplineMsg>::SharedPtr m_spline_subscription;
            rclcpp::Subscription<TwistMsg>::SharedPtr m_world_twist_subscription;
            rclcpp::Subscription<QuatMsg>::SharedPtr m_world_quat_subscription;
            rclcpp::Subscription<PoseMsg>::SharedPtr m_world_pose_subscription;


            // Action double buffer

            /** Read side of the action double buffer. Action publisher reads the most recent action from this field. */
            std::unique_ptr<Action> m_action_read;

            /**
             * Write side of action double buffer. MPPI writes to this action at the end of a cycle, then swaps the
             * buffers.
             */
            std::unique_ptr<Action> m_action_write;

            /** Mutex protecting `m_action_read` */
            std::mutex m_action_read_mut;

            /** Mutex protecting `m_action_read` */
            std::mutex m_action_write_mut;


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
        };
    }
}
