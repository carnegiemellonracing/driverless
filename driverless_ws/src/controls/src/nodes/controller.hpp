/**
 * @file controller.hpp The main controller process.
 *
 * --- OVERVIEW ---
 * Running `main` starts up to two tasks:
 *    1. Spinning the controller node
 *    2. Starting the OpenGL display, if DISPLAY is defined
 *
 * Each of these is started asynchronously, and the process terminates after either tasks exits.
 */

#pragma once

//TODO: MPPI is supposedly included by state_estimator but I can't find it. Weird
#include <state/state_estimator.hpp>
#include <condition_variable>


namespace controls {
    namespace nodes {

        /**
         * Controller node!
         *
         * If you're unfamiliar with ROS2, check out https://docs.ros.org/en/humble/Tutorials.html.
         *
         * The controller hinges on the operation of two objects that it owns - the @ref state::StateEstimator "StateEstimator"
         * and the @ref mppi::MppiController "MppiController".
         * Both of these rely on the @rst `dynamics model <../../../_static/model.pdf>`_. @endrst
         *
         *  - StateEstimator: given twist and spline, esimates inertial state and a inertial to curvilinear lookup table.
         *  - MPPIController: given inertial state and the lookup table, calculates the optimal control action to take
         * using the @rst `MPPI Algorithm <../../../_static/mppi.pdf>`_. @endrst
         *
         * This is how the node works:
         *
         *  - Initialization: Construction creates subscribers, publishers, and timers, then launches MPPI in a new thread.
         *
         *  - Subscribers:
         *    - Spline: path planning spline. On receive, the state estimator is updated, and the curvilinear state
         *      recalculated. The MPPI thread is notified that the state is dirty and to refire if idle.
         *    - State: ditto. just twist
         *    @TODO: double confirm this
         *    linear velocity from twist, yaw rate from steering wheel angle, better than time-syncing
         *    baked into model
         *
         *    These callbacks are mutually exclusive.
         *
         *  - Publishers:
         *    - Control Action: every @ref controller_period, the most recently calculated action is published to the
         *    `control_action` topic, which is received by the Actuators node.
         *
         *      The action is double buffered, to minimize the delay that MPPI will have on action publishing. The
         *      timer callback is parallel with any other callbacks, so while the consistency of the publishing isn't
         *      guaranteed, it won't be delayed by MPPI or state updates.
         *      todo add link to double buffering, inquire about consistency of publishing
         *    - Controller Info: every controller step, the most recent info message is published to the `controller_info`
         *    topic for debugging.
         *
         * Code for the node can be found in *controls/src/nodes/controller.cpp*.
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
             * @TODO: pretty sure this is unused
             */
            void world_pose_callback(const PoseMsg& pose_msg);

            /**
             * Publishes a control action to the `control_action` topic.
             *
             * @param action Action to publish
             */
            void publish_action(const Action& action);

            /**
             * Launch MPPI thread, which loops the following routine persistently:
             *  - Wait to be notified that the state is dirty.
             *  - Run MPPI and write an action to the write buffer.
             *  - Swap the read and write buffers.
             *
             * @return the launched thread
             */
            std::thread launch_mppi();

            /** Notify MPPI thread that the state is dirty, and to refire if idle TODO idle as in waiting to be notified? */
            /// @todo: time is changing so everything is always changing. Consider max control frequency
            void notify_state_dirty();

            /// Converts MPPI control action output to a ROS2 message. Affected by drive mode (FWD, RWD, AWD).
            /// @param[in] action Control action - output of MPPI.
            ActionMsg action_to_msg(const Action& action);

            /// Converts state to a ROS2 message.
            /// @param[in] state Vehicle state.
            static StateMsg state_to_msg(const State& state);

            ///TODO: doesn't this only print not publish
            /** Prints the information in a InfoMsg to the given stream (usually the console).
             * @param[in] stream Stream to print to
             * @param[in] info Info message to print
             */
            static void publish_and_print_info(std::ostream& stream, InfoMsg info);


            /** State estimator instance */
            std::shared_ptr<state::StateEstimator> m_state_estimator;

            /** MPPI Controller instance */
            std::shared_ptr<mppi::MppiController> m_mppi_controller;

            rclcpp::Publisher<ActionMsg>::SharedPtr m_action_publisher; ///< Publishes control action for actuators
            rclcpp::Publisher<InfoMsg>::SharedPtr m_info_publisher; ///< Publishes controller info for debugging
            rclcpp::Subscription<SplineMsg>::SharedPtr m_spline_subscription; ///< Subscribes to path planning spline
            rclcpp::Subscription<TwistMsg>::SharedPtr m_world_twist_subscription; ///< Subscribes to intertial twist
            rclcpp::Subscription<QuatMsg>::SharedPtr m_world_quat_subscription; ///< Subscribes to intertial quaternion
            rclcpp::Subscription<PoseMsg>::SharedPtr m_world_pose_subscription; ///< Subscribes to inertial pose

            /**
             * Mutex protecting `m_state_estimator`. This needs to be acquired when forwarding callbacks or waiting
             * on `m_state_cond_var`
             */
            std::mutex m_state_mut;

            /**
             * Condition variable for notifying state dirty-ing. MPPI waits on this variable while state and spline
             * callbacks notify it. `m_state_mut` must be acquired before waiting on this.
             * @see std::condition_variable
             */
            std::condition_variable m_state_cond_var;
        };
    }
}
