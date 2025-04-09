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
#include <state/naive_state_tracker.hpp>


namespace controls {
    namespace nodes {
        /**
         * Controller node! ROS node that subscribes to spline and twist, and publishes control actions and debugging
         * information.
         *
         * Refer to the @rst :doc:`explainer </source/explainers/controller>` @endrst for a more detailed overview.
         *
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

            void cone_callback(const ConeMsg& cone_msg);

            /**
             * Callback for world twist subscription. Forwards message to `StateEstimator::on_world_twist`, and notifies MPPI
             * thread of the dirty state. Likely from GPS.
             *
             * @param twist_msg Received twist message
             */
            void world_twist_callback(const TwistMsg& twist_msg);

            /**
             * Callback for world pose subscription. Forwards message to `StateEstimator::on_world_pose`, and notifies MPPI
             * thread of the dirty state. Likely from GPS. Currently unused.
             *
             * @param pose_msg Received pose message
             */
            void world_pose_callback(const PoseMsg& pose_msg);

            void pid_callback(const PIDMsg& pid_msg);

            void world_position_lla_callback(const PositionLLAMsg& position_lla_msg);

            void world_quat_callback(const QuatMsg& quat_ms);

            void imu_accel_callback(const IMUAccelerationMsg& imu_accel_msg);

            /**
             * Publishes a control action to the `control_action` topic.
             *
             * @param action Action to publish
             */
            void publish_action(const Action& action, rclcpp::Time current_time);

            /// Converts MPPI control action output to a ROS2 message. Affected by drive mode (FWD, RWD, AWD).
            /// @param[in] action Control action - output of MPPI.
            ActionMsg action_to_msg(const Action& action);

            /// Converts state to a ROS2 message.
            /// @param[in] state Vehicle state.
            static StateMsg state_to_msg(const State& state);

            //TODO: change the name to print_info or add publishing into it
            /** Prints the information in a InfoMsg to the given stream (usually the console).
             * @param[in] stream Stream to print to
             * @param[in] info Info message to print
             */
            void publish_and_print_info(InfoMsg info, const std::string& additional_info = "");


            /** State estimator instance */
            std::shared_ptr<state::StateEstimator> m_state_estimator;

            /** MPPI Controller instance */
            std::shared_ptr<mppi::MppiController> m_mppi_controller;

            rclcpp::Publisher<ActionMsg>::SharedPtr m_action_publisher; ///< Publishes control action for actuators
            rclcpp::Publisher<InfoMsg>::SharedPtr m_info_publisher; ///< Publishes controller info for debugging
            rclcpp::Publisher<ConeMsg>::SharedPtr m_perc_cones_republisher;
            rclcpp::Subscription<SplineMsg>::SharedPtr m_spline_subscription; ///< Subscribes to path planning spline
            rclcpp::Subscription<TwistMsg>::SharedPtr m_world_twist_subscription; ///< Subscribes to intertial twist
            rclcpp::Subscription<PoseMsg>::SharedPtr m_world_pose_subscription; ///< Subscribes to inertial pose
            rclcpp::Subscription<ConeMsg>::SharedPtr m_cone_subscription;
            rclcpp::Subscription<PIDMsg>::SharedPtr m_pid_subscription;
            rclcpp::Subscription<IMUAccelerationMsg>::SharedPtr m_imu_accel_subscription;
            rclcpp::Subscription<QuatMsg>::SharedPtr m_world_quat_subscription; ///< Subscribes to intertial quaternion
            rclcpp::Subscription<PositionLLAMsg>::SharedPtr m_position_lla_subscription;


            /**
             * Mutex protecting `m_state_estimator`. This needs to be acquired when forwarding callbacks to the
             * state estimator or waiting on `m_state_cond_var`
             */
            std::mutex m_state_mut;

            /**
             * Condition variable for notifying state dirty-ing. MPPI waits on this variable while state and spline
             * callbacks notify it. `m_state_mut` must be acquired before waiting on this.
             * @see std::condition_variable
             */
            std::condition_variable m_state_cond_var;

            std::fstream m_data_trajectory_log;
            float m_last_cone_process_time = 0.0f;
            float m_last_svm_time = 0.0f;

            struct ActionSignal {
                int16_t front_torque_mNm = 0;
                int16_t back_torque_mNm = 0;
                uint16_t velocity_rpm = 0;
                uint16_t rack_displacement_adc = 0;
            };

            ActionSignal action_to_signal(Action action);

            ActionSignal m_last_action_signal;
            std::thread m_aim_communication_thread;
            std::atomic<bool> m_keep_sending_aim_signal = true;
            std::thread launch_aim_communication();
            float m_p_value;
            float m_last_speed;
            float m_last_x_velocity;
            float m_last_y_velocity;
            rclcpp::Time m_last_imu_acceleration_time;
            State get_state_under_strategy(rclcpp::Time current_time);
            state::NaiveStateTracker m_naive_state_tracker;

            // Stuff for the naive state estimator
            
        };
    }
}
