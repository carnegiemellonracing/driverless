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

// TODO: MPPI is supposedly included by state_estimator but I can't find it. Weird
#include <state/state_estimator.hpp>
#include <mppi/mppi.hpp>
#include <condition_variable>

#include <rclcpp/rclcpp.hpp>

namespace controls
{
    /**
     * Controller node! ROS node that subscribes to spline and twist, and publishes control actions and debugging
     * information.
     *
     * Refer to the @rst :doc:`explainer </source/explainers/controller>` @endrst for a more detailed overview.
     *
     */
    class OfflineController : public rclcpp::Node
    {
    public:
        /**
         * Construct the controller node. Launches MPPI in a new (detached) thread.
         *
         * @param state_estimator Shared pointer to a state estimator
         * @param mppi_controller Shared pointer to an mppi controller
         */
        OfflineController(
            std::shared_ptr<state::StateEstimator> state_estimator,
            std::shared_ptr<mppi::MppiController> mppi_controller,
            const std::string& input_file);

        void process_file(const std::string &input_file,
                    const std::string &output_file);

    private:


        /** State estimator instance */
        std::shared_ptr<state::StateEstimator> m_state_estimator;

        /** MPPI Controller instance */
        std::shared_ptr<mppi::MppiController> m_mppi_controller;
    };
    
}
