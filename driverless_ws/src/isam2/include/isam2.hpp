#pragma once 
#include <bits/stdc++.h>
// Camera observations of landmarks will be stored as Point2 (x, y).
#include <gtsam/geometry/Point2.h>
#include <gtsam/geometry/Pose2.h>
#include <eigen3/Eigen/Dense>
// Each variable in the system (sposes and landmarks) must be identified with a
// unique key. We can either use simple integer keys (1, 2, 3, ...) or symbols
// (X1, X2, L1). Here we will use Symbols
#include <gtsam/inference/Symbol.h>

// We want to use iSAM2 to solve the structure-from-motion problem
// incrementally, so include iSAM2 here
#include <gtsam/nonlinear/ISAM2.h>

// iSAM2 requires as input a set of new factors to be added stored in a factor
// graph, and initial guesses for any new variables used in the added factors
#include <gtsam/nonlinear/NonlinearFactorGraph.h>
#include <gtsam/nonlinear/Values.h>
#include <gtsam/nonlinear/LevenbergMarquardtOptimizer.h>
#include <gtsam/nonlinear/DoglegOptimizer.h>
#include <gtsam/nonlinear/GaussNewtonOptimizer.h>
#include <gtsam/nonlinear/NonlinearConjugateGradientOptimizer.h>
// In GTSAM, measurement functions are represented as 'factors'. Several common
// factors have been provided with the library for solving robotics/SLAM/Bundle
// Adjustment problems. Here we will use Projection factors to model the
// camera's landmark observations. Also, we will initialize the robot at some
// location using a Prior factor.
#include <gtsam/slam/ProjectionFactor.h>
#include <gtsam/slam/BetweenFactor.h>
#include <gtsam/sam/BearingRangeFactor.h>

#include "ros_utils.hpp"
#include "data_association.hpp"
#include "unary_factor.hpp"
#include "slam_est_and_mcov.hpp"
#include "loop_closure.hpp"

namespace slam {
    using slam_output_t = std::tuple<std::vector<geometry_msgs::msg::Point>, 
                                std::vector<geometry_msgs::msg::Point>, 
                                geometry_msgs::msg::Point>;
    
    enum class RunSettings {
        Real,
        EUFSSim,
        ControlsSim
    };


    class slamISAM {
        private:    
            /* Variables related to ISAM2 factor graph*/
            gtsam::ISAM2Params parameters;
            std::shared_ptr<gtsam::ISAM2> isam2;
            gtsam::NonlinearFactorGraph graph;
            gtsam::Values values;

            /* Functions for adding symbols to the ISAM2 factor graph */
            static gtsam::Symbol X(int robot_pose_id);
            static gtsam::Symbol BLUE_L(int cone_pose_id);
            static gtsam::Symbol YELLOW_L(int cone_pose_id);

            /* Pose and odometry information */
            gtsam::Pose2 first_pose;
            std::size_t pose_num;
            bool first_pose_added = false;

            /* Current vector of mahalanobis distance calculations */
            std::vector<double> m_dist;

            /* Lap and loop closure related variables */
            bool loop_closure;
            bool new_lap;
            std::size_t lap_count;


            /* SLAM history of estimates and marginal covariance matrices */
            SLAMEstAndMCov blue_slam_est_and_mcov;
            SLAMEstAndMCov yellow_slam_est_and_mcov;


            std::size_t checkpoint_to_update_beginning;
            std::size_t blue_checkpoint_id;
            std::size_t yellow_checkpoint_id;



            std::chrono::high_resolution_clock::time_point start;
            std::chrono::high_resolution_clock::time_point end;

            /* Noise models */
            gtsam::Vector LandmarkNoiseModel;
            gtsam::noiseModel::Diagonal::shared_ptr landmark_model;
            gtsam::Vector PriorNoiseModel;
            gtsam::noiseModel::Diagonal::shared_ptr prior_model;
            gtsam::Vector OdomNoiseModel;
            gtsam::noiseModel::Diagonal::shared_ptr odom_model;
            gtsam::Vector UnaryNoiseModel;
            gtsam::noiseModel::Diagonal::shared_ptr unary_model;
            std::optional<rclcpp::Logger> logger;


            /* Tunable and adjustable parameters */
            std::size_t look_radius;
            std::size_t min_cones_update_all;
            std::size_t window_update;
            std::size_t update_start_n;
            std::size_t update_recent_n;

            double imu_offset; // meters; offset from the center of the car
            double lidar_offset; // meters; offset from the center of the car
            double max_cone_range; // meters; how far from the car will we accept a cone to process
            double turning_max_cone_range; //meters; how far from the car will we accept a cone to process
            double dist_from_start_loop_closure_th;
            double m_dist_th;
            double turning_m_dist_th;
            std::size_t update_iterations_n;
            std::size_t return_n_cones;


            /**
             * @brief Initializes the noise models for the SLAM model
             * 
             * @param yaml_noise_inputs An optional NoiseInputs struct containing
             * information from the config yaml file, if one was provided. 
             */
            void init_noise_models(const std::optional<yaml_params::NoiseInputs> &yaml_noise_inputs);

            /**
             * @brief Initializes the tunable parameters for the SLAM model
             * 
             * @param yaml_noise_inputs An optional NoiseInputs struct containing
             * information from the config yaml file, if one was provided. 
             */
            void init_tunable_constants(const std::optional<yaml_params::NoiseInputs> &yaml_noise_inputs);

            /**
             * @brief A helper function for the constructor to initialize the parameters
             * 
             * @param input_logger 
             */
            void init_params(std::optional<rclcpp::Logger> input_logger);

            void print_estimates();

            void log_params_in_use(bool has_value);

            slam_output_t get_recent_SLAM_estimates(gtsam::Pose2 cur_pose);

            /**
             * @brief Updates the poses in the SLAM model. During the first pose, the estimate is not returned
             * for stability.
             * 
             * @param gps_position: An optional GPS position of the car
             * @param yaw: the heading of the car
             * @param velocity: the velocity of the car
             * @param dt: the change in time
             * @param logger: the logger
             * 
             * @return: gtsam::Pose2 representing the current pose of the car
             */
            gtsam::Pose2 update_poses(
                std::optional<gtsam::Point2> gps_position,
                double yaw,
                gtsam::Pose2 velocity, 
                double dt, 
                std::optional<rclcpp::Logger> logger
            );

            /**
             * @brief Updates the landmarks in the SLAM model. This function
             * is used to update the landmarks for a given cone color at a time.
             * This function will update the SLAM model accordingly using the 
             * cone information stored in old_cones and new_cones.
             * This function will also update the slam_est_and_mcov object.
             * 
             * @param old_cones: the old cones
             * @param new_cones: the new cones
             * @param cur_pose: the current pose of the car
             * @param slam_est_and_mcov
             * 
             * @return: returns the new number of landmarks 
             */
            void update_landmarks(
                const std::vector<data_association_utils::OldConeInfo> &old_cones,
                const std::vector<data_association_utils::NewConeInfo> &new_cones,
                gtsam::Pose2 cur_pose,
                SLAMEstAndMCov &slam_est_and_mcov);

        public: 
            slamISAM(std::optional<rclcpp::Logger> input_logger, std::optional<yaml_params::NoiseInputs> &yaml_noise_inputs);
            slamISAM(){}; /* Empty constructor */
            slam_output_t step(
                std::optional<gtsam::Point2> gps_opt, 
                double yaw,
                const std::vector<gtsam::Point2> &cone_obs_blue, 
                const std::vector<gtsam::Point2> &cone_obs_yellow,
                const std::vector<gtsam::Point2> &orange_ref_cones, 
                gtsam::Pose2 velocity,
                double dt
            );
    };
}
