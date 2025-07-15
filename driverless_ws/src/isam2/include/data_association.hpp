#pragma once

#include <vector>
#include <tuple>

#include <eigen3/Eigen/Dense>
#include <gtsam/inference/Symbol.h>
#include <gtsam/nonlinear/ISAM2.h>

#include <algorithm>
#include <float.h>
#include "ros_utils.hpp"
#include "slam_est_and_mcov.hpp"

namespace data_association_utils {

    struct NewConeInfo {
        gtsam::Point2 local_cone_pos;
        double bearing;
        gtsam::Point2 global_cone_pos;

        NewConeInfo (gtsam::Point2 local_cone_pos, double bearing, gtsam::Point2 global_cone_pos)
            : local_cone_pos(local_cone_pos)
            , bearing(bearing)
            , global_cone_pos(global_cone_pos)
        {}
    };

    struct OldConeInfo {
        gtsam::Point2 local_cone_pos;
        double bearing;
        int min_id; // The id of the old cone observed cone was associated with

        OldConeInfo (gtsam::Point2 local_cone_pos, double bearing, int min_id)
            : local_cone_pos(local_cone_pos)
            , bearing(bearing)
            , min_id(min_id)
        {}
    };

    
    /**
     * @brief Distinguishes obsered cones into old and new cones.
     * 
     * @param global_cone_obs The observed cones in the global frame
     * @param cone_obs The observed cones in the local frame
     * @param distances A vector where the ith vector represents 
     *                  mahalanobis distances wrt the ith observed cone
     * @param m_dist_th The mahalanobis distance threshold
     * @param logger 
     * @return std::pair<std::vector<OldConeInfo>, std::vector<NewConeInfo>>
     */
    std::pair<std::vector<OldConeInfo>, std::vector<NewConeInfo>> get_old_new_cones(
        std::vector<gtsam::Point2> global_cone_obs, 
        std::vector<gtsam::Point2> cone_obs, 
        std::vector<std::vector<double>> distances, 
        double m_dist_th,
        std::optional<rclcpp::Logger> logger
    );



    /**
     * @brief Performs data association between the observed cones and the SLAM estimates.
     * 
     * @param cur_pose The current pose of the car
     * @param cone_obs A vector of observed cones
     * @param logger 
     * @param slam_est_and_mcov 
     * @param m_dist_th The Mahalanobis distance threshold
     * 
     * @param cone_dist_th The range threshold to decide which cones to data 
     * associate and which to ignore
     */
    std::pair<std::vector<OldConeInfo>, std::vector<NewConeInfo>> perform_data_association (
        gtsam::Pose2 cur_pose, 
        const std::vector<gtsam::Point2> &cone_obs, 
        std::optional<rclcpp::Logger> logger,
        slam::SLAMEstAndMCov& slam_est_and_mcov,  
        double m_dist_th, 
        double cone_dist_th
    );
}