/**
 * @file data_association.cpp
 * @brief Perform data association in preparation for feeding data to SLAM
 */
#include "data_association.hpp"
namespace data_association_utils {
    /**
     * @brief Distinguishes obsered cones into old and new cones.
     * 
     * @param global_cone_obs The observed cones in the global frame
     * @param cone_obs The observed cones in the local frame
     * @param distances A vector where the ith vector represents 
     *                  mahalanobis distances wrt the ith observed cone
     * 
     * @param logger 
     * @return std::pair<std::vector<OldConeInfo>, std::vector<NewConeInfo>>
     */
    std::pair<std::vector<OldConeInfo>, std::vector<NewConeInfo>> get_old_new_cones(
        std::vector<gtsam::Point2> global_cone_obs, 
        std::vector<gtsam::Point2> cone_obs, 
        std::vector<std::vector<double>> distances, 
        double m_dist_th,
        std::optional<rclcpp::Logger> logger
    ) {
        assert(distances.size() == cone_obs.size());
        assert(global_cone_obs.size() == cone_obs.size());

        if (cone_obs.size() == 0) {
            return std::make_pair(std::vector<OldConeInfo>(), std::vector<NewConeInfo>());
        }

        std::vector<OldConeInfo> old_cones = {};
        std::vector<NewConeInfo> new_cones = {};

        Eigen::MatrixXd bearing = cone_utils::calc_cone_bearing_from_car(cone_obs);

        for (std::size_t i = 0; i < cone_obs.size(); i++) {
            /* Get the mahalanobis distances wrt cone_obs.at(i) */
            std::vector<double> cur_m_dist = distances.at(i);

            if (cur_m_dist.size() != static_cast<std::size_t>(0)) {
                std::vector<double>::iterator min_id_ptr = std::min_element(cur_m_dist.begin(), cur_m_dist.end());
                std::size_t min_id = std::distance(cur_m_dist.begin(), min_id_ptr);

                assert(min_id < cur_m_dist.size());
                assert(min_id >= 0);

                double min_dist = cur_m_dist.at(min_id);

                if (min_dist >= m_dist_th ) {
                    new_cones.emplace_back(cone_obs.at(i),
                                            bearing(i, 0),
                                            global_cone_obs.at(i));
                } else {
                    old_cones.emplace_back(cone_obs.at(i), 
                                            bearing(i, 0), 
                                            min_id);
                }
            } else { /* no landmarks seen before means the current observed cones are all new cones */
                new_cones.emplace_back(cone_obs.at(i),
                                        bearing(i, 0),
                                        global_cone_obs.at(i));
            }

        }

        return std::make_pair(old_cones, new_cones);
    }

    std::pair<std::vector<OldConeInfo>, std::vector<NewConeInfo>> perform_data_association(
        gtsam::Pose2 cur_pose, 
        const std::vector<gtsam::Point2> &cone_obs, 
        std::optional<rclcpp::Logger> logger,
        slam::SLAMEstAndMCov& slam_est_and_mcov,  
        double m_dist_th, 
        double cone_dist_th
    ) {

        std::vector<double> m_dist = {};

        std::vector< gtsam::Point2> relevant_cone_obs = cone_utils::remove_far_cones(cone_obs, cone_dist_th);
        /* From this point onward, only use relevant_cone_obs in place of cone_obs */

        std::vector<gtsam::Point2> global_cone_obs = cone_utils::local_to_global_frame(relevant_cone_obs, cur_pose);
        
        /* A vector where the ith element represents the distances associated with the ith obs cone*/
        std::vector<std::vector<double>> distances = {};
        for (gtsam::Point2 cone : global_cone_obs) {
            distances.push_back(slam_est_and_mcov.calculate_mdist(cone));
        }
        return get_old_new_cones(global_cone_obs, relevant_cone_obs, distances, m_dist_th, logger);
    }


}

