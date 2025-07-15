#include "slam_est_and_mcov.hpp"
namespace slam {
    /**
     * @brief Construct a new SLAMEstAndMCov::SLAMEstAndMCov object
     * 
     * @param isam2 A shared pointer to the iSAM2 model that will be used for obtaining
     * cone estimates and marginal covariance matrices.
     * 
     * @param cone_key_fn A function pointer used for getting the symbol of a cone/landmark
     * of interest. This symbol is used for retrieving cone estimates and the marginal 
     * covariance matrix of any cone or landmark given its ID. 
     * 
     * @param look_radius The number of cones to recalculate estimates and marginal covariances
     * for, surrounding some pivot cone ID. 
     * 
     * @param update_iterations_n The number of iterations to run the iSAM2 update for.
     */
    SLAMEstAndMCov::SLAMEstAndMCov(
        std::shared_ptr<gtsam::ISAM2> isam2, 
        gtsam::Symbol(*cone_key_fn)(int), 
        std::size_t look_radius,
        std::size_t update_iterations_n
    ) : isam2(isam2), cone_key_fn(cone_key_fn), look_radius(look_radius), update_iterations_n(update_iterations_n) 
    {
        assert(cone_key_fn != nullptr);
        assert(isam2 != nullptr);
        slam_est = {};
        slam_mcov = {};
        n_landmarks = static_cast<std::size_t>(0);
    }

    SLAMEstAndMCov::SLAMEstAndMCov() {
        slam_est = {};
        slam_mcov = {};
        isam2 = nullptr;
        cone_key_fn = nullptr;
        n_landmarks = static_cast<std::size_t>(0);
        look_radius = static_cast<std::size_t>(0);
        update_iterations_n = static_cast<std::size_t>(0);
    }

    /**
     * @brief An invariant function to check that the lengths between the slam estimates
     * and the marginal covariance matrices are the same. 
     * 
     * @return true if the lengths are the same
     * @return false if the lengths are not the same
     */
    bool SLAMEstAndMCov::check_lengths() {
        return slam_est.size() == slam_mcov.size() && slam_est.size() == n_landmarks;
    }

    

    /* Define functions for updating the estimates */

    /**
     * @brief Recalculates all of the slam estimates and the marginal covariance matrices
     * 
     */       
    void SLAMEstAndMCov::update_and_recalculate_all() {
        for (std::size_t i = 0; i < update_iterations_n; i++) {
            isam2->update();
        }

        for (std::size_t i = static_cast<std::size_t>(0); i < n_landmarks; i++) {
            gtsam::Symbol cone_key = cone_key_fn(i);
            gtsam::Point2 estimate = isam2->calculateEstimate(cone_key).cast<gtsam::Point2>();
            slam_est.at(i) = estimate; 
        }

        for (std::size_t i = static_cast<std::size_t>(0); i < n_landmarks; i++) {
            gtsam::Symbol cone_key = cone_key_fn(i);
            Eigen::MatrixXd mcov = isam2->marginalCovariance(cone_key);
            slam_mcov.at(i) = mcov;
        }
    }


    /**
     * @brief Recalulates the estiamtes and the marginal covariance matrices 
     * for the IDs provided in the vector
     * 
     * @param old_cone_ids The IDs of the cones to recalculate estimates and 
     * marginal covariances for
     */
    void SLAMEstAndMCov::update_and_recalculate_by_ID(const std::vector<std::size_t>& old_cone_ids) {
        for (std::size_t i = 0; i < update_iterations_n; i++) {
            isam2->update();
        }

        for (const std::size_t id: old_cone_ids) {
            gtsam::Symbol cone_key = cone_key_fn(id);
            gtsam::Point2 estimate = isam2->calculateEstimate(cone_key).cast<gtsam::Point2>();
            slam_est.at(id) = estimate; 
        }

        for (std::size_t id: old_cone_ids) {
            gtsam::Symbol cone_key = cone_key_fn(id);
            Eigen::MatrixXd mcov = isam2->marginalCovariance(cone_key);
            slam_mcov.at(id) = mcov;
        }
    }

    /**
     * @brief Recalculates the cone estimates and the marginal covariance matrices
     * for the first num_start_cones IDs. 
     * 
     * @param num_start_cones The number of cones at the beginning to recalculate
     * estimates and marginal covariances for.
     */
    void SLAMEstAndMCov::update_and_recalculate_beginning(std::size_t num_start_cones) {
        for (std::size_t i = 0; i < update_iterations_n; i++) {
            isam2->update();
        }

        for (std::size_t i = static_cast<std::size_t>(0); i < num_start_cones; i++) {
            gtsam::Symbol cone_key = cone_key_fn(i);
            gtsam::Point2 estimate = isam2->calculateEstimate(cone_key).cast<gtsam::Point2>();
            slam_est.at(i) = estimate; 
        }

        for (std::size_t i = static_cast<std::size_t>(0); i < num_start_cones; i++) {
            gtsam::Symbol cone_key = cone_key_fn(i);
            Eigen::MatrixXd mcov = isam2->marginalCovariance(cone_key);
            slam_mcov.at(i) = mcov;
        }
    }

    /**
     * @brief Recalculates the cone estimates and the marginal covariance matrices
     * for the cones in the look radius of the pivot cone. For some look_radius, 
     * calculate estimates and marginal covariances for the cones with IDs in 
     * [pivot - look_radius, pivot + look_radius] and [pivot, pivot + look_radius].
     * Still checks if you are in bounds of the IDs.
     * 
     * @param pivot The ID of the pivot cone
     */
    void SLAMEstAndMCov::update_and_recalculate_cone_proximity(std::size_t pivot) {
        for (std::size_t i = 0; i < update_iterations_n; i++) {
            isam2->update();
        }

        for (std::size_t i = std::max(pivot - look_radius, static_cast<std::size_t>(0)); i < std::min(pivot + look_radius, n_landmarks); i++) {
            gtsam::Symbol cone_key = cone_key_fn(i);
            gtsam::Point2 estimate = isam2->calculateEstimate(cone_key).cast<gtsam::Point2>();
            slam_est.at(i) = estimate; 
        }

        for (std::size_t i = std::max(pivot - look_radius, static_cast<std::size_t>(0)); i < std::min(pivot + look_radius, n_landmarks); i++) {
            gtsam::Symbol cone_key = cone_key_fn(i);
            gtsam::Point2 mcov = isam2->marginalCovariance(cone_key);
            slam_mcov.at(i) = mcov; 
        }

    }

    void SLAMEstAndMCov::update_with_old_cones(const std::vector<std::size_t>& old_cone_ids) {

        if (old_cone_ids.size() == 0) {
            return;
        }

        std::size_t lowest_id = std::numeric_limits<std::size_t>::max();
        std::size_t highest_id = std::numeric_limits<std::size_t>::min();
        for (std::size_t id: old_cone_ids) {
            if (id < lowest_id) {
                lowest_id = id;
            }
            if (id > highest_id) {
                highest_id = id;
            }

        }

        update_and_recalculate_by_ID(old_cone_ids);

        update_and_recalculate_cone_proximity(lowest_id);
        update_and_recalculate_cone_proximity(n_landmarks);
        update_and_recalculate_cone_proximity(highest_id);

        assert(check_lengths());
    }

    /**
     * @brief Adds information about the new cones to slam_est and slam_mcov. 
     * Also updates n_landmarks appropriately
     * 
     * @param num_new_cones The number of new cones/estimates/marginal covariances to add
     */
    void SLAMEstAndMCov::update_with_new_cones(std::size_t num_new_cones) {
        for (std::size_t i = n_landmarks; i < n_landmarks + num_new_cones; i++) {
            gtsam::Symbol cone_key = cone_key_fn(i);
            gtsam::Point2 estimate = isam2->calculateEstimate(cone_key).cast<gtsam::Point2>();
            slam_est.push_back(estimate); 
        }

        for (std::size_t i = n_landmarks; i < n_landmarks + num_new_cones; i++) {
            gtsam::Symbol cone_key = cone_key_fn(i);
            Eigen::MatrixXd mcov = isam2->marginalCovariance(cone_key);
            slam_mcov.push_back(mcov);
        }

        n_landmarks += num_new_cones;

        assert(check_lengths());

    }

    /* Interface for data association */

    /**
     * @brief Calculates the Mahalanobis distance between the observed cone and the
     * old cone estimates. This is the slower version, used for correctness verification.
     * 
     * @param global_obs_cone 
     * @return bool
     */
    bool SLAMEstAndMCov::check_mdist_correctness(gtsam::Point2 global_obs_cone, std::vector<double> mdist_to_check) {
        assert(check_lengths());

        std::vector<double> mdist(n_landmarks);
        for (std::size_t i = 0; i < n_landmarks; i++) {

            Eigen::MatrixXd diff(1, 2);
            diff << global_obs_cone.x() - slam_est.at(i).x(),
                    global_obs_cone.y() - slam_est.at(i).y();

            mdist.at(i) = (diff * slam_mcov.at(i) * diff.transpose())(0, 0);

        }


        bool correct = true;
        for (std::size_t i = 0; i < n_landmarks; i++) {
            if (!correct || (mdist.at(i) != mdist_to_check.at(i))) {
                correct = false;
                break;
            }
        }

        return correct;
    }

    /**
     * @brief Calculates the Mahalanobis distance between the observed cone and the
     * old cone estimates. The distances are calculated using a SIMD method through 
     * the Eigen library. 
     * 
     * @param global_obs_cone The observed cone that we are data associating in global frame
     * @return Eigen::MatrixXd 
     */
    std::vector<double> SLAMEstAndMCov::calculate_mdist (gtsam::Point2 global_obs_cone) {
        assert(check_lengths()); 

        /* 1.) (2*i) and (2*i +1) column be (x,y) difference vector between slam_est.at(i) and global_obs_cone */
        Eigen::MatrixXd diff_dupe(2, 2 * n_landmarks);
        for (std::size_t i = static_cast<std::size_t>(0); i < n_landmarks; i++) {
            Eigen::MatrixXd cur_diff(2, 1);
            cur_diff << slam_est.at(i).x() - global_obs_cone.x(),
                        slam_est.at(i).y() - global_obs_cone.y();

            diff_dupe.block(0, 2*i, 2, 1) = cur_diff;
            diff_dupe.block(0, 2*i + 1, 2, 1) = cur_diff;
        }

        /* 2.) ith 2x2 block row-wise should be the marginal covariance matrix */
        // TODO: Should it really be the inverse of the covariance matrix?
        // Experiment with the inverse of the covariance matrix
        Eigen::MatrixXd sigma(2, 2 * n_landmarks);
        for (std::size_t i = static_cast<std::size_t>(0); i < n_landmarks; i++) {
            sigma.block(0, 2 * i, 2, 2) = slam_mcov.at(i);
        }
        

        /* 3.) Perform element-wise multiplication with covariance matrices. (Matmul but no adding)*/
        Eigen::MatrixXd diff_with_sigma_half_matmul = diff_dupe.array() * sigma.array();
        /* Note: the ith diff vector matmul with ith covariance matrix is stored as 1x2 row vectors */
        Eigen::MatrixXd pre_diff_matmul_sigma = diff_with_sigma_half_matmul.row(0) + diff_with_sigma_half_matmul.row(1);


        /** 4.) Reorganize the matrix so that the ith column represents the product between 
         * the ith diff vector with the ith covariance matrix 
         */
        Eigen::MatrixXd diff_matmul_sigma(2, n_landmarks);
        for (std::size_t i = static_cast<std::size_t>(0); i < n_landmarks; i++) {
            diff_matmul_sigma.block(0, i, 2, 1) = (pre_diff_matmul_sigma.block(0, 2*i, 1, 2)).transpose();
        }
        /* 5.) ith column represents the diff between slam_est.at(i) and cone_obs*/
        Eigen::MatrixXd diff = Eigen::MatrixXd::Zero(2, n_landmarks);
        for (std::size_t i = static_cast<std::size_t>(0); i < n_landmarks; i++) {
            diff.block(0, i, 2, 1) << slam_est.at(i).x() - global_obs_cone.x(),
                                    slam_est.at(i).y() - global_obs_cone.y();
        }

        /* 6.) Complete the mahalanobis distance calculation */
        Eigen::MatrixXd mdist_half_matmul = diff_matmul_sigma.array() * diff.array();
        Eigen::MatrixXd mdist_eigen = mdist_half_matmul.row(0) + mdist_half_matmul.row(1);

        std::vector<double> mdist(n_landmarks);
        for (std::size_t i = static_cast<std::size_t>(0); i < n_landmarks; i++) {
            mdist.at(i) = mdist_eigen(0, i);
        }

        assert(check_mdist_correctness(global_obs_cone, mdist));

        return mdist;
    }

    /**
     * @brief Gets the number of landmarks in the SLAM estimates
     * 
     * @return std::size_t 
     */
    std::size_t SLAMEstAndMCov::get_n_landmarks() {
        return n_landmarks;
    }

    /**
     * @brief Gets the SLAM cone estimates
     * 
     * @return std::vector<gtsam::Point2> 
     */
    std::vector<gtsam::Point2> SLAMEstAndMCov::get_all_est() {
        return slam_est;
    }

    /**
     * @brief Get the landmark symbol object
     * 
     * @param id 
     * @return gtsam::Symbol 
     */
    gtsam::Symbol SLAMEstAndMCov::get_landmark_symbol (int id) {
        assert(cone_key_fn != nullptr);
        return cone_key_fn(id);
    }
}