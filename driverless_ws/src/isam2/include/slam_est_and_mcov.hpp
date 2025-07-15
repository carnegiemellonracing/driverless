#pragma once
#include <gtsam/inference/Symbol.h>
#include <gtsam/nonlinear/ISAM2.h>
#include <gtsam/geometry/Point2.h>
#include <gtsam/geometry/Pose2.h>

#include <eigen3/Eigen/Dense>

#include "ros_utils.hpp"
namespace slam {
    class SLAMEstAndMCov {
        private:
            /* SLAM history of estimates and marginal covariance matrices */
            std::vector<gtsam::Point2> slam_est;
            std::vector<Eigen::MatrixXd> slam_mcov;
            std::size_t n_landmarks;

            /* The function for getting the symbol of the variable that we want to retrieve */
            std::shared_ptr<gtsam::ISAM2> isam2;
            gtsam::Symbol (*cone_key_fn)(int);

            /* Tunable and adjustable parameters */
            std::size_t look_radius;
            std::size_t update_iterations_n;

            /**
             * @brief Calculates the Mahalanobis distance between the observed cone and the
             * old cone estimates. This is the slower version, used for correctness verification.
             * 
             * @param global_obs_cone 
             * @return bool
             */
            bool check_mdist_correctness(gtsam::Point2 global_obs_cone, std::vector<double> mdist_to_check);

        public:
            /* Constructor */
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
            SLAMEstAndMCov(
                std::shared_ptr<gtsam::ISAM2> isam2, 
                gtsam::Symbol(*cone_key_fn)(int), 
                std::size_t look_radius,
                std::size_t update_iterations_n
            );

            SLAMEstAndMCov();


            /* Define functions for updating the estimates */

            /**
             * @brief Recalculates all of the slam estimates and the marginal covariance matrices
             * 
             */
            void update_and_recalculate_all();

            /**
             * @brief Recalulates the estiamtes and the marginal covariance matrices 
             * for the IDs provided in the vector
             * 
             * @param old_cone_ids The IDs of the cones to recalculate estimates and 
             * marginal covariances for
             */
            void update_and_recalculate_by_ID(const std::vector<std::size_t>& old_cone_ids);

            /**
             * @brief Recalculates the cone estimates and the marginal covariance matrices
             * for the first num_start_cones IDs. 
             * 
             * @param num_start_cones The number of cones at the beginning to recalculate
             * estimates and marginal covariances for.
             */
            void update_and_recalculate_beginning(std::size_t num_start_cones);

            /**
             * @brief Recalculates the cone estimates and the marginal covariance matrices
             * for the cones in the look radius of the pivot cone. For some look_radius, 
             * calculate estimates and marginal covariances for the cones with IDs in 
             * [pivot - look_radius, pivot + look_radius] and [pivot, pivot + look_radius].
             * 
             * @param pivot The ID of the pivot cone
             */
            void update_and_recalculate_cone_proximity(std::size_t pivot);

            /**
             * @brief Recalulates the estiamtes and the marginal covariance matrices 
             * for the IDs provided in the vector
             * 
             * @param old_cone_ids The IDs of the cones to recalculate estimates and 
             * marginal covariances for
             */
            void update_with_old_cones(const std::vector<std::size_t>& old_cone_ids);

            /**
            * @brief Adds information about the new cones to slam_est and slam_mcov. 
            * Also updates n_landmarks appropriately
            * 
            * @param num_new_cones The number of new cones/estimates/marginal covariances to add
            */
            void update_with_new_cones(std::size_t num_new_cones);

            /* Interface for data association */

            /**
             * @brief Calculates the Mahalanobis distance between the observed cone and the
             * old cone estimates. The distances are calculated using a SIMD method through 
             * the Eigen library. 
             * 
             * @param global_obs_cone The observed cone that we are data associating in global frame
             * @return std::vector<double> A vector where the ith element represents the mahalanobis distances
             */
            std::vector<double> calculate_mdist (gtsam::Point2 global_obs_cone); //Vectorize this 

            std::size_t get_n_landmarks();

            std::vector<gtsam::Point2> get_all_est();

            /**
             * @brief An invariant function to check that the lengths between the slam estimates
             * and the marginal covariance matrices are the same. 
             * 
             * @return true if the lengths are the same
             * @return false if the lengths are not the same
             */
            bool check_lengths();

            /**
             * @brief Get the landmark symbol object
             * 
             * @param id 
             * @return gtsam::Symbol 
             */
            gtsam::Symbol get_landmark_symbol (int id);
            



    };

}