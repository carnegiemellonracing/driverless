#include "isam2.hpp"


namespace slam {
    /**
     * @brief Initializes the noise models for the SLAM model
     * 
     * @param yaml_noise_inputs An optional NoiseInputs struct containing
     * information from the config yaml file, if one was provided. 
     */
    void slamISAM::init_noise_models(const std::optional<yaml_params::NoiseInputs> &yaml_noise_inputs) {
        LandmarkNoiseModel = gtsam::Vector(2);
        OdomNoiseModel = gtsam::Vector(3);
        PriorNoiseModel = gtsam::Vector(3);
        UnaryNoiseModel = gtsam::Vector(2);
        if (!yaml_noise_inputs.has_value()) {
            switch (RunSettings::ControlsSim) {
                case RunSettings::Real:
                    LandmarkNoiseModel(0) = BEARING_STD_DEV; 
                    LandmarkNoiseModel(1) = RANGE_STD_DEV; 

                    OdomNoiseModel(0) = IMU_X_STD_DEV;
                    OdomNoiseModel(1) = IMU_Y_STD_DEV; 
                    OdomNoiseModel(2) = IMU_HEADING_STD_DEV; 
                    // used to be all 0s for EUFS_SIM
                    PriorNoiseModel(0) = IMU_X_STD_DEV;
                    PriorNoiseModel(1) = IMU_Y_STD_DEV;
                    PriorNoiseModel(2) = IMU_HEADING_STD_DEV;

                    UnaryNoiseModel(0) = GPS_X_STD_DEV;
                    UnaryNoiseModel(1) = GPS_Y_STD_DEV;
                    break;
                case RunSettings::EUFSSim:
                    LandmarkNoiseModel(0) = EUFS_SIM_BEARING_STD_DEV; 
                    LandmarkNoiseModel(1) = EUFS_SIM_RANGE_STD_DEV; 

                    OdomNoiseModel(0) = EUFS_SIM_IMU_X_STD_DEV;
                    OdomNoiseModel(1) = EUFS_SIM_IMU_Y_STD_DEV; 
                    OdomNoiseModel(2) = EUFS_SIM_IMU_HEADING_STD_DEV; 
                    // used to be all 0s for EUFS_SIM
                    PriorNoiseModel(0) = EUFS_SIM_IMU_X_STD_DEV;
                    PriorNoiseModel(1) = EUFS_SIM_IMU_Y_STD_DEV;
                    PriorNoiseModel(2) = EUFS_SIM_IMU_HEADING_STD_DEV;

                    UnaryNoiseModel(0) = EUFS_SIM_GPS_X_STD_DEV;
                    UnaryNoiseModel(1) = EUFS_SIM_GPS_Y_STD_DEV;
                    break;
                case RunSettings::ControlsSim:
                    LandmarkNoiseModel(0) = CONTROLS_BEARING_STD_DEV; 
                    LandmarkNoiseModel(1) = CONTROLS_RANGE_STD_DEV; 

                    OdomNoiseModel(0) = CONTROLS_IMU_X_STD_DEV;
                    OdomNoiseModel(1) = CONTROLS_IMU_Y_STD_DEV; 
                    OdomNoiseModel(2) = CONTROLS_IMU_HEADING_STD_DEV; 
                    // used to be all 0s for EUFS_SIM
                    PriorNoiseModel(0) = CONTROLS_IMU_X_STD_DEV;
                    PriorNoiseModel(1) = CONTROLS_IMU_Y_STD_DEV;
                    PriorNoiseModel(2) = CONTROLS_IMU_HEADING_STD_DEV;

                    UnaryNoiseModel(0) = CONTROLS_GPS_X_STD_DEV;
                    UnaryNoiseModel(1) = CONTROLS_GPS_Y_STD_DEV;
                    break;
            }
        } else {
            LandmarkNoiseModel(0) = yaml_noise_inputs.value().yaml_bearing_std_dev; 
            LandmarkNoiseModel(1) = yaml_noise_inputs.value().yaml_range_std_dev; 

            OdomNoiseModel(0) = yaml_noise_inputs.value().yaml_imu_x_std_dev;
            OdomNoiseModel(1) = yaml_noise_inputs.value().yaml_imu_y_std_dev; 
            OdomNoiseModel(2) = yaml_noise_inputs.value().yaml_imu_heading_std_dev; 
            // used to be all 0s for EUFS_SIM
            PriorNoiseModel(0) = yaml_noise_inputs.value().yaml_prior_imu_x_std_dev;
            PriorNoiseModel(1) = yaml_noise_inputs.value().yaml_prior_imu_y_std_dev;
            PriorNoiseModel(2) = yaml_noise_inputs.value().yaml_prior_imu_heading_std_dev;

            UnaryNoiseModel(0) = yaml_noise_inputs.value().yaml_gps_x_std_dev;          
            UnaryNoiseModel(1) = yaml_noise_inputs.value().yaml_gps_y_std_dev;
        }
    }

    /**
     * @brief Initializes the tunable parameters for the SLAM model
     * 
     * @param yaml_noise_inputs An optional NoiseInputs struct containing
     * information from the config yaml file, if one was provided. 
     */
    void slamISAM::init_tunable_constants(const std::optional<yaml_params::NoiseInputs> &yaml_noise_inputs) {
        if  (yaml_noise_inputs.has_value() ) {
            look_radius = yaml_noise_inputs.value().yaml_look_radius;
            min_cones_update_all = yaml_noise_inputs.value().yaml_min_cones_update_all;
            window_update = yaml_noise_inputs.value().yaml_window_update;
            update_start_n = yaml_noise_inputs.value().yaml_update_start_n;
            update_recent_n = yaml_noise_inputs.value().yaml_update_recent_n;

            imu_offset = yaml_noise_inputs.value().yaml_imu_offset;
            lidar_offset = yaml_noise_inputs.value().yaml_lidar_offset;

            max_cone_range = yaml_noise_inputs.value().yaml_max_cone_range;
            turning_max_cone_range = yaml_noise_inputs.value().yaml_turning_max_cone_range;
            dist_from_start_loop_closure_th = yaml_noise_inputs.value().yaml_dist_from_start_loop_closure_th;

            m_dist_th = yaml_noise_inputs.value().yaml_m_dist_th;
            turning_m_dist_th = yaml_noise_inputs.value().yaml_turning_m_dist_th;
            update_iterations_n = yaml_noise_inputs.value().yaml_update_iterations_n;

            return_n_cones = yaml_noise_inputs.value().yaml_return_n_cones;
        } else {
            look_radius = LOOK_RADIUS; // tell us how many cones back and forth to update in slam_est
            min_cones_update_all = MIN_CONES_UPDATE_ALL;
            window_update = WINDOW_UPDATE;
            update_start_n = UPDATE_START_N;
            update_recent_n = UPDATE_RECENT_N;

            imu_offset = IMU_OFFSET; //meters; offset from the center of the car
            lidar_offset = LIDAR_OFFSET; //meters; offset from the center of the car

            max_cone_range = MAX_CONE_RANGE;
            turning_max_cone_range = TURNING_MAX_CONE_RANGE;
            dist_from_start_loop_closure_th = DIST_FROM_START_LOOP_CLOSURE_TH; //meters; distance from the start for loop closure detection

            m_dist_th = M_DIST_TH;
            turning_m_dist_th = TURNING_M_DIST_TH;
            update_iterations_n = UPDATE_ITERATIONS_N;

            return_n_cones = RETURN_N_CONES;
        }
    }

    /**
     * @brief A helper function for the constructor to initialize the parameters
     * Also calls the init_noise_models and init_tunable_params functions
     * 
     * @param yaml_noise_inputs 
     */
    void slamISAM::init_params(std::optional<rclcpp::Logger> input_logger) {

        /* Initializing iSAM2 model */
        parameters = gtsam::ISAM2Params(gtsam::ISAM2DoglegParams(),0.1,10,true);
        parameters.setFactorization("QR");
        logger = input_logger;
        isam2 = std::make_shared<gtsam::ISAM2>(gtsam::ISAM2(parameters));
        graph = gtsam::NonlinearFactorGraph();
        values = gtsam::Values();


        blue_slam_est_and_mcov = SLAMEstAndMCov(isam2, &BLUE_L, look_radius, update_iterations_n);
        yellow_slam_est_and_mcov = SLAMEstAndMCov(isam2, &YELLOW_L, look_radius, update_iterations_n);

        pose_num = static_cast<std::size_t>(0);
        first_pose_added = false;

        // TODO: Consider moving this to SLAMEstAndMCov
        blue_checkpoint_id = static_cast<std::size_t>(0);
        yellow_checkpoint_id = static_cast<std::size_t>(0);

        /* Loop closure variables */
        loop_closure = false;
        new_lap = false;
        lap_count = 0;

    }

    


    slamISAM::slamISAM(std::optional<rclcpp::Logger> input_logger, std::optional<yaml_params::NoiseInputs> &yaml_noise_inputs) {

        /* Initializing SLAM Parameters */
        init_noise_models(yaml_noise_inputs);
        init_tunable_constants(yaml_noise_inputs);
        init_params(input_logger); 

        log_params_in_use(yaml_noise_inputs.has_value());
        


        landmark_model = gtsam::noiseModel::Diagonal::Sigmas(LandmarkNoiseModel);
        odom_model = gtsam::noiseModel::Diagonal::Sigmas(OdomNoiseModel);
        prior_model = gtsam::noiseModel::Diagonal::Sigmas(PriorNoiseModel);
        unary_model = gtsam::noiseModel::Diagonal::Sigmas(UnaryNoiseModel);
    }

    /**
     * @brief Helper function to create a symbol representing the 
     * robot pose in the factor graph of the iSAM2 SLAM model.
     * 
     * @param robot_pose_id 
     * @return gtsam::Symbol 
     */
    gtsam::Symbol slamISAM::X(int robot_pose_id) {
        return gtsam::Symbol('x', robot_pose_id);
    }

    /**
     * @brief Helper function to create a symbol representing the 
     * blue landmark estimate in the factor graph of the iSAM2 SLAM model.
     * 
     * @param cone_id
     * @return gtsam::Symbol 
     */
    gtsam::Symbol slamISAM::BLUE_L(int cone_id) {
        return gtsam::Symbol('b', cone_id);
    }

    /**
     * @brief Helper function to create a symbol representing the 
     * yellow landmark estimate in the factor graph of the iSAM2 SLAM model.
     * 
     * @param cone_id
     * @return gtsam::Symbol 
     */
    gtsam::Symbol slamISAM::YELLOW_L(int cone_id) {
        return gtsam::Symbol('y', cone_id);
    }

    /**
     * @brief Sanity check function for logging the parameters and constants that
     * are in use and whether the yaml file is being used.
     * 
     * @param yaml_has_value Boolean indicating whether or not the yaml file exists.
     */
    void slamISAM::log_params_in_use(bool yaml_has_value) {
        if (yaml_has_value) {
            logging_utils::log_string(logger, "--------Using yaml file --------\n", DEBUG_PARAMS_IN_USE);

        } else {
            logging_utils::log_string(logger, "--------Using default params --------\n", DEBUG_PARAMS_IN_USE);
        }
        logging_utils::log_string(logger, fmt::format("look_radius: {}\n", look_radius), DEBUG_PARAMS_IN_USE);
        logging_utils::log_string(logger, fmt::format("min_cones_update_all: {}\n", min_cones_update_all), DEBUG_PARAMS_IN_USE);
        logging_utils::log_string(logger, fmt::format("window_update: {}\n", window_update), DEBUG_PARAMS_IN_USE);

        logging_utils::log_string(logger, fmt::format("imu_offset: {}\n", imu_offset), DEBUG_PARAMS_IN_USE);
        logging_utils::log_string(logger, fmt::format("max_cone_range: {}\n", max_cone_range), DEBUG_PARAMS_IN_USE);

        logging_utils::log_string(logger, fmt::format("m_dist_th: {}\n", m_dist_th), DEBUG_PARAMS_IN_USE);
        logging_utils::log_string(logger, fmt::format("turning_m_dist_th: {}\n", turning_m_dist_th), DEBUG_PARAMS_IN_USE);
        logging_utils::log_string(logger, fmt::format("update_iterations_n: {}\n", update_iterations_n), DEBUG_PARAMS_IN_USE);
    }


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
    gtsam::Pose2 slamISAM::update_poses(
        std::optional<gtsam::Point2> gps_position,
        double yaw,
        gtsam::Pose2 velocity,
        double dt, 
        std::optional<rclcpp::Logger> logger
    ) {
        logging_utils::log_string(logger, "--------update_poses--------\n", DEBUG_POSES);
        /* Adding poses to the SLAM factor graph */
        gtsam::Point2 offset_xy = motion_modeling::calc_offset_imu_to_car_center(yaw);
        double offset_x = offset_xy.x();
        double offset_y = offset_xy.y(); 

        if (pose_num == 0)
        {
            logging_utils::log_string(logger, "Processing first pose", DEBUG_POSES);
            first_pose = gps_position.has_value() ? 
                            gtsam::Pose2(gps_position.value().x() -offset_x, gps_position.value().y() -offset_y, yaw) :
                            gtsam::Pose2(-offset_x, -offset_y, yaw);


            gtsam::PriorFactor<gtsam::Pose2> prior_factor = gtsam::PriorFactor<gtsam::Pose2>(X(0), first_pose, prior_model);
            graph.add(prior_factor);

            values.insert(X(0), first_pose);

            first_pose_added = true;

            //ASSUMES THAT YOU SEE ORANGE CONES ON YOUR FIRST MEASUREMENT OF LANDMARKS
            //Add orange cone left and right
            //hopefully it's only 2 cones
            logging_utils::log_string(logger, "Finished processing first pose", DEBUG_POSES);
        }
        else
        {
            gtsam::Pose2 prev_pose = isam2->calculateEstimate(X(pose_num - 1)).cast<gtsam::Pose2>();
            logging_utils::log_string(logger, fmt::format("\tprev_pose | x: {} | y: {} | yaw: {}\n", prev_pose.x(), prev_pose.y(), prev_pose.theta()), DEBUG_POSES);

            std::pair<gtsam::Pose2, gtsam::Pose2> new_pose_and_odom = motion_modeling::velocity_motion_model(velocity, dt, prev_pose, yaw);

            gtsam::Pose2 new_pose = gtsam::Pose2(new_pose_and_odom.first.x() - offset_x, 
                                                new_pose_and_odom.first.y() - offset_y, 
                                                new_pose_and_odom.first.theta());
            gtsam::Pose2 odometry = new_pose_and_odom.second;

            gtsam::BetweenFactor<gtsam::Pose2> odom_factor = gtsam::BetweenFactor<gtsam::Pose2>(X(pose_num - 1),
                                                                            X(pose_num),
                                                                            odometry,
                                                                            odom_model);

            graph.add(odom_factor);
            
            if (gps_position.has_value()) {
                gtsam::Pose2 imu_offset_gps_position = gtsam::Pose2(gps_position.value().x() - offset_x, gps_position.value().y() - offset_y, yaw);
                graph.emplace_shared<UnaryFactor>(X(pose_num), imu_offset_gps_position, unary_model);
            }
            
            values.insert(X(pose_num), new_pose);
        }

        isam2->update(graph, values);
        graph.resize(0);
        values.clear();

        for (std::size_t i = 0; i < update_iterations_n; i++) {
            //update the graph
            isam2->update();
        }

        if (pose_num == 0)
        {
            return first_pose;
        }
        return isam2->calculateEstimate(X(pose_num)).cast<gtsam::Pose2>();
        

    }



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
    void slamISAM::update_landmarks(
        const std::vector<data_association_utils::OldConeInfo> &old_cones,
        const std::vector<data_association_utils::NewConeInfo> &new_cones,
        gtsam::Pose2 cur_pose, 
        SLAMEstAndMCov &slam_est_and_mcov)
    {

        /* Bearing range factor will need
        * Types for car pose to landmark node (Pose2, Point2)
        * Bearing of type Rot2 (Rot2 fromAngle)
        * Range of type double
        * Look at PlanarSLAM example in gtsam
        *
        * When adding values:
        * insert Point2 for the cones and their actual location
        *
        */
        for (std::size_t o = 0; o < old_cones.size(); o++)
        {
            gtsam::Point2 cone_pos_car_frame = old_cones.at(o).local_cone_pos;
            int min_id = (old_cones.at(o)).min_id;
            gtsam::Rot2 b = gtsam::Rot2::fromAngle((old_cones.at(o)).bearing);
            double r = gtsam::norm2(cone_pos_car_frame);


            gtsam::Symbol landmark_symbol = slam_est_and_mcov.get_landmark_symbol(min_id);
            graph.add(gtsam::BearingRangeFactor<gtsam::Pose2, gtsam::Point2>(X(pose_num), landmark_symbol,
                                                        b,
                                                        r,
                                                        landmark_model));
        }

        isam2->update(graph, values);
        graph.resize(0);
        // values should be empty
        std::size_t cur_n_landmarks = slam_est_and_mcov.get_n_landmarks();

        for (std::size_t n = 0; n < new_cones.size(); n++)
        {
            gtsam::Point2 cone_pos_car_frame = (new_cones.at(n).local_cone_pos);
            gtsam::Rot2 b = gtsam::Rot2::fromAngle((new_cones.at(n)).bearing);
            double r = gtsam::norm2(cone_pos_car_frame);

            gtsam::Point2 cone_global_frame = (new_cones.at(n).global_cone_pos);

            gtsam::Symbol landmark_symbol = slam_est_and_mcov.get_landmark_symbol(cur_n_landmarks);
            graph.add(gtsam::BearingRangeFactor<gtsam::Pose2, gtsam::Point2>(X(pose_num), landmark_symbol,
                                                        b,
                                                        r,
                                                        landmark_model));

            values.insert(landmark_symbol, cone_global_frame);
            cur_n_landmarks++;
        }

        /* NOTE: All values in graph must be in values parameter */
        values.insert(X(pose_num), cur_pose);
        gtsam::Values optimized_val = gtsam::LevenbergMarquardtOptimizer(graph, values).optimize();
        optimized_val.erase(X(pose_num));
        isam2->update(graph, optimized_val);

        graph.resize(0); // Not resizing your graph will result in long update times
        values.clear();

        /* Update and recalculate estimates in slam_est_and_mcov after updating the iSAM2 model */
        if (old_cones.size() > static_cast<std::size_t>(min_cones_update_all)) {
            std::vector<std::size_t> old_cone_ids(old_cones.size());
            for (std::size_t i = 0; i < old_cones.size(); i++) {
                old_cone_ids.at(i) = static_cast<std::size_t>(old_cones.at(i).min_id);
            }
            slam_est_and_mcov.update_with_old_cones(old_cone_ids);
        } else {
            slam_est_and_mcov.update_and_recalculate_all();
        }

        slam_est_and_mcov.update_with_new_cones(new_cones.size());




    }


     
    slam_output_t slamISAM::get_recent_SLAM_estimates (gtsam::Pose2 cur_pose) {

        std::vector<geometry_msgs::msg::Point> geometry_points_blue = {};
        std::vector<geometry_msgs::msg::Point> geometry_points_yellow = {};

        std::vector<gtsam::Point2> blue_slam_est = blue_slam_est_and_mcov.get_all_est();
        std::vector<gtsam::Point2> yellow_slam_est = yellow_slam_est_and_mcov.get_all_est();

        // If there is less than N cones, take the entire vector
        // Otherwise, take N most recent (from back)
        if (blue_slam_est.size() < return_n_cones) {
            geometry_points_blue = ros_msg_conversions::slam_est_to_points(blue_slam_est, cur_pose);
        } else {
            std::vector<gtsam::Point2> last_n_blue(blue_slam_est.end() - return_n_cones, blue_slam_est.end());
            geometry_points_blue = ros_msg_conversions::slam_est_to_points(last_n_blue, cur_pose);
        } 

        if (yellow_slam_est.size() < return_n_cones){
            geometry_points_yellow = ros_msg_conversions::slam_est_to_points(yellow_slam_est, cur_pose);
        } else   {
            std::vector<gtsam::Point2> last_n_yellow(yellow_slam_est.end() - return_n_cones, yellow_slam_est.end());
            geometry_points_yellow = ros_msg_conversions::slam_est_to_points(last_n_yellow, cur_pose);
        }

        geometry_msgs::msg::Point final_pose = geometry_msgs::msg::Point();
        final_pose.x = cur_pose.x();
        final_pose.y = cur_pose.y();
        final_pose.z = 0.0;

        return std::make_tuple(geometry_points_blue, geometry_points_yellow, final_pose);
    }

    /**
     * @brief Processes odometry information and cone information 
     * to perform an update step on the SLAM model. 
     * 
     * @param gps_opt: the GPS position of the car
     * @param yaw: the heading of the car
     * @param cone_obs: the observed cones
     * @param cone_obs_blue: the observed blue cones
     * @param cone_obs_yellow: the observed yellow cones
     * @param orange_ref_cones: the orange reference cones
     * @param velocity: the velocity of the car
     * @param dt: the change in time
     * 
     * @return: std::tuple<std::vector<Point2>, std::vector<Point2>, Pose2>
     */
    slam_output_t slamISAM::step(
        std::optional<gtsam::Point2> gps_opt, 
        double yaw,
        const std::vector<gtsam::Point2> &cone_obs_blue, 
        const std::vector<gtsam::Point2> &cone_obs_yellow,
        const std::vector<gtsam::Point2> &orange_ref_cones, 
        gtsam::Pose2 velocity,
        double dt
    ) {
        std::size_t old_blue_n_landmarks = blue_slam_est_and_mcov.get_n_landmarks();
        std::size_t old_yellow_n_landmarks = yellow_slam_est_and_mcov.get_n_landmarks();

        if (old_blue_n_landmarks + old_yellow_n_landmarks > 0)
        {
            auto start_step  = std::chrono::high_resolution_clock::now();
            auto dur_betw_step = std::chrono::duration_cast<std::chrono::milliseconds>(start_step - start);
            // logging_utils::log_string(logger, fmt::format("--------End of prev step. Time between step calls: {}--------\n\n", dur_betw_step.count()), DEBUG_STEP);
        }

        start = std::chrono::high_resolution_clock::now();
        
        logging_utils::log_string(logger, "--------Start of SLAM Step--------", DEBUG_STEP);

        

        std::pair<bool, bool> movement_info = motion_modeling::determine_movement(velocity);
        bool is_moving = movement_info.first;
        bool is_turning = movement_info.second;

        /*Quit the update step if the car is not moving*/ 
        if (!is_moving && pose_num > 0) {
            gtsam::Pose2 cur_pose = isam2->calculateEstimate(X(pose_num - 1)).cast<gtsam::Pose2>();
            return get_recent_SLAM_estimates(cur_pose);
        }

        /**** Update the car pose ****/
        auto start_update_poses = std::chrono::high_resolution_clock::now();
        gtsam::Pose2 cur_pose = update_poses(gps_opt, yaw, velocity, dt, logger);
        auto end_update_poses = std::chrono::high_resolution_clock::now();
        auto dur_update_poses = std::chrono::duration_cast<std::chrono::milliseconds>(end_update_poses - start_update_poses);
        logging_utils::log_string(logger, fmt::format("\tUpdate_poses time: {}", dur_update_poses.count()) , true);
        


        /**** Perform loop closure ****/
        auto start_loop_closure = std::chrono::high_resolution_clock::now();
        bool prev_new_lap_value = new_lap;
        new_lap = loop_closure_utils::detect_loop_closure(dist_from_start_loop_closure_th, cur_pose, first_pose, pose_num, logger);

        if (!loop_closure && new_lap) {
            loop_closure = true;
            for (std::size_t i = 0; i < update_iterations_n; i++) {
                isam2->update();
            }
        }

        bool completed_new_lap = prev_new_lap_value && !new_lap && !loop_closure_utils::start_pose_in_front(cur_pose, first_pose, logger);
        if (completed_new_lap) {
            lap_count++;
        }

        auto end_loop_closure = std::chrono::high_resolution_clock::now();
        auto dur_loop_closure = std::chrono::duration_cast<std::chrono::milliseconds>(end_loop_closure - start_loop_closure);
        logging_utils::log_string(logger, fmt::format("\tLoop closure time: {}", dur_loop_closure.count()), DEBUG_STEP);

        if (loop_closure) {
            logging_utils::log_string(logger, "\tLoop closure detected. No longer updating", DEBUG_STEP);
        }

        /**** Retrieve the old cones SLAM estimates & marginal covariance matrices ****/
        if (!loop_closure) {

            /**** Data association ****/
            auto start_DA = std::chrono::high_resolution_clock::now();
            /* For numerical stability, update all estimates and marginal covariances when few cones seen */
            bool has_seen_cones = old_blue_n_landmarks > 0 || old_yellow_n_landmarks > 0; 
            if (has_seen_cones && !(old_blue_n_landmarks > static_cast<std::size_t>(min_cones_update_all) && old_yellow_n_landmarks > static_cast<std::size_t>(min_cones_update_all))) {
                for (std::size_t i = 0; i < update_iterations_n; i++) {
                    isam2->update();
                }
                blue_slam_est_and_mcov.update_and_recalculate_all();
                yellow_slam_est_and_mcov.update_and_recalculate_all();
            } 

            double m_dist_th_to_use = is_turning ? turning_m_dist_th : m_dist_th;
            double cone_dist_th_to_use = is_turning ? turning_max_cone_range : max_cone_range;
            
            auto blue_data_association_info = data_association_utils::perform_data_association(cur_pose, cone_obs_blue, logger, blue_slam_est_and_mcov, m_dist_th_to_use, cone_dist_th_to_use);
            std::vector<data_association_utils::OldConeInfo> blue_old_cones = blue_data_association_info.first;
            std::vector<data_association_utils::NewConeInfo> blue_new_cones = blue_data_association_info.second;

            auto yellow_data_association_info = data_association_utils::perform_data_association(cur_pose, cone_obs_yellow, logger, yellow_slam_est_and_mcov, m_dist_th_to_use, cone_dist_th_to_use);
            std::vector<data_association_utils::OldConeInfo> yellow_old_cones = yellow_data_association_info.first;
            std::vector<data_association_utils::NewConeInfo> yellow_new_cones = yellow_data_association_info.second;
            
            auto end_DA = std::chrono::high_resolution_clock::now();
            auto dur_DA = std::chrono::duration_cast<std::chrono::milliseconds>(end_DA - start_DA);
            logging_utils::log_string(logger, fmt::format("\tData association time: {}", dur_DA.count()), DEBUG_STEP);


            auto start_update_landmarks = std::chrono::high_resolution_clock::now();

            logging_utils::log_string(logger, fmt::format("\t\tStarted updating isam2 model with new and old cones"), DEBUG_STEP);

            update_landmarks(blue_old_cones, blue_new_cones, cur_pose, blue_slam_est_and_mcov);
            update_landmarks(yellow_old_cones, yellow_new_cones, cur_pose, yellow_slam_est_and_mcov);
            
            logging_utils::log_string(logger, fmt::format("\t\tFinished updating isam2 model with new and old cones"), DEBUG_STEP);
            
            auto end_update_landmarks = std::chrono::high_resolution_clock::now();
            auto dur_update_landmarks = std::chrono::duration_cast<std::chrono::milliseconds>(end_update_landmarks - start_update_landmarks);

            logging_utils::log_string(logger, fmt::format("\tUpdate_landmarks time: {}", dur_update_landmarks.count()), DEBUG_STEP);
        }

        



        /* Logging estimates for visualization */
        auto start_vis_setup = std::chrono::high_resolution_clock::now();
        print_estimates();
        auto end_vis_setup = std::chrono::high_resolution_clock::now();
        auto dur_vis_setup = std::chrono::duration_cast<std::chrono::milliseconds>(end_vis_setup - start_vis_setup);

        logging_utils::log_string(logger, fmt::format("\tVis_setup time: {}", dur_vis_setup.count()), DEBUG_VIZ);

        end = std::chrono::high_resolution_clock::now();
        


        auto dur_step_call = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
        logging_utils::log_string(logger, fmt::format("\tSLAM run step | Step call time: {}\n", dur_step_call.count()), DEBUG_STEP);

        logging_utils::log_string(logger, fmt::format("\tpose_num: {} | blue_n_landmarks: {} | yellow_n_landmarks : {}", 
                            pose_num, blue_slam_est_and_mcov.get_n_landmarks(), yellow_slam_est_and_mcov.get_n_landmarks()), DEBUG_STEP);

        pose_num++;

        return get_recent_SLAM_estimates(cur_pose);
    }



    void slamISAM::print_estimates() {
        std::ofstream ofs;
        
        ofs.open(ESTIMATES_FILE, std::ofstream::out | std::ofstream::trunc);
        std::streambuf *coutbuf = std::cout.rdbuf(); //save old buf
        std::cout.rdbuf(ofs.rdbuf());
        std::size_t blue_n_landmarks = blue_slam_est_and_mcov.get_n_landmarks();
        std::size_t yellow_n_landmarks = yellow_slam_est_and_mcov.get_n_landmarks();

        std::vector<gtsam::Point2> blue_slam_est = blue_slam_est_and_mcov.get_all_est();
        std::vector<gtsam::Point2> yellow_slam_est = yellow_slam_est_and_mcov.get_all_est();

        for (std::size_t i = static_cast<std::size_t>(0); i < blue_n_landmarks; i++) {
            gtsam::Point2 blue_cone = blue_slam_est.at(i);
            std::cout << "Value b:" << blue_cone.x() << ":" << blue_cone.y() << std::endl;
        }
        
        for (std::size_t i = static_cast<std::size_t>(0); i < yellow_n_landmarks; i++) {
            gtsam::Point2 yellow_cone = yellow_slam_est.at(i);
            std::cout << "Value y:" << yellow_cone.x() << ":" << yellow_cone.y() << std::endl;
        }

        for (std::size_t i = static_cast<std::size_t>(0); i < pose_num; i++) {
            gtsam::Pose2 cur_pose = isam2->calculateEstimate(X(i)).cast<gtsam::Pose2>();
            std::cout << "Value x:" << cur_pose.x() << ":" << cur_pose.y() << std::endl;
        }
        ofs.close();
        std::cout.rdbuf(coutbuf); //reset to standard output again
    }

}




