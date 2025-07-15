/**
 * @file ros_utils.cpp
 * @brief This file contains utility functions used for converting ROS messages
 * to appropriate data types and for any necessary calculations.
 *
 * Most functions in this value pass in the result container by reference.
 * This reference is modified to store the results. The result reference
 * is always the first parameter that is passed in.
 */
#include "ros_utils.hpp"


namespace ros_msg_conversions {
    void cone_msg_to_vectors(
        const interfaces::msg::ConeArray::ConstSharedPtr &cone_data,
        std::vector<gtsam::Point2> &cones,
        std::vector<gtsam::Point2> &blue_cones,
        std::vector<gtsam::Point2> &yellow_cones,
        std::vector<gtsam::Point2> &orange_cones
    ) {
        // TODO Trying to find a library that converts ros2 msg array to vector
        for (std::size_t i = 0; i < cone_data->blue_cones.size(); i++) {
            gtsam::Point2 to_add = gtsam::Point2(cone_data->blue_cones.at(i).x,
                                cone_data->blue_cones.at(i).y);
            blue_cones.push_back(to_add);
            cones.push_back(to_add);
        }

        for (std::size_t i = 0; i < cone_data->yellow_cones.size(); i++) {
            gtsam::Point2 to_add = gtsam::Point2(cone_data->yellow_cones.at(i).x,
                                cone_data->yellow_cones.at(i).y);
            yellow_cones.push_back(to_add);
            cones.push_back(to_add);
        }

        for (std::size_t i = 0; i < cone_data->orange_cones.size(); i++) {
            gtsam::Point2 to_add = gtsam::Point2(cone_data->orange_cones.at(i).x,
                                cone_data->orange_cones.at(i).y);
            orange_cones.push_back(to_add);
            cones.push_back(to_add);
        }
    }

    gtsam::Pose2 velocity_msg_to_pose2(const geometry_msgs::msg::TwistStamped::ConstSharedPtr &vehicle_vel_data) {
        double dx = vehicle_vel_data->twist.linear.x;
        double dy = vehicle_vel_data->twist.linear.y;
        double dw = vehicle_vel_data->twist.angular.z;
        
        return gtsam::Pose2(dx, dy, dw);
    }

    gtsam::Pose2 posestamped_msg_to_pose2(const geometry_msgs::msg::PoseStamped::ConstSharedPtr &vehicle_pos_data, gtsam::Point2 init_x_y, rclcpp::Logger logger) {
        /** 
         * PoseStamped message has:
         * 1.) Header header
         * 2.) Pose pose
         * 
         * geometry_msgs Pose type has:
         * 1.) Point position
         * 2.) Quaternion orientation
         */

        /* Process the pose */
        double x = vehicle_pos_data->pose.position.x;
        double y = vehicle_pos_data->pose.position.y;

        x -= init_x_y.x();
        y -= init_x_y.y();

        /* Process the orientation */
        double yaw = vehicle_pos_data->pose.position.z;

        return gtsam::Pose2(x, y, yaw);

    }


    double quat_msg_to_yaw(const geometry_msgs::msg::QuaternionStamped::ConstSharedPtr &vehicle_angle_data) {
        double qw = vehicle_angle_data->quaternion.w;
        double qx = vehicle_angle_data->quaternion.x;
        double qy = vehicle_angle_data->quaternion.y;
        double qz = vehicle_angle_data->quaternion.z;

        return std::atan2(2 * (qz * qw + qx * qy), -1 + 2 * (qw * qw + qx * qx));
    }


    gtsam::Point2 vector3_msg_to_gps(const geometry_msgs::msg::Vector3Stamped::ConstSharedPtr &vehicle_pos_data, gtsam::Point2 init_lon_lat, rclcpp::Logger logger) {
        /* Doesn't depend on imu axes. These are global coordinates */ 
        double latitude =  vehicle_pos_data->vector.x;
        double longitude = vehicle_pos_data->vector.y;
 
        longitude -= init_lon_lat.x();
        latitude -=  init_lon_lat.y();
        /* Print statements for longitude and latitude debugging
        RCLCPP_INFO(logger, "init_lon_lat: %.10f | %.10f", init_lon_lat.value().x(),
                                                    init_lon_lat.value().y());
        RCLCPP_INFO(logger, "cur lon_lat: %.10f | %.10f", longitude, latitude);
        RCLCPP_INFO(logger, "cur change in lon_lat: %.10f | %.10f",
                                                longitude, latitude); */

        double LAT_DEG_TO_METERS = 111111; //! Move this over to constants.hpp

        /* Intuition: Find the radius of the circle at current latitude
        * Convert the change in longitude to radians
        * Calculate the distance in meters
        *
        * The range should be the earth's radius: 6378137 meters.
        * The radius of the circle at current latitude: 6378137 * cos(latitude_rads)
        * To get the longitude, we need to convert change in longitude to radians
        * - longitude_rads = longitude * (M_PI / 180.0)
        *
        * Observe: 111320 = 6378137 * M_PI / 180.0
        */

        /* Represents the radius used to multiply angle in radians */
        double LON_DEG_TO_METERS = 111319.5 * std::cos(motion_modeling::degrees_to_radians(latitude));

        double x = LON_DEG_TO_METERS * longitude;
        double y = LAT_DEG_TO_METERS * latitude;

        return gtsam::Point2(x, y);
    }

    geometry_msgs::msg::Point point2_to_geometry_msg (gtsam::Point2 gtsam_point) {
        geometry_msgs::msg::Point point = geometry_msgs::msg::Point();
        point.x = gtsam_point.x();
        point.y = gtsam_point.y();
        point.z = 0.0;
        return point;
    }

    std::vector<geometry_msgs::msg::Point> slam_est_to_points (std::vector<gtsam::Point2> gtsam_points, gtsam::Pose2 pose) {
        std::vector<geometry_msgs::msg::Point> points = {};
        std::vector<gtsam::Point2> local_cones = cone_utils::global_to_local_frame(gtsam_points, pose);

        for (gtsam::Point2 cone : local_cones) {
            geometry_msgs::msg::Point point = geometry_msgs::msg::Point();
            point.x = cone.x();
            point.y = cone.y();
            point.z =  0.0;
            points.push_back(point);
        }
        
        return points;
    }
}




namespace motion_modeling {


    void imu_axes_to_DV_axes(double &x, double &y) {
        double temp_x = x;
        x = -1 * y;
        y = temp_x;
    }

    /**
     * @brief Calculates the lateral velocity error as a result of turning.
     *        The IMU is offset from the center of the car. There's extra lateral
     *        velocity error due to the offset.
     * @return Returns a pair where the first element is the new_pose and the 
     * second element is the odometry.
     */
    gtsam::Point2 calc_lateral_velocity_error(double ang_velocity, double yaw) {
        /** Intuition: In the local frame, create a triangle T from the global bearing of the car
         *  
         * ang_velocity represents the magnitude of the angular velocity vector
         * - To get the x component (global frame x), 
         *      let the vertical side of the triangle be the radius
         * 
         * - To get the y component (global frame y),
         *      let the horizontal side of the triangle be the radius
         */
        double dx_error = ang_velocity * (IMU_OFFSET * std::sin(yaw));
        double dy_error = ang_velocity * (IMU_OFFSET * std::cos(yaw));
        return gtsam::Point2(dx_error, dy_error);
    }

    std::pair<gtsam::Pose2, gtsam::Pose2> velocity_motion_model(gtsam::Pose2 velocity, double dt, gtsam::Pose2 prev_pose, double yaw) {
        
        gtsam::Point2 dx_dy_error = calc_lateral_velocity_error(velocity.theta(), yaw);
        
        double dx = velocity.x() - dx_dy_error.x();
        double dy = velocity.y() - dx_dy_error.y(); 

        gtsam::Pose2 new_pose = gtsam::Pose2(prev_pose.x() + dx * dt,
                        prev_pose.y() + dy * dt,
                        yaw);

        //TODO: how does change in theta results differ from dt * velocity.z()
        gtsam::Pose2 odometry = gtsam::Pose2(dx * dt,
                        dy * dt,
                        yaw - prev_pose.theta());
        return std::make_pair(new_pose, odometry);
    } 

    gtsam::Point2 calc_offset_imu_to_car_center(double yaw) {
        double offset_x = IMU_OFFSET * std::cos(yaw);
        double offset_y = IMU_OFFSET * std::sin(yaw);
        return gtsam::Point2(offset_x, offset_y);
    }

    gtsam::Point2 calc_offset_lidar_to_car_center(double yaw) {
        double offset_x = LIDAR_OFFSET * std::cos(yaw);
        double offset_y = LIDAR_OFFSET * std::sin(yaw);
        return gtsam::Point2(offset_x, offset_y);
    }

    /**
     * @brief 
     * 
     * @param odometry 
     * @param velocity 
     * @param dt 
     * @param prev_pose 
     * @param global_odom 
     * @return std::pair<gtsam::Pose2, gtsam::Pose2> Returns a pair where the first element is the new_pose and the 
     * second element is the odometry. 
     */
    std::pair<gtsam::Pose2, gtsam::Pose2> gps_motion_model(gtsam::Pose2 prev_pose, gtsam::Pose2 global_odom) {

        gtsam::Point2 offset = calc_offset_imu_to_car_center(global_odom.theta());
        double offset_x = offset.x();
        double offset_y = offset.y();

        gtsam::Pose2 new_pose = gtsam::Pose2(global_odom.x() - offset_x, global_odom.y() - offset_y, global_odom.theta());
        gtsam::Pose2 odometry = gtsam::Pose2(
                            new_pose.x() - prev_pose.x(),
                            new_pose.y() - prev_pose.y(),
                            global_odom.theta() - prev_pose.theta());

        return std::make_pair(new_pose, odometry);
    }

    std::pair<bool, bool> determine_movement(gtsam::Pose2 velocity) {
        bool is_moving = false;
        bool is_turning = false;

        if (std::sqrt(std::pow(velocity.x(), 2) + std::pow(velocity.y(), 2)) > VELOCITY_MOVING_TH) {
            is_moving = true;
        } else {
            is_moving = false;
        }
        if (std::abs(velocity.theta()) > TURNING_TH) {
            is_turning = true;
        } else {
            is_turning = false;
        }

        return std::make_pair(is_moving, is_turning);
        
    }


    double header_to_nanosec(const std_msgs::msg::Header &header) {
        return (header.stamp.sec * (double)1e9) + header.stamp.nanosec;
    }

    double header_to_dt(std::optional<std_msgs::msg::Header> prev, std::optional<std_msgs::msg::Header> cur) {
        if (!(prev.has_value() && cur.has_value())) {
            return header_to_nanosec(cur.value()) * 1e-9;
        }
        // dt units is meters per second
        return (header_to_nanosec(cur.value()) - header_to_nanosec(prev.value())) * (double)1e-9;
    }

    double degrees_to_radians(double degrees) {
        return degrees * M_PI / 180.0;
    }

}



namespace cone_utils {
    /* Vectorized functions */
    Eigen::MatrixXd calc_cone_range_from_car(const std::vector<gtsam::Point2> &cone_obs) {
        Eigen::MatrixXd range(cone_obs.size(), 1);

        for (size_t i = 0; i < cone_obs.size(); i++)
        {
            range(i,0) = gtsam::norm2(cone_obs.at(i));
        }

        return range;
    }

    /* Bearing of cone from the car */
    Eigen::MatrixXd calc_cone_bearing_from_car(const std::vector<gtsam::Point2> &cone_obs) {
        Eigen::MatrixXd bearing(cone_obs.size(), 1);

        for (std::size_t i = 0; i < cone_obs.size(); i++)
        {   
            /**
             * Intuition:
             * Perceptions gives us cones in DV axes: y forwards and x right
             * 
             * Directly in front of us, the bearing is 0
             * - The bearing is the angle from the axis directly in front of us
             * - counter clockwise is positive bearing
             * - clockwise is negative bearing
             * 
             * Given the DV axes, to calculate the bearing is atan2(-x, y)
             * - If the bearing was to the left (counter clockwise from 0)
             * - The bearing would be a positive angle 
             * 
             */
            bearing(i,0) = std::atan2(-cone_obs.at(i).x(), cone_obs.at(i).y());
        }
        return bearing;
    }

    /**
     * @brief Removes far away observed cones. Observed cones that are far away are more erroneous
     * 
     * @param cone_obs The observed cones
     * @param threshold The threshold distance from the car
     */
    std::vector<gtsam::Point2> remove_far_cones(std::vector<gtsam::Point2> cone_obs, double threshold) {
        for (std::size_t i = 0; i < cone_obs.size(); i++) {
            if (gtsam::norm2(cone_obs.at(i)) > threshold) {
                cone_obs.erase(cone_obs.begin() + i);
                i--;
            }
        }

        return cone_obs;

        
    }

    /* Unvectorized version */
    // gtsam::Point2 global_to_local_frame(
    //     const gtsam::Point2 cone_position,
    //     const gtsam::Pose2 cur_pose
    // ) {
    //     double global_dx = cone_position.x() - cur_pose.x();
    //     double global_dy = cone_position.y() - cur_pose.y();

    //     /* Think of right triangles but put the right angle on the CMR/DV axes instead of IMU axes */
    //     double local_dy = global_dx * std::cos(cur_pose.theta()) + global_dy * std::sin(cur_pose.theta());
    //     double local_dx = global_dx * std::sin(cur_pose.theta()) - global_dy * std::cos(cur_pose.theta());

    //     gtsam::Point2 local_point = gtsam::Point2(local_dx, local_dy);
    //     return local_point;
    // }

    std::vector<gtsam::Point2> global_to_local_frame(std::vector<gtsam::Point2> cone_obs, gtsam::Pose2 cur_pose) {
        Eigen::MatrixXd global_dx(cone_obs.size(), 1);
        Eigen::MatrixXd global_dy(cone_obs.size(), 1);

        for (std::size_t i = 0; i < cone_obs.size(); i++) {
            global_dx(i, 0) = cone_obs.at(i).x() - cur_pose.x();
        }

        for (std::size_t i = 0; i < cone_obs.size(); i++) {
            global_dy(i, 0) = cone_obs.at(i).y() - cur_pose.y();
        }

        Eigen::MatrixXd local_dx = global_dx.array() * std::cos(cur_pose.theta()) + global_dy.array() * std::sin(cur_pose.theta());
        Eigen::MatrixXd local_dy = global_dx.array() * std::sin(cur_pose.theta()) - global_dy.array() * std::cos(cur_pose.theta());

        std::vector<gtsam::Point2> local_cone_obs = {};
        for (std::size_t i = 0; i < cone_obs.size(); i++) {
            local_cone_obs.emplace_back(local_dx(i, 0), local_dy(i, 0));    
        }

        return local_cone_obs;
    }



    std::vector<gtsam::Point2> local_to_global_frame( std::vector<gtsam::Point2> cone_obs, gtsam::Pose2 cur_pose) {

        Eigen::MatrixXd bearing = calc_cone_bearing_from_car(cone_obs);
        Eigen::MatrixXd range = calc_cone_range_from_car(cone_obs);
        
        gtsam::Point2 lidar_offset = motion_modeling::calc_offset_lidar_to_car_center(cur_pose.theta());
        double lidar_offset_x = lidar_offset.x();       
        double lidar_offset_y = lidar_offset.y();

        Eigen::MatrixXd global_heading = bearing.array() + cur_pose.theta();
        Eigen::MatrixXd global_cone_x = cur_pose.x() + range.array()*global_heading.array().cos();
        global_cone_x = global_cone_x.array() + lidar_offset_x;
        Eigen::MatrixXd global_cone_y = cur_pose.y() + range.array()*global_heading.array().sin();
        global_cone_y = global_cone_y.array() + lidar_offset_y;

        std::vector<gtsam::Point2> global_cone_obs = {};
        for (std::size_t i = 0; i < cone_obs.size(); i++) {
            global_cone_obs.emplace_back(global_cone_x(i, 0), global_cone_y(i,0));
        }
        return global_cone_obs;
        
    }



    
}

namespace logging_utils {
    /**
     * @brief This function cases on whether the logger exists and 
     * uses it to print out the following string
     * 
     * @param logger An optional logger 
     * @param input_string The input string to log
     * @param flag A flag to determine whether to actually print the outputs
     */
    void log_string (std::optional<rclcpp::Logger> logger, std::string input_string, bool flag) {
        if (logger.has_value() && flag) {
            RCLCPP_INFO(logger.value(), input_string.c_str());
        }
    }


    void print_cone_obs(const std::vector<gtsam::Point2> &cone_obs, const std::string& cone_color, std::optional<rclcpp::Logger> logger) {
        for (std::size_t i = 0; i < cone_obs.size(); i++) {
            if (logger.has_value()) {
                RCLCPP_INFO(logger.value(), "%s.at(%ld): %f %f", cone_color.c_str(), i, 
                                                        cone_obs.at(i).x(),
                                                        cone_obs.at(i).y());
            }
        }
    }

    void log_step_input(
        std::optional<rclcpp::Logger> logger, 
        std::optional<gtsam::Point2> gps_opt,
        double yaw,
        const std::vector<gtsam::Point2> &cone_obs_blue, 
        const std::vector<gtsam::Point2> &cone_obs_yellow,
        const std::vector<gtsam::Point2> &orange_ref_cones, 
        gtsam::Pose2 velocity, 
        double dt
    ) {
        if (logger.has_value()) {
            RCLCPP_INFO(logger.value(), "PRINTING STEP INPUTS");
            if (gps_opt.has_value()) {
                RCLCPP_INFO(logger.value(), "gps_opt: %f %f", gps_opt.value().x(), gps_opt.value().y());
            } else {
                RCLCPP_INFO(logger.value(), "gps_opt: None");
            }
            RCLCPP_INFO(logger.value(), "yaw: %f", yaw);
            print_cone_obs(cone_obs_blue, "cone_obs_blue", logger);
            print_cone_obs(cone_obs_yellow, "cone_obs_yellow", logger);
            print_cone_obs(orange_ref_cones, "orange_ref_cones", logger);

        // At the moment, cone_obs_blue and cone_obs_yellow are empty because we don't have coloring
        // Similarly for the orange_ref_cones
            RCLCPP_INFO(logger.value(), "velocity: %f %f %f", velocity.x(), velocity.y(), velocity.theta());
            RCLCPP_INFO(logger.value(), "dt: %f", dt);
        }
    }
        
                            
    void print_update_poses(gtsam::Pose2 &prev_pose, gtsam::Pose2 &new_pose, gtsam::Pose2 &odometry, gtsam::Pose2 &imu_offset_global_odom, std::optional<rclcpp::Logger> logger) {
        if (logger.has_value()) {
            RCLCPP_INFO(logger.value(), "\tLOGGING UPDATE_POSES POSE2 VARIABLES:");
            RCLCPP_INFO(logger.value(), "prev_pose: %f %f %f", prev_pose.x(), prev_pose.y(), prev_pose.theta());
            RCLCPP_INFO(logger.value(), "new_pose: %f %f %f", new_pose.x(), new_pose.y(), new_pose.theta());
            RCLCPP_INFO(logger.value(), "odometry: %f %f %f", odometry.x(), odometry.y(), odometry.theta());
            RCLCPP_INFO(logger.value(), "imu_offset_global_odom: %f %f %f", imu_offset_global_odom.x(), 
                                                                    imu_offset_global_odom.y(), 
                                                                    imu_offset_global_odom.theta());
            RCLCPP_INFO(logger.value(), "\n");
        }
    }

    void record_step_inputs(
        std::optional<rclcpp::Logger> logger, 
        std::optional<gtsam::Point2> gps_opt,
        double yaw,
        const std::vector<gtsam::Point2> &cone_obs_blue, 
        const std::vector<gtsam::Point2> &cone_obs_yellow,
        const std::vector<gtsam::Point2> &orange_ref_cones, 
        gtsam::Pose2 velocity, 
        double dt
    ) {

        std::ofstream outfile;
        outfile.open(STEP_INPUT_FILE, std::ofstream::out | std::ios_base::app); 
        if (gps_opt.has_value()) {
            outfile << "gps_opt: " << gps_opt.value().x() << " " << gps_opt.value().y() << std::endl;
        } else {
            outfile << "gps_opt: No value" << std::endl;
        }

        outfile << "yaw: " << yaw << std::endl;
        for (std::size_t i = 0; i < cone_obs_blue.size(); i++) {
            outfile << "cone_obs_blue: " << cone_obs_blue.at(i).x() << " " << cone_obs_blue.at(i).y() << std::endl;
        }
        for (std::size_t i = 0; i < cone_obs_yellow.size(); i++) {
            outfile << "cone_obs_yellow: " << cone_obs_yellow.at(i).x() << " " << cone_obs_yellow.at(i).y() << std::endl;
        }
        for (std::size_t i = 0; i < orange_ref_cones.size(); i++) {
            outfile << "orange_ref_cones: " << orange_ref_cones.at(i).x() << " " << orange_ref_cones.at(i).y() << std::endl;
        }
        outfile << "velocity: " << velocity.x() << " " << velocity.y() << " " << velocity.theta() << std::endl;
        outfile << "dt: " << dt << "\n" << std::endl;

        outfile.close();

    }
}
