#include "ros_utils.hpp"

namespace loop_closure_utils {
    bool start_pose_in_front(gtsam::Pose2 &cur_pose, gtsam::Pose2 &first_pose, std::optional<rclcpp::Logger> logger);

    bool detect_loop_closure(double dist_from_start_loop_closure_th, gtsam::Pose2 &cur_pose, gtsam::Pose2 &first_pose, int pose_num, std::optional<rclcpp::Logger> logger);
}