#include "rclcpp/rclcpp.hpp"

#include "geometry_msgs/msg/point.hpp"
#include "geometry_msgs/msg/vector3_stamped.hpp"
#include "geometry_msgs/msg/twist_stamped.hpp"

#include "interfaces/msg/cone_array.hpp"
#include "../cones/svm/svm_recoloring.hpp"


// Standard Imports
#include <deque>
#include <queue>
#include <memory>
#include <chrono>
#include <filesystem>


/**
 * @brief We motion model on the position of the car to the observer, instead
 * of thinking in global frame wrt to the start. This is because we want the 
 * current position to be the most accurate, and other positions to be less accurate.
 * 
 */
struct ObsConeInfo {
    // Global frame
    double cur_car_to_observer_x;
    double cur_car_to_observer_y;
    double observer_yaw;

    // CMR/Local frame 
    double observer_position_to_cone_x;
    double observer_position_to_cone_y;

    int lifespan;

    ObsConeInfo (
        double input_cur_car_to_observer_x, 
        double input_cur_car_to_observer_y, 
        double input_observer_yaw,
        double input_observer_position_to_cone_x,
        double input_observer_position_to_cone_y,
        int input_ls
    ) : cur_car_to_observer_x(input_cur_car_to_observer_x),
      cur_car_to_observer_y(input_cur_car_to_observer_y),
      observer_yaw(input_observer_yaw),
      observer_position_to_cone_x(input_observer_position_to_cone_x),
      observer_position_to_cone_y(input_observer_position_to_cone_y),
      lifespan(input_ls) {}

};
class ConeHistoryTestNode : public rclcpp::Node {
public:
    ConeHistoryTestNode();

private:
    // Message queues 
    std::deque<geometry_msgs::msg::TwistStamped::SharedPtr> velocity_deque;
    std::deque<geometry_msgs::msg::Vector3Stamped::SharedPtr> yaw_deque;
    static constexpr int max_deque_size = 100;
    static constexpr int max_timesteps_in_cone_history = 10;
    static constexpr int max_long_term_history_size = 300;

    std::queue<ObsConeInfo> yellow_cone_history;
    std::queue<ObsConeInfo> blue_cone_history;
    std::queue<ObsConeInfo> long_term_blue_cone_history;
    std::queue<ObsConeInfo> long_term_yellow_cone_history;
    double min_dist_th = 0.35;

    uint64_t prev_time_stamp;

    std::pair<geometry_msgs::msg::TwistStamped::SharedPtr, geometry_msgs::msg::Vector3Stamped::SharedPtr> get_velocity_yaw(uint64_t frameTime);

    // Motion Modeling Callbacks
    void cone_callback(const interfaces::msg::ConeArray::SharedPtr cone_msg);
    void velocity_callback(const geometry_msgs::msg::TwistStamped::SharedPtr msg);
    void yaw_callback(const geometry_msgs::msg::Vector3Stamped::SharedPtr msg);

    // Define subscribers and publishers
    rclcpp::Subscription<interfaces::msg::ConeArray>::SharedPtr perc_cones_sub_;
    rclcpp::Subscription<geometry_msgs::msg::TwistStamped>::SharedPtr velocity_sub_;
    rclcpp::Subscription<geometry_msgs::msg::Vector3Stamped>::SharedPtr yaw_sub_;

    rclcpp::Publisher<interfaces::msg::ConeArray>::SharedPtr associated_cones_pub_;

    std::mutex velocity_mutex;
    std::mutex yaw_mutex;

    int classify_through_data_association(geometry_msgs::msg::Vector3 lidar_point, double yaw);
    void motion_model_on_cone_history(std::queue<ObsConeInfo>& cone_history, std::pair<double, double> global_xy_change);
    // void add_lidar_point_to_cone_history(std::queue<ObsConeInfo>& cone_history, geometry_msgs::msg::Vector3 lidar_point);
    void maintain_cone_history_lifespans(std::queue<ObsConeInfo>& cone_history);
    double find_closest_distance_in_cone_history(std::queue<ObsConeInfo> &cone_history, geometry_msgs::msg::Vector3 lidar_point, double yaw);
};