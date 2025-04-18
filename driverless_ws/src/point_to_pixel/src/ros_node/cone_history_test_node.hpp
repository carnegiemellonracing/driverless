#include "rclcpp/rclcpp.hpp"

#include "geometry_msgs/msg/point.hpp"
#include "geometry_msgs/msg/vector3_stamped.hpp"
#include "geometry_msgs/msg/twist_stamped.hpp"

// Standard Imports
#include <deque>
#include <memory>
#include <chrono>
#include <filesystem>

struct ObsConeInfo {
    double cur_car_to_observer_position_x;
    double cur_car_to_observer_position_y;

    double observer_position_to_cone_x;
    double observer_position_to_cone_y;
    int lifespan;

    ObsConeInfo (
        double car_to_observer_x, 
        double car_to_observer_y, 
        double observer_to_cone_x,
        double observer_to_cone_y,
        double ls
    ) : cur_car_to_observer_position_x(car_to_observer_x),
      cur_car_to_observer_position_y(car_to_observer_y),
      observer_position_to_cone_x(observer_to_cone_x),
      observer_position_to_cone_y(observer_to_cone_y) 
      lifespan(ls) {}

};
class ConeHistoryTestNode : public rclcpp::Node {
public:
    ConeHistoryTestNode();

private:
    // Message queues 
    std::deque<geometry_msgs::msg::TwistStamped::SharedPtr> velocity_deque;
    std::deque<geometry_msgs::msg::Vector3Stamped::SharedPtr> yaw_deque;

    std::queue<ObsConeInfo> yellow_cone_history;
    std::queue<ObsConeInfo> blue_cone_history;

    uint64_t prev_time_stamp;

    std::pair<geometry_msgs::msg::TwistStamped::SharedPtr, geometry_msgs::msg::Vector3Stamped::SharedPtr> PointToPixelNode::get_velocity_yaw(
        uint64_t frameTime);

    // Motion Modeling Callbacks
    void cone_callback(const interfaces::msg::ConeArray::SharedPtr cone_msg);
    void velocity_callback(const geometry_msgs::msg::TwistStamped::SharedPtr msg);
    void yaw_callback(const geometry_msgs::msg::Vector3Stamped::SharedPtr msg);

    // Define subscribers and publishers
    rclcpp::Subscription<interfaces::msg::ConeArray>::SharedPtr perc_cones_sub_;
    rclcpp::Subscription<geometry_msgs::msg::TwistStamped>::SharedPtr velocity_sub_;
    rclcpp::Subscription<geometry_msgs::msg::Vector3Stamped>::SharedPtr yaw_sub_;

    rclcpp::Publisher<interfaces::msg::ConeArray>::SharedPtr associated_cones_pub_;

    void classify_through_data_association(geometry_msgs::msg::Vector3 lidar_point);
    void motion_model_on_cone_history(std::queue<ObsConeInfo>& cone_history, std::pair<double, double> long_lat_change);
    // void add_lidar_point_to_cone_history(std::queue<ObsConeInfo>& cone_history, geometry_msgs::msg::Vector3 lidar_point);
    void maintain_cone_history_lifespans(std::queue<ObsConeInfo>& cone_history);

};