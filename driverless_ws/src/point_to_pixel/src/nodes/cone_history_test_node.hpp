#include "rclcpp/rclcpp.hpp"

#include "geometry_msgs/msg/point.hpp"
#include "geometry_msgs/msg/vector3_stamped.hpp"
#include "geometry_msgs/msg/twist_stamped.hpp"

#include "interfaces/msg/cone_array.hpp"

#include "../cones/svm/svm_recolour.hpp"


// Standard Imports
#include <deque>
#include <queue>
#include <memory>
#include <chrono>
#include <filesystem>
#include <atomic>

/**
 * @brief We motion model on the position of the car to the observer, instead
 * of thinking in global frame wrt to the start. This is because we want the 
 * current position to be the most accurate, and other positions to be less accurate.
 * 
 */

 /**
  * what do you need to be able to do
  * - update an old cone with some new cone that data associates to it 
  * - much easier to just track the global position with respect to the start position
  * - track a counter for how many times the cone was seen with a certain color
  * - - then vote on the most commonly seen color 
  * 
  * 
  * obsconeinfo should track the calculated global cone position 
  * everytime we data associate a cone to it we will update the global position with the more recent one 
  * 
  * 1.) make a change to obsconeinfo to just store 
  * - global cone position normalized to the start pose
  * - store a lifespan for the current history 
  * 
  * 2.) 
  */
struct ObsConeInfo {
    double global_cone_x;
    double global_cone_y;

    int times_seen_blue;
    int times_seen_yellow;
    int id; //Track the ID in case we want a smaller short term history in the future with smaller latency

    ObsConeInfo (
        double input_global_x,
        double input_global_y,
        int input_id
    ) : global_cone_x(input_global_x),
        global_cone_y(input_global_y), 
        times_seen_blue(0),
        times_seen_yellow(0),
        id(input_id)
        {}

};

class ConeHistoryTestNode : public rclcpp::Node {
public:
    ConeHistoryTestNode();

private:
    // Current position
    std::pair<double, double> cur_position;

    // Message queues 
    std::deque<geometry_msgs::msg::TwistStamped::SharedPtr> velocity_deque;
    std::deque<geometry_msgs::msg::Vector3Stamped::SharedPtr> yaw_deque;
    static constexpr int max_deque_size = 100;
    static constexpr int max_timesteps_in_cone_history = 10;
    static constexpr int max_long_term_history_size = 300;

    std::vector<ObsConeInfo> cone_history;
    double min_dist_th = 0.25;
    double max_order_dist = 5.0; // meters

    uint64_t prev_time_stamp;

    std::pair<geometry_msgs::msg::TwistStamped::SharedPtr, geometry_msgs::msg::Vector3Stamped::SharedPtr> get_velocity_yaw(uint64_t frameTime);

    // Motion Modeling Callbacks
    void cone_callback(const interfaces::msg::ConeArray::SharedPtr cone_msg);
    void republished_cone_callback(const interfaces::msg::ConeArray::SharedPtr republished_cone_msg);
    void velocity_callback(const geometry_msgs::msg::TwistStamped::SharedPtr msg);
    void yaw_callback(const geometry_msgs::msg::Vector3Stamped::SharedPtr msg);

    // Define subscribers and publishers
    rclcpp::Subscription<interfaces::msg::ConeArray>::SharedPtr perc_cones_sub_;
    rclcpp::Subscription<geometry_msgs::msg::TwistStamped>::SharedPtr velocity_sub_;
    rclcpp::Subscription<geometry_msgs::msg::Vector3Stamped>::SharedPtr yaw_sub_;
    rclcpp::Subscription<interfaces::msg::ConeArray>::SharedPtr republished_perc_cones_sub_;

    rclcpp::Publisher<interfaces::msg::ConeArray>::SharedPtr associated_cones_pub_;

    std::mutex velocity_mutex;
    std::mutex yaw_mutex;
    std::mutex cone_history_mutex;
    std::mutex controller_receive_time_mutex;
    builtin_interfaces::msg::Time controller_receive_time_;
    std::atomic<bool> controller_receive_time_ready_;

    std::pair<int, double> find_closest_cone_id(std::pair<double, double> global_cone_position);
    std::pair<double, double> lidar_point_to_global_cone_position(geometry_msgs::msg::Vector3 lidar_point, std::pair<double, double> cur_position, double yaw);
    int determine_color(int id);
    int classify_through_data_association(std::pair<double, double> global_cone_position);
    void maintain_cone_history_lifespans(std::queue<ObsConeInfo>& cone_history);
    double find_closest_distance_in_cone_history(std::queue<ObsConeInfo> &cone_history, geometry_msgs::msg::Vector3 lidar_point, double yaw);
    void update_cone_history_with_colored_cone(std::vector<geometry_msgs::msg::Point> cone_msg, std::pair<double, double> cur_position, double cur_yaw, int color);
};