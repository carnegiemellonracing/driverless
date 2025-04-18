#include "cone_history_test_node.hpp"


std::pair<double, double> global_frame_to_local_frame(
    std::pair<double, double> global_frame_change,
    double yaw)
{
    double global_frame_dx = global_frame_change.first;
    double global_frame_dy = global_frame_change.second;

    double cmr_y = global_frame_dx * std::cos(yaw * M_PI / 180.0) + global_frame_dy * std::sin(yaw * M_PI / 180.0);
    double cmr_x = global_frame_dx * std::sin(yaw * M_PI / 180.0) - global_frame_dy * std::cos(yaw * M_PI / 180.0);

    return std::make_pair(cmr_x, cmr_y);
}

ConeHistoryTestNode::ConeHistoryTestNode() : Node("cone_history_test_node") 
{
    // Initialize the velocity and yaw deques
    velocity_deque = {};
    yaw_deque = {};

    // Initialize prev_time_stamp
    prev_time_stamp = -1;

    // Initialize subscribers

    // Subscriber that reads the input topic that contains an array of cone_point arrays from LiDAR stack
    auto cone_callback_group_ = create_callback_group(rclcpp::CallbackGroupType::MutuallyExclusive);
    rclcpp::SubscriptionOptions cone_options;
    cone_options.callback_group = cone_callback_group_;
    perc_cones_sub_ = create_subscription<interfaces::msg::ConeArray>(
        "/perc_cones",
        10,
        [this](const interfaces::msg::ConeArray::SharedPtr msg)
        { cone_callback(msg); },
        cone_options);

    // auto velocity_callback_group_ = create_callback_group(rclcpp::CallbackGroupType::MutuallyExclusive);
    rclcpp::SubscriptionOptions velocity_options;
    velocity_options.callback_group = cone_callback_group_;
    velocity_sub_ = create_subscription<geometry_msgs::msg::TwistStamped>(
        "/filter/twist",
        10,
        [this](const geometry_msgs::msg::TwistStamped::SharedPtr msg)
        { velocity_callback(msg); },
        velocity_options);

    // auto yaw_callback_group_ = create_callback_group(rclcpp::CallbackGroupType::MutuallyExclusive);
    rclcpp::SubscriptionOptions yaw_options;
    yaw_options.callback_group = cone_callback_group_;
    yaw_sub_ = create_subscription<geometry_msgs::msg::Vector3Stamped>(
        "/filter/euler",
        10,
        [this](const geometry_msgs::msg::Vector3Stamped::SharedPtr msg)
        { yaw_callback(msg); },
        yaw_options);

    // Initialize publishers
    associated_cones_pub_ = create_publisher<interfaces::msg::ConeArray>(
        "/associated_cones", 
        10
    );
}

/**
    * @brief Get the velocity and yaw of the car at a specific time
    * 
    * @param frameTime The time to get the velocity and yaw of the car
    * @return std::pair<geometry_msgs::msg::TwistStamped::SharedPtr, geometry_msgs::msg::Vector3Stamped::SharedPtr> The velocity and yaw of the car at the specified time
**/
std::pair<geometry_msgs::msg::TwistStamped::SharedPtr, geometry_msgs::msg::Vector3Stamped::SharedPtr> ConeHistoryTestNode::get_velocity_yaw(uint64_t frameTime) {
    geometry_msgs::msg::TwistStamped::SharedPtr closest_velocity_msg;
    geometry_msgs::msg::Vector3Stamped::SharedPtr closest_yaw_msg;

    geometry_msgs::msg::Vector3Stamped::SharedPtr yaw_msg;
    geometry_msgs::msg::TwistStamped::SharedPtr velocity_msg;

    // Check if deque empty
    // yaw_mutex.lock();
    if (yaw_deque.empty())
    {
        RCLCPP_INFO(get_logger(), "Yaw deque is empty! Cannot find matching yaw.");
        return std::make_pair(nullptr, nullptr);
    }

    yaw_msg = yaw_deque.back();
    // Iterate through deque to find the closest frame by timestamp
    for (const auto &yaw : yaw_deque)
    {   
        uint64_t yaw_time_ns = yaw->header.stamp.sec * 1e9 + yaw->header.stamp.nanosec;

        if (yaw_time_ns >= frameTime)
        {
            yaw_msg = yaw;
        }
    }
    // yaw_mutex.unlock();

    // Check if deque empty
    // velocity_mutex.lock();
    if (velocity_deque.empty())
    {
        RCLCPP_INFO(get_logger(), "Velocity deque is empty! Cannot find matching velocity.");
        return std::make_pair(nullptr, nullptr);
    }

    velocity_msg = velocity_deque.back();
    // Iterate through deque to find the closest frame by timestamp
    for (const auto &velocity : velocity_deque)
    {
        uint64_t velocity_time_ns = velocity->header.stamp.sec * 1e9 + velocity->header.stamp.nanosec;
        if (velocity_time_ns >= frameTime)
        {
            velocity_msg = velocity;
        }
    }
    // velocity_mutex.unlock();

    // Return closest velocity, yaw pair if found
    if (yaw_msg != NULL && velocity_msg != NULL)
    {
        return std::make_pair(velocity_msg, yaw_msg);
    }

    RCLCPP_INFO(get_logger(), "Callback time out of range! Cannot find matching velocity or yaw.");
    return std::make_pair(nullptr, nullptr);
}

/**
 * @brief Store the velocity of the car in queue
 * 
 * @param msg The current velocity of the car
 */
void ConeHistoryTestNode::velocity_callback(geometry_msgs::msg::TwistStamped::SharedPtr msg)
{
    // Deque Management and Updating
    // velocity_mutex.lock();
    while (velocity_deque.size() >= max_deque_size)
    {
        velocity_deque.pop_front();
    }
    velocity_deque.push_back(msg);
    // velocity_mutex.unlock();
}

/** 
    @brief Stores the yaw of the car in queue
    @param msg The current yaw of the car
**/
void ConeHistoryTestNode::yaw_callback(geometry_msgs::msg::Vector3Stamped::SharedPtr msg)
{
    // Deque Management and Updating
    // yaw_mutex.lock();
    while (yaw_deque.size() >= max_deque_size)
    {
        yaw_deque.pop_front();
    }
    yaw_deque.push_back(msg);
    // yaw_mutex.unlock();
}

float ConeHistoryTestNode::find_closest_distance_in_cone_history(std::queue<ObsConeInfo> &cone_history, geometry_msgs::msg::Vector3 lidar_point)
{
    float min_dist = std::numeric_limits<float>::max();
    for (int i = 0; i < cone_history.size(); i++)
    {
        ObsConeInfo cone_info = cone_history.front();
        cone_history.pop();
        float dx = (cone_info.cur_car_to_observer_position_x + cone_info.observer_position_to_cone_x) - lidar_point.x;
        float dy = (cone_info.cur_car_to_observer_position_y + cone_info.observer_position_to_cone_y) - lidar_point.y;
        
        float dist = sqrt(pow(dx, 2) + pow(dy, 2));
        if (dist < min_dist)
        {
            min_dist = dist;
        }
        cone_history.push(cone_info);
    }
    return min_dist;
}

int ConeHistoryTestNode::classify_through_data_association(geometry_msgs::msg::Vector3 lidar_point) {
    // Find the closest point wrt yellow points
    float min_dist_from_yellow = find_closest_distance_in_cone_history(yellow_cone_history, lidar_point);

    // Find the closest point wrt blue points
    float min_dist_from_blue = find_closest_distance_in_cone_history(blue_cone_history, lidar_point);

    // Between the 2 colors, determine which is closer
    if (min_dist_from_blue < min_dist_from_yellow && min_dist_from_blue < min_dist_th) { // most like a blue cone 
        return 2;
    } else if (min_dist_from_yellow < min_dist_from_blue && min_dist_from_yellow < min_dist_th) {
        return 1;
    } else {
        min_dist_from_yellow = find_closest_distance_in_cone_history(long_term_yellow_cone_history, lidar_point);
        min_dist_from_blue = find_closest_distance_in_cone_history(long_term_blue_cone_history, lidar_point);
        if (min_dist_from_blue < min_dist_from_yellow && min_dist_from_blue < min_dist_th) { // most like a blue cone 
            return 2;
        } else if (min_dist_from_yellow < min_dist_from_blue && min_dist_from_yellow < min_dist_th) {
            return 1;
        } else {
            RCLCPP_INFO(get_logger(), "No classification possible: min_dist_from_blue: %f | min_dist_from_yellow: %f", min_dist_from_blue, min_dist_from_yellow);
            return -1;
        }

    }
}

void ConeHistoryTestNode::motion_model_on_cone_history(std::queue<ObsConeInfo>& cone_history, std::pair<double, double> long_lat_change) {
    /**
     * Note: In the case that you plan to multithread the code in the future, 
     * this expression is moved out because cone_history's size changes 
     * during each iteration.
     */
    int cone_history_size = cone_history.size();

    for (int i = 0; i < cone_history_size; i++) {
        ObsConeInfo cone_info = cone_history.front();
        cone_info.cur_car_to_observer_position_y -= long_lat_change.first;
        cone_info.cur_car_to_observer_position_x -= long_lat_change.second;
        cone_history.pop();
        cone_history.push(cone_info);
    }
}

// void ConeHistoryTestNode::add_lidar_point_to_cone_history(std::queue<ObsConeInfo>& cone_history, geometry_msgs::msg::Vector3 lidar_point) {
//     ObsConeInfo cone_info = {
//         .cur_car_to_observer_position_x = lidar_point.x,
//         .cur_car_to_observer_position_y = lidar_point.y,
//         .observer_position_to_cone_x = 0.0f,
//         .observer_position_to_cone_y = 0.0f,
//         .lifespan = 0
//     };
//     cone_history.push(cone_info);
// }

void ConeHistoryTestNode::maintain_cone_history_lifespans(std::queue<ObsConeInfo>& cone_history) {
    int cone_history_size = cone_history.size();

    for (int i = 0; i < cone_history_size; i++) {
        ObsConeInfo cone_info = cone_history.front();
        cone_history.pop();
        assert(cone_info.lifespan <= max_timesteps_in_cone_history);
        if (cone_info.lifespan == max_timesteps_in_cone_history) {
            continue;
        } else {
            cone_info.lifespan++;
            cone_history.push(cone_info);
        }
    }
}
/**
 * @brief Store cones that are classified blue and yellow into our cone histories
 * For unknown cones, these will be what we are classifying 
 * This node will also publish the classified cones
 * 
 * @param msg 
 */
void ConeHistoryTestNode::cone_callback(interfaces::msg::ConeArray::SharedPtr msg) 
{
    uint64_t cur_time_stamp = msg->header.stamp.sec * 1e9 + msg->header.stamp.nanosec;
    if (prev_time_stamp == -1) {
        prev_time_stamp = cur_time_stamp;
    }

    double time_diff_seconds = (cur_time_stamp - prev_time_stamp) / 1e9;

    std::pair<geometry_msgs::msg::TwistStamped::SharedPtr, geometry_msgs::msg::Vector3Stamped::SharedPtr> cur_velocity_yaw = get_velocity_yaw(msg->header.stamp.sec * 1e9 + msg->header.stamp.nanosec);

    if (cur_velocity_yaw.first == nullptr || cur_velocity_yaw.second == nullptr) {
        RCLCPP_INFO(get_logger(), "Velocity or Yaw deque is empty! Cannot find matching velocity.");
        return;
    }

    RCLCPP_INFO(get_logger(), "-------------Motion Modeling--------------");
    RCLCPP_INFO(get_logger(), "\tTime diff: %f", time_diff_seconds);
    RCLCPP_INFO(get_logger(), "\tVelocity: %f, %f", cur_velocity_yaw.first->twist.linear.x, cur_velocity_yaw.first->twist.linear.y);
    RCLCPP_INFO(get_logger(), "\tYaw: %f", cur_velocity_yaw.second->vector.z);
    RCLCPP_INFO(get_logger(), "-------------End Motion Modeling--------------\n");

    double cur_velocity_x = cur_velocity_yaw.first->twist.linear.x;
    double cur_velocity_y = cur_velocity_yaw.first->twist.linear.y;
    double cur_yaw = cur_velocity_yaw.second->vector.z;
    
    double global_change_dx = cur_velocity_x * time_diff_seconds;
    double global_change_dy = cur_velocity_y * time_diff_seconds;


    RCLCPP_INFO(get_logger(), "-------------Processing Cones--------------"); 
    RCLCPP_INFO(get_logger(), "\tNumber of blue cones received: %zu", msg->blue_cones.size());
    RCLCPP_INFO(get_logger(), "\tNumber of yellow cones received: %zu", msg->yellow_cones.size());
    RCLCPP_INFO(get_logger(), "\tNumber of old blue cones: %zu", long_term_blue_cone_history.size());
    RCLCPP_INFO(get_logger(), "\tNumber of old yellow cones: %zu", long_term_yellow_cone_history.size());
    std::vector<geometry_msgs::msg::Point> blue_cones_to_publish = msg->blue_cones;
    std::vector<geometry_msgs::msg::Point> yellow_cones_to_publish = msg->yellow_cones;
    

    std::pair<double, double> long_lat_l = global_frame_to_local_frame(std::make_pair(global_change_dx, global_change_dy), cur_yaw);


    for (int i= 0; i < msg->blue_cones.size(); i++) {
        blue_cone_history.emplace(0.0f, 0.0f, msg->blue_cones[i].x, msg->blue_cones[i].y, 0);
        geometry_msgs::msg::Vector3 point;
        point.x = msg->blue_cones[i].x;
        point.y = msg->blue_cones[i].y;
        point.z = 0;
        float min_dist = find_closest_distance_in_cone_history(blue_cone_history, point);
        if (min_dist < min_dist_th) {
            long_term_blue_cone_history.emplace(0.0f, 0.0f, msg->blue_cones[i].x, msg->blue_cones[i].y, 0);
        }
    }
    
    for (int i= 0; i < msg->yellow_cones.size(); i++) {
        yellow_cone_history.emplace(0.0f, 0.0f, msg->yellow_cones[i].x, msg->yellow_cones[i].y, 0);
        geometry_msgs::msg::Vector3 point;
        point.x = msg->yellow_cones[i].x;
        point.y = msg->yellow_cones[i].y;
        point.z = 0;
        float min_dist = find_closest_distance_in_cone_history(yellow_cone_history, point);
        if (min_dist < min_dist_th) {
            long_term_yellow_cone_history.emplace(0.0f, 0.0f, msg->yellow_cones[i].x, msg->yellow_cones[i].y, 0);
        }
    }
    RCLCPP_INFO(get_logger(), "-------------End Processing Cones--------------\n");

    // Classify each unknown cone.
    RCLCPP_INFO(get_logger(), "-------------Start Motion Modeling On Cones--------------\n");
    motion_model_on_cone_history(blue_cone_history, long_lat_l);
    motion_model_on_cone_history(yellow_cone_history, long_lat_l);
    RCLCPP_INFO(get_logger(), "-------------End Motion Modeling On Cones--------------\n");

    int num_unable_to_classify_cones = 0;
    RCLCPP_INFO(this->get_logger(), "Num blue_cones_in_history: %d", blue_cone_history.size());
    RCLCPP_INFO(this->get_logger(), "Num yellow_cones_in_history: %d", yellow_cone_history.size());
    for (int i = 0; i < msg->unknown_color_cones.size(); i++) {
        RCLCPP_INFO(get_logger(), "-------------Classifying Cone--------------\n");
        geometry_msgs::msg::Vector3 lidar_point;
        lidar_point.x = msg->unknown_color_cones[i].x;
        lidar_point.y = msg->unknown_color_cones[i].y;
        lidar_point.z = msg->unknown_color_cones[i].z;
        RCLCPP_INFO(get_logger(), "\t Point: (%f, %f, %f)", lidar_point.x, lidar_point.y, lidar_point.z);
        int cone_class = classify_through_data_association(lidar_point);

        // Classify and add the newly classified cone to the cone history
        switch (cone_class) {
            //yellow
            case 1:
                RCLCPP_INFO(this->get_logger(), "\tClassified cone @ (%f, %f) as yellow", msg->unknown_color_cones[i].x, msg->unknown_color_cones[i].y);
                yellow_cones_to_publish.push_back(msg->unknown_color_cones[i]);
                break;
            //blue 
            case 2:
                RCLCPP_INFO(this->get_logger(), "\tClassified cone @ (%f, %f) as blue", msg->unknown_color_cones[i].x, msg->unknown_color_cones[i].y);
                blue_cones_to_publish.push_back(msg->unknown_color_cones[i]);
                break;
            default: 
                num_unable_to_classify_cones++;
                RCLCPP_INFO(this->get_logger(), "\tUnable to classify cone @ (%f, %f)", 
                                                msg->unknown_color_cones[i].x, 
                                                msg->unknown_color_cones[i].y);
                break;    
        }
    }

    RCLCPP_INFO(this->get_logger(), "Num unclassified cones: %d", num_unable_to_classify_cones);

    // Create the associated cones message
    interfaces::msg::ConeArray associated_cones_msg_;
    associated_cones_msg_.header.stamp = msg->header.stamp;
    associated_cones_msg_.header.frame_id = msg->header.frame_id;
    associated_cones_msg_.blue_cones = blue_cones_to_publish;
    associated_cones_msg_.yellow_cones = yellow_cones_to_publish;

    // Publish the associated cones
    associated_cones_pub_->publish(associated_cones_msg_);

    // Update the previous time stamp
    prev_time_stamp = cur_time_stamp;

    // Update the cone histories
    maintain_cone_history_lifespans(blue_cone_history);
    maintain_cone_history_lifespans(yellow_cone_history);
    
}

int main(int argc, char **argv)
{
    rclcpp::init(argc, argv);
    
    rclcpp::executors::MultiThreadedExecutor executor;
    rclcpp::Node::SharedPtr node = std::make_shared<ConeHistoryTestNode>();
    executor.add_node(node);
    executor.spin();
    
    rclcpp::shutdown();
    return 0;
}