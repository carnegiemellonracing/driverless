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

/** 
 * @brief Used for determining the position of a cone in global frame given local frame information
 */
std::pair<double, double> local_to_global_frame(
    std::pair<double, double> cmr_x_cmr_y_change, 
    double yaw
)
{
    double cmr_x = cmr_x_cmr_y_change.first;
    double cmr_y = cmr_x_cmr_y_change.second;

    double global_x = cmr_y * std::cos(yaw * M_PI / 180.0) + cmr_x * std::sin(yaw * M_PI / 180.0);
    double global_y = cmr_y * std::sin(yaw * M_PI / 180.0) - cmr_x * std::cos(yaw * M_PI / 180.0);

    return std::make_pair(global_x, global_y);
}

ConeHistoryTestNode::ConeHistoryTestNode() : Node("cone_history_test_node") 
{
    // Initialize the velocity and yaw deques
    velocity_deque = {};
    yaw_deque = {};

    // Initialize prev_time_stamp
    prev_time_stamp = -1;

    // Initialize subscribers
    blue_cone_history = {};
    yellow_cone_history = {};
    long_term_yellow_cone_history = {};
    long_term_blue_cone_history = {};

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
        "/associated_perc_cones", 
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
    yaw_mutex.lock();
    if (yaw_deque.empty())
    {
        yaw_mutex.unlock();
        RCLCPP_INFO(get_logger(), "Yaw deque is empty! Cannot find matching yaw.");
        return std::make_pair(nullptr, nullptr);
    }

    yaw_msg = yaw_deque.back();
    // Iterate through deque to find the closest frame by timestamp
    for (size_t i = 0; i < yaw_deque.size(); i++)
    {   
        uint64_t yaw_time_ns = yaw_deque[i]->header.stamp.sec * 1e9 + yaw_deque[i]->header.stamp.nanosec;

        if (yaw_time_ns >= frameTime)
        {
            yaw_msg = yaw_deque[i];
            if (i > 0)
            {
                auto prev_yaw_ts = yaw_deque[i-1]->header.stamp.sec * 1e9 + yaw_deque[i-1]->header.stamp.nanosec;
                auto yaw = yaw_deque[i-1]->vector.z + (yaw_msg->vector.z - yaw_deque[i-1]->vector.z) * (frameTime - prev_yaw_ts) / (yaw_time_ns - prev_yaw_ts);
                yaw_msg = std::make_shared<geometry_msgs::msg::Vector3Stamped>();
                yaw_msg->vector.z = yaw;
            }
            break;
        }
    }
    yaw_mutex.unlock();

    // Check if deque empty
    velocity_mutex.lock();
    if (velocity_deque.empty())
    {
        velocity_mutex.unlock();
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
            break;
        }
    }
    velocity_mutex.unlock();

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
    velocity_mutex.lock();
    while (velocity_deque.size() >= max_deque_size)
    {
        velocity_deque.pop_front();
    }
    velocity_deque.push_back(msg);
    velocity_mutex.unlock();
}

/** 
    @brief Stores the yaw of the car in queue
    @param msg The current yaw of the car
**/
void ConeHistoryTestNode::yaw_callback(geometry_msgs::msg::Vector3Stamped::SharedPtr msg)
{
    // Deque Management and Updating
    yaw_mutex.lock();
    while (yaw_deque.size() >= max_deque_size)
    {
        yaw_deque.pop_front();
    }
    yaw_deque.push_back(msg);
    yaw_mutex.unlock();
}

/**
 * @brief 
 * 
 * @param cone_history 
 * @param lidar_point This is a point in CMR/local car frame. This needs to be transformed into global frame for comparison
 * @return float 
 */
double ConeHistoryTestNode::find_closest_distance_in_cone_history(std::queue<ObsConeInfo> &cone_history, geometry_msgs::msg::Vector3 lidar_point, double yaw)
{
    double cur_min_dist = std::numeric_limits<double>::max();
    // RCLCPP_INFO(get_logger(), "Initilalized min_dist: %lf | Cone history size: %d", cur_min_dist, cone_history.size());
    for (int i = 0; i < cone_history.size(); i++)
    {
        ObsConeInfo cone_info = cone_history.front();
        cone_history.pop();

        // Transforming the lidar point into global_frame
        std::pair<double, double> local_lidar_point_xy = std::make_pair(lidar_point.x, lidar_point.y);
        std::pair<double, double> global_car_to_lidar = local_to_global_frame(local_lidar_point_xy, yaw);
        double global_car_to_lidar_x = global_car_to_lidar.first;
        double global_car_to_lidar_y = global_car_to_lidar.second;

        // Transforming the observer_position to lidar point information into global_frame
        std::pair<double, double> local_observer_to_cone_xy = std::make_pair(cone_info.observer_position_to_cone_x, cone_info.observer_position_to_cone_y);
        std::pair<double, double> global_observer_to_cone = local_to_global_frame(local_observer_to_cone_xy, cone_info.observer_yaw);
        double global_observer_to_cone_x = global_observer_to_cone.first;
        double global_observer_to_cone_y = global_observer_to_cone.second;

        double dx = (cone_info.cur_car_to_observer_x + global_observer_to_cone_x) - global_car_to_lidar_x;
        double dy = (cone_info.cur_car_to_observer_y + global_observer_to_cone_y) - global_car_to_lidar_y;
        
        double dist = (double)sqrt(pow(dx, 2) + pow(dy, 2));
        if (dist < cur_min_dist)
        {
            cur_min_dist = dist;
        }
        cone_history.push(cone_info);
    }
    // RCLCPP_INFO(get_logger(), "Final min_dist: %lf", cur_min_dist);
    return cur_min_dist;
}

int ConeHistoryTestNode::classify_through_data_association(geometry_msgs::msg::Vector3 lidar_point, double yaw) {
    // Find the closest point wrt yellow points
    double min_dist_from_yellow = find_closest_distance_in_cone_history(yellow_cone_history, lidar_point, yaw);

    // Find the closest point wrt blue points
    double min_dist_from_blue = find_closest_distance_in_cone_history(blue_cone_history, lidar_point, yaw);

    // Between the 2 colors, determine which is closer
    if (min_dist_from_blue < min_dist_from_yellow && min_dist_from_blue < min_dist_th) { // most like a blue cone 
        return 2;
    } else if (min_dist_from_yellow < min_dist_from_blue && min_dist_from_yellow < min_dist_th) {
        return 1;
    } else {
        min_dist_from_yellow = find_closest_distance_in_cone_history(long_term_yellow_cone_history, lidar_point, yaw);
        min_dist_from_blue = find_closest_distance_in_cone_history(long_term_blue_cone_history, lidar_point, yaw);
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

/**
 * @brief This function will treat the current car position as world origin, and motion model the other
 * observer positions to be with respect to the current observer position/cur car position (in 
 * globla frame).
 * The reason why we do this is because the further away from 0, 0 you are, the 
 * less accurate you become. We would want our current position to be 0, 0 and have other positions
 * build off from here. 
 * 
 * @param cone_history 
 * @param global_xy_change 
 */
void ConeHistoryTestNode::motion_model_on_cone_history(std::queue<ObsConeInfo>& cone_history, std::pair<double, double> global_xy_change) {
    /**
     * Note: In the case that you plan to multithread the code in the future, 
     * this expression is moved out because cone_history's size changes 
     * during each iteration.
     */
    int cone_history_size = cone_history.size();

    for (int i = 0; i < cone_history_size; i++) {
        ObsConeInfo cone_info = cone_history.front();
        cone_info.cur_car_to_observer_x += global_xy_change.first;
        cone_info.cur_car_to_observer_y += global_xy_change.second;
        cone_history.pop();
        cone_history.push(cone_info);
    }
}

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
    std::pair<double, double> global_xy_change = std::make_pair(global_change_dx, global_change_dy);

    RCLCPP_INFO(get_logger(), "-------------Start Motion Modeling On Cone History--------------\n");
    motion_model_on_cone_history(blue_cone_history, global_xy_change);
    motion_model_on_cone_history(yellow_cone_history, global_xy_change);
    motion_model_on_cone_history(long_term_blue_cone_history, global_xy_change);
    motion_model_on_cone_history(long_term_yellow_cone_history, global_xy_change);

    RCLCPP_INFO(get_logger(), "-------------End Motion Modeling On Cone History--------------\n");

    RCLCPP_INFO(get_logger(), "-------------Processing Cones--------------"); 
    RCLCPP_INFO(get_logger(), "\tNumber of blue cones received: %zu", msg->blue_cones.size());
    RCLCPP_INFO(get_logger(), "\tNumber of yellow cones received: %zu", msg->yellow_cones.size());
    RCLCPP_INFO(get_logger(), "\tNumber of old blue cones: %zu", long_term_blue_cone_history.size());
    RCLCPP_INFO(get_logger(), "\tNumber of old yellow cones: %zu", long_term_yellow_cone_history.size());
    
    cones::TrackBounds cones_to_publish;
    
    for (int i = 0; i < msg->blue_cones.size(); i++) {
        cones::Cone cone(msg->blue_cones[i]);
        cones_to_publish.blue.push_back(cone);
        
        blue_cone_history.emplace(0.0, 0.0, cur_yaw, msg->blue_cones[i].x, msg->blue_cones[i].y, 0);
        geometry_msgs::msg::Vector3 point;
        point.x = msg->blue_cones[i].x;
        point.y = msg->blue_cones[i].y;
        point.z = 0;
        // Check if you have an old or new cone
        double min_dist = find_closest_distance_in_cone_history(long_term_blue_cone_history, point, cur_yaw);
        if (min_dist > min_dist_th) { // Greater than the threshold means that its far away from everything enough, we believe it's new
            long_term_blue_cone_history.emplace(0.0, 0.0, cur_yaw, msg->blue_cones[i].x, msg->blue_cones[i].y, 0);
        }
    }
    
    while (long_term_blue_cone_history.size() > max_long_term_history_size) {
        long_term_blue_cone_history.pop();
    }

    while (long_term_yellow_cone_history.size() > max_long_term_history_size) {
        long_term_yellow_cone_history.pop();
    }

    // Fix for type difference: Convert geometry_msgs::msg::Point to cones::Cone
    for (int i = 0; i < msg->yellow_cones.size(); i++) {
        cones::Cone cone(msg->yellow_cones[i]);
        cones_to_publish.yellow.push_back(cone);
        
        yellow_cone_history.emplace(0.0, 0.0, cur_yaw, msg->yellow_cones[i].x, msg->yellow_cones[i].y, 0);
        geometry_msgs::msg::Vector3 point;
        point.x = msg->yellow_cones[i].x;
        point.y = msg->yellow_cones[i].y;
        point.z = 0;
        // Check if you have an old or new cone
        double min_dist = find_closest_distance_in_cone_history(long_term_yellow_cone_history, point, cur_yaw);
        if (min_dist > min_dist_th) { // Greater than the threshold means that its far away from everything enough, we believe it's new
            long_term_yellow_cone_history.emplace(0.0, 0.0, cur_yaw, msg->yellow_cones[i].x, msg->yellow_cones[i].y, 0);
        }
    }
    RCLCPP_INFO(get_logger(), "-------------End Processing Cones--------------\n");

    // Classify each unknown cone.
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
        int cone_class = classify_through_data_association(lidar_point, cur_yaw);

        // Classify and add the newly classified cone to the cone history
        switch (cone_class) {
            //yellow
            case 1: {
                RCLCPP_INFO(this->get_logger(), "\tClassified cone @ (%f, %f) as yellow", 
                           msg->unknown_color_cones[i].x, msg->unknown_color_cones[i].y);
                cones::Cone cone(msg->unknown_color_cones[i]);
                cones_to_publish.yellow.push_back(cone);
                break;
            }
            //blue 
            case 2: {
                RCLCPP_INFO(this->get_logger(), "\tClassified cone @ (%f, %f) as blue", 
                           msg->unknown_color_cones[i].x, msg->unknown_color_cones[i].y);
                cones::Cone cone(msg->unknown_color_cones[i]);
                cones_to_publish.blue.push_back(cone);
                break;
            }
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

    cones::TrackBounds recoloured_cones_to_publish = cones::recolouring::recolour_cones(cones_to_publish, 10.0);

    if (!recoloured_cones_to_publish.yellow.empty()) {
        for (const auto& cone : cones::order_cones(recoloured_cones_to_publish.yellow)) {
            associated_cones_msg_.yellow_cones.push_back(cone.point);
        }
    }

    if (!recoloured_cones_to_publish.blue.empty()) {
        for (const auto& cone : cones::order_cones(recoloured_cones_to_publish.blue)) {
            associated_cones_msg_.blue_cones.push_back(cone.point);
        }
    }

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