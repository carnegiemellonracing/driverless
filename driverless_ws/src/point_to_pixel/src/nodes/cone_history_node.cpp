#include "cone_history_node.hpp"


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

ConeHistoryTestNode::ConeHistoryTestNode() : Node("cone_history_node") 
{
    // Initialize the velocity and yaw deques
    velocity_deque = {};
    yaw_deque = {};

    // Initialize prev_time_stamp
    prev_time_stamp = -1;

    // Initialize subscribers
    cone_history = {};

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
    velocity_options.callback_group = create_callback_group(rclcpp::CallbackGroupType::MutuallyExclusive);
    velocity_sub_ = create_subscription<geometry_msgs::msg::TwistStamped>(
        "/filter/twist",
        10,
        [this](const geometry_msgs::msg::TwistStamped::SharedPtr msg)
        { velocity_callback(msg); },
        velocity_options);

    // auto yaw_callback_group_ = create_callback_group(rclcpp::CallbackGroupType::MutuallyExclusive);
    rclcpp::SubscriptionOptions yaw_options;
    yaw_options.callback_group = create_callback_group(rclcpp::CallbackGroupType::MutuallyExclusive);
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
 * @brief Determine the color of the cone indicated by ID by 
 * which color it was seen the most with. 
 * 
 * @param cone_history 
 * @param id 
 */
int ConeHistoryTestNode::determine_color(int id) {
    ObsConeInfo cone_info = cone_history.at(id);

    if (cone_info.times_seen_blue > cone_info.times_seen_yellow) {
        return 2;
    } else if (cone_info.times_seen_blue < cone_info.times_seen_yellow) {
        return 1;
    } else {
        return -1; // ! determine what to do in this inconclusive case 
    }
}

std::pair<double, double> ConeHistoryTestNode::lidar_point_to_global_cone_position(geometry_msgs::msg::Vector3 lidar_point, std::pair<double, double> cur_position, double yaw) {
    std::pair<double, double> lidar_point_from_car_global = local_to_global_frame(std::make_pair(lidar_point.x, lidar_point.y), yaw);
    double global_lidar_point_x = cur_position.first + lidar_point_from_car_global.first;
    double global_lidar_point_y = cur_position.second + lidar_point_from_car_global.second;
    return std::make_pair(global_lidar_point_x, global_lidar_point_y);
}

std::pair<int, double> ConeHistoryTestNode::find_closest_cone_id(std::pair<double, double> global_cone_position)
{
    double min_dist = std::numeric_limits<double>::max();
    int min_id = -1;

    // cone_history_mutex.lock();
    for (int i = 0; i < cone_history.size(); i++)
    {
        ObsConeInfo cone_info = cone_history.at(i);
        double diff_x = cone_info.global_cone_x - global_cone_position.first;
        double diff_y = cone_info.global_cone_y - global_cone_position.second;
        double dist = sqrt(pow(diff_x, 2) + pow(diff_y, 2));

        if (dist < min_dist) {
            min_dist = dist;
            min_id = i;
        }
    }
    // cone_history_mutex.unlock();

    return std::make_pair(min_id, min_dist);

    
}


void ConeHistoryTestNode::update_cone_history_with_colored_cone(std::vector<geometry_msgs::msg::Point> cone_msg, std::pair<double, double> cur_position, double cur_yaw, int color) {
    for (int i= 0; i < cone_msg.size(); i++) {
        geometry_msgs::msg::Vector3 point;
        point.x = cone_msg[i].x;
        point.y = cone_msg[i].y;
        point.z = 0;

        std::pair<double, double> global_cone_position = lidar_point_to_global_cone_position(point, cur_position, cur_yaw);
        // !todo use a semaphore so that multiple threads can read
        // cone_history_mutex.lock();
        std::pair<int, double> nearest_cone = find_closest_cone_id(global_cone_position);
        // cone_history_mutex.unlock();

        if (nearest_cone.second > min_dist_th) { // Greater than the threshold means that its far away from everything enough, we believe it's new
            // cone_history_mutex.lock();
            cone_history.emplace_back(global_cone_position.first, global_cone_position.second, nearest_cone.first);
            // cone_history_mutex.unlock();
        } else { // Old cone
            int min_id = nearest_cone.first; 
            //Update with the current position
            // cone_history_mutex.lock();
            cone_history.at(min_id).global_cone_x = global_cone_position.first;
            cone_history.at(min_id).global_cone_y = global_cone_position.second;

            //Update the current counters
            if (color == 1) {// yellow
                cone_history.at(min_id).times_seen_yellow++;
            } else if (color == 2) {// blue
                cone_history.at(min_id).times_seen_blue++;
            }
            // cone_history_mutex.unlock();
        }
    }
}

int ConeHistoryTestNode::classify_through_data_association(std::pair<double, double> global_cone_position) {
    // cone_history_mutex.lock();
    std::pair<int, double> nearest_cone = find_closest_cone_id( global_cone_position);
    // cone_history_mutex.unlock();

    // cone_history_mutex.lock();
    int cone_color = determine_color(nearest_cone.first);
    // cone_history_mutex.unlock();
    return cone_color;
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

    double cur_velocity_x = cur_velocity_yaw.first->twist.linear.x;
    double cur_velocity_y = cur_velocity_yaw.first->twist.linear.y;
    double cur_yaw = cur_velocity_yaw.second->vector.z;
    
    double global_change_dx = cur_velocity_x * time_diff_seconds;
    double global_change_dy = cur_velocity_y * time_diff_seconds;
    std::pair<double, double> global_xy_change = std::make_pair(global_change_dx, global_change_dy);
    std::pair<double, double> new_position = std::make_pair(cur_position.first + global_xy_change.first, cur_position.second + global_xy_change.second);
    cur_position = new_position;

    RCLCPP_INFO(get_logger(), "-------------End Motion Modeling--------------\n");
    


    RCLCPP_INFO(get_logger(), "-------------Processing Cones--------------"); 
    RCLCPP_INFO(get_logger(), "\tNumber of blue cones received: %zu", msg->blue_cones.size());
    RCLCPP_INFO(get_logger(), "\tNumber of yellow cones received: %zu", msg->yellow_cones.size());
    // cone_history_mutex.lock();
    RCLCPP_INFO(get_logger(), "\tNumber of old cones: %zu", cone_history.size());
    // cone_history_mutex.unlock();
    std::vector<geometry_msgs::msg::Point> blue_cones_to_publish = msg->blue_cones;
    std::vector<geometry_msgs::msg::Point> yellow_cones_to_publish = msg->yellow_cones;
    

    update_cone_history_with_colored_cone(msg->blue_cones, cur_position, cur_yaw, 2);
    update_cone_history_with_colored_cone(msg->yellow_cones, cur_position, cur_yaw, 1);
    
    RCLCPP_INFO(get_logger(), "-------------End Processing Cones--------------\n");

    // Classify each unknown cone.


    int num_unable_to_classify_cones = 0;
    for (int i = 0; i < msg->unknown_color_cones.size(); i++) {
        RCLCPP_INFO(get_logger(), "-------------Classifying Cone--------------\n");
        geometry_msgs::msg::Vector3 lidar_point;
        lidar_point.x = msg->unknown_color_cones[i].x;
        lidar_point.y = msg->unknown_color_cones[i].y;
        lidar_point.z = msg->unknown_color_cones[i].z;
        std::pair<double, double> global_cone_position = lidar_point_to_global_cone_position(lidar_point, cur_position, cur_yaw);
        RCLCPP_INFO(get_logger(), "\t Global Point: (%f, %f)", global_cone_position.first, global_cone_position.second);
        RCLCPP_INFO(get_logger(), "\t Point: (%f, %f)", lidar_point.x, lidar_point.y);
        int cone_class = classify_through_data_association(global_cone_position);

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

    // cones::TrackBounds recolored_cones_to_publish = cones::recoloring::recolor_cones(cones_to_publish, 10.0);

    cones::Cones yellow_cones;
    cones::Cones blue_cones;
    for (const auto &cone : yellow_cones_to_publish) {
        yellow_cones.push_back(cones::Cone(cone));
    }
    for (const auto& cone : cones::order_cones(yellow_cones, max_order_dist)) {
        associated_cones_msg_.yellow_cones.push_back(cone.point);
    }
    for (const auto &cone : blue_cones_to_publish) {
        blue_cones.push_back(cones::Cone(cone));
    }
    for (const auto& cone : cones::order_cones(blue_cones, max_order_dist)) {
        associated_cones_msg_.blue_cones.push_back(cone.point);
    }

    // associated_cones_msg_.blue_cones = blue_cones_to_publish;
    // associated_cones_msg_.yellow_cones = yellow_cones_to_publish;

    // Publish the associated cones
    associated_cones_pub_->publish(associated_cones_msg_);

    // Update the previous time stamp
    prev_time_stamp = cur_time_stamp;

    
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