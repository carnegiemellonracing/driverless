#include "point_to_pixel_node.hpp"

// Standard Imports
#include <deque>
#include <memory>
#include <chrono>
#include <filesystem>

// Constructor definition
PointToPixelNode::PointToPixelNode() : Node("point_to_pixel"),
    params([]() {sl_oc::video::VideoParams p; p.res = sl_oc::video::RESOLUTION::HD1080; p.fps = sl_oc::video::FPS::FPS_30; return p;}()),
    cap_l(sl_oc::video::VideoCapture(params)),
    cap_r(sl_oc::video::VideoCapture(params))
{
    // ---------------------------------------------------------------------------
    //                              CAMERA INITIALIZATION
    // ---------------------------------------------------------------------------

    // Initialize cameras
    if (!initialize_camera(cap_l, 0, map_left_x_ll, map_left_y_ll, map_right_x_lr, map_right_y_lr, get_logger())) {
        rclcpp::shutdown(); // Shutdown node if camera initialization fails
        return;
    }
    
    if (!initialize_camera(cap_r, 2, map_left_x_rl, map_left_y_rl, map_right_x_rr, map_right_y_rr, get_logger())) {
        rclcpp::shutdown(); // Shutdown node if camera initialization fails
        return;
    }
    // ---------------------------------------------------------------------------
    //                               PARAMETERS
    // ---------------------------------------------------------------------------

    // Initialize Empty Image Deques
    img_deque_l = {};
    img_deque_r = {};

    // Initialize Empty Velocity/Yaw Deques
    velocity_deque = {};
    yaw_deque = {};

    // Projection matrix that takes LiDAR points to pixels
    std::vector<double> param_default(12, 1.0f); 

    declare_parameter("projection_matrix_ll", param_default);
    declare_parameter("projection_matrix_rl", param_default);
    declare_parameter("projection_matrix_lr", param_default);
    declare_parameter("projection_matrix_rr", param_default);

    // Threshold that determines whether it reports the color on a cone or not
    declare_parameter("confidence_threshold", 0.01);

    #if use_yolo
        // Load YOLO Model
        net = coloring::yolo::init_model("src/point_to_pixel/config/best164.onnx");
        if (net.empty()) {
            RCLCPP_ERROR(get_logger(), "Error Loading YOLO Model");
            rclcpp::shutdown();
        }
    #endif

    // Default Color Parameters
    std::vector<long int> ly_filter_default{0, 0, 0};
    std::vector<long int> uy_filter_default{0, 0, 0};
    std::vector<long int> lb_filter_default{0, 0, 0};
    std::vector<long int> ub_filter_default{255, 255, 255};
    std::vector<long int> lo_filter_default{255, 255, 255};
    std::vector<long int> uo_filter_default{255, 255, 255};

    // Color Parameters
    declare_parameter("yellow_filter_high", ly_filter_default);
    declare_parameter("yellow_filter_low", uy_filter_default);
    declare_parameter("blue_filter_high", lb_filter_default);
    declare_parameter("blue_filter_low", ub_filter_default);
    declare_parameter("orange_filter_high", lo_filter_default);
    declare_parameter("orange_filter_low", uo_filter_default);

    // Load Projection Matrix if inner is set to true, then load lr and rl, else load ll and rr
    std::vector<double> param_l, param_r;
    
    #if inner
        param_l = get_parameter("projection_matrix_lr").as_double_array();
        param_r = get_parameter("projection_matrix_rl").as_double_array();
    #else
        param_l = get_parameter("projection_matrix_ll").as_double_array();
        param_r = get_parameter("projection_matrix_rr").as_double_array();
    #endif
    
    projection_matrix_l = Eigen::Map<Eigen::Matrix<double, 3, 4, Eigen::RowMajor>>(param_l.data());
    projection_matrix_r = Eigen::Map<Eigen::Matrix<double, 3, 4, Eigen::RowMajor>>(param_r.data());

    #if verbose
    // Create a stringstream to log the matrix
    std::stringstream ss_l;
    std::stringstream ss_r;

    // Iterate over the rows and columns of the matrix and format the output
    for (int i = 0; i < projection_matrix_l.rows(); ++i){
        for (int j = 0; j < projection_matrix_l.cols(); ++j){
            ss_l << projection_matrix_l(i, j) << " ";
            ss_r << projection_matrix_r(i, j) << " ";
        }
        ss_l << "\n";
        ss_r << "\n";
    }
    // Log the projection_matrix using ROS 2 logger
    RCLCPP_INFO(get_logger(), "Projection Matrix Left:\n%s", ss_l.str().c_str());
    RCLCPP_INFO(get_logger(), "Projection Matrix Right:\n%s", ss_r.str().c_str());
    #endif

    // Load Confidence Threshold
    confidence_threshold = get_parameter("confidence_threshold").as_double();

    // Load Color Filter Params
    std::vector<long int> uy_filt_arr = get_parameter("yellow_filter_high").as_integer_array();
    std::vector<long int> ly_filt_arr = get_parameter("yellow_filter_low").as_integer_array();
    std::vector<long int> lb_filt_arr = get_parameter("blue_filter_low").as_integer_array();
    std::vector<long int> ub_filt_arr = get_parameter("blue_filter_high").as_integer_array();
    std::vector<long int> lo_filt_arr = get_parameter("orange_filter_low").as_integer_array();
    std::vector<long int> uo_filt_arr = get_parameter("orange_filter_high").as_integer_array();

    yellow_filter_high = cv::Scalar(uy_filt_arr[0], uy_filt_arr[1], uy_filt_arr[2]);
    yellow_filter_low = cv::Scalar(ly_filt_arr[0], ly_filt_arr[1], ly_filt_arr[2]);
    blue_filter_high = cv::Scalar(ub_filt_arr[0], ub_filt_arr[1], ub_filt_arr[2]);
    blue_filter_low = cv::Scalar(lb_filt_arr[0], lb_filt_arr[1], lb_filt_arr[2]);
    orange_filter_high = cv::Scalar(uo_filt_arr[0], uo_filt_arr[1], uo_filt_arr[2]);
    orange_filter_low = cv::Scalar(lo_filt_arr[0], lo_filt_arr[1], lo_filt_arr[2]);

    // ---------------------------------------------------------------------------
    //                              ROS2 OBJECTS
    // ---------------------------------------------------------------------------

    // Publisher that returns colored cones
    cone_pub_ = create_publisher<interfaces::msg::ConeArray>("/perc_cones", 10);
    
    // Subscriber that reads the input topic that contains an array of cone_point arrays from LiDAR stack
    auto cone_callback_group_ = create_callback_group(rclcpp::CallbackGroupType::MutuallyExclusive);
    rclcpp::SubscriptionOptions cone_options;
    cone_options.callback_group = cone_callback_group_;
    cone_sub_ = create_subscription<interfaces::msg::PPMConeArray>(
        "/cpp_cones", 
        10, 
        [this](const interfaces::msg::PPMConeArray::SharedPtr msg) {cone_callback(msg);},
        cone_options
    );

    // auto velocity_callback_group_ = create_callback_group(rclcpp::CallbackGroupType::MutuallyExclusive);
    rclcpp::SubscriptionOptions velocity_options;
    velocity_options.callback_group = cone_callback_group_;
    velocity_sub_ = create_subscription<geometry_msgs::msg::TwistStamped>(
        "/filter/twist", 
        10, 
        [this](const geometry_msgs::msg::TwistStamped::SharedPtr msg) {velocity_callback(msg);},
        velocity_options
    );

    // auto yaw_callback_group_ = create_callback_group(rclcpp::CallbackGroupType::MutuallyExclusive);
    rclcpp::SubscriptionOptions yaw_options;
    yaw_options.callback_group = cone_callback_group_;
    yaw_sub_ = create_subscription<geometry_msgs::msg::Vector3Stamped>(
        "/filter/euler", 
        10, 
        [this](const geometry_msgs::msg::Vector3Stamped::SharedPtr msg) {yaw_callback(msg);},
        yaw_options
    );

    // Camera Callback (25 fps)
    // auto camera_callback_group_ = create_callback_group(rclcpp::CallbackGroupType::MutuallyExclusive);
    // camera_timer_ = create_wall_timer(
    //     std::chrono::milliseconds(40),
    //     [this](){camera_callback();},
    //     camera_callback_group_
    // );

    // ---------------------------------------------------------------------------
    //                       INITIALIZATION COMPLETE SEQUENCE
    // ---------------------------------------------------------------------------

    rclcpp::sleep_for(std::chrono::seconds(5));

    // Capture and rectify frames for calibration
    std::pair<uint64_t, cv::Mat> frame_ll = capture_and_rectify_frame(
        get_logger(),
        cap_l,
        map_left_x_ll,
        map_left_y_ll,
        map_right_x_lr,
        map_right_y_lr,
        true, // left_camera==true
        false // outer == false
    );

    if (frame_ll.second.empty()) {
        RCLCPP_ERROR(get_logger(), "Failed to capture frame from left camera left frame.");
    };

    std::pair<uint64_t, cv::Mat> frame_lr = capture_and_rectify_frame(
        get_logger(),
        cap_l,
        map_left_x_ll,
        map_left_y_ll,
        map_right_x_lr,
        map_right_y_lr,
        true, // left_camera==true
        true // inner == true
    );

    if (frame_lr.second.empty()) {
        RCLCPP_ERROR(get_logger(), "Failed to capture frame from left camera right frame.");
    };

        // Capture and rectify frames for calibration
    std::pair<uint64_t, cv::Mat> frame_rr = capture_and_rectify_frame(
        get_logger(),
        cap_r,
        map_left_x_rl,
        map_left_y_rl,
        map_right_x_rr,
        map_right_y_rr,
        false, // right_camera==false
        false // outer == false
    );

    if (frame_rr.second.empty()) {
        RCLCPP_ERROR(get_logger(), "Failed to capture frame from right camera right frame.");
    };

    std::pair<uint64_t, cv::Mat> frame_rl = capture_and_rectify_frame(
        get_logger(),
        cap_r,
        map_left_x_rl,
        map_left_y_rl,
        map_right_x_rr,
        map_right_y_rr,
        false, // right_camera==false
        true // inner == true
    );

    if (frame_rl.second.empty()) {
        RCLCPP_ERROR(get_logger(), "Failed to capture frame from right camera left frame.");
    };

    // Save freeze images
    cv::imwrite("src/point_to_pixel/config/freeze_ll.png", frame_ll.second);
    cv::imwrite("src/point_to_pixel/config/freeze_lr.png", frame_lr.second);
    cv::imwrite("src/point_to_pixel/config/freeze_rr.png", frame_rr.second);
    cv::imwrite("src/point_to_pixel/config/freeze_rl.png", frame_rl.second);

    // Clear directory
    #if save_frames
    save_path = "src/point_to_pixel/img_log/";
    camera_callback_count = 0;

    try
    {
        if (std::filesystem::remove_all(save_path))
        {
            RCLCPP_INFO(get_logger(), "Cleared previous image log.");
        }
        else
        {
            RCLCPP_ERROR(get_logger(), "Failed to delete directory.");
        }
    }
    catch (const std::exception &e)
    {
        RCLCPP_ERROR(get_logger(), "Failed to delete directory: %s", e.what());
    }
    #endif


    // Launch camera thread
    launch_camera_communication().detach();

    // Initialization Complete Message Suite
    RCLCPP_INFO(get_logger(), "Point to Pixel Node INITIALIZED");
    #if verbose
        RCLCPP_INFO(get_logger(), "Verbose Logging On");
    #endif
    #if use_yolo
        RCLCPP_INFO(get_logger(), "Using YOLO Color Detection");
    #else
        RCLCPP_INFO(get_logger(), "Using HSV Color Detection");
    #endif
    #if inner
        RCLCPP_INFO(get_logger(), "Using Inner Cameras on ZEDs");
    #else
        RCLCPP_INFO(get_logger(), "Using Outer Cameras on ZEDs");
    #endif
}

// Returns closest frame to callback time from both cameras
std::tuple<uint64_t, cv::Mat, uint64_t, cv::Mat> PointToPixelNode::get_camera_frame(rclcpp::Time callbackTime)
{
    // Get closest frame from each camera
    l_img_mutex.lock();
    std::pair<uint64_t, cv::Mat> closestFrame_l = find_closest_frame(img_deque_l, callbackTime, get_logger());
    l_img_mutex.unlock();

    r_img_mutex.lock();
    std::pair<uint64_t, cv::Mat> closestFrame_r = find_closest_frame(img_deque_r, callbackTime, get_logger());
    r_img_mutex.unlock();

    return std::make_tuple(closestFrame_l.first, closestFrame_l.second, closestFrame_r.first, closestFrame_r.second);
}

// Returns first velocity and yaw after frame time
std::pair<geometry_msgs::msg::TwistStamped::SharedPtr, geometry_msgs::msg::Vector3Stamped::SharedPtr> PointToPixelNode::get_velocity_yaw(uint64_t frameTime) {
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

// Wrapper function for retrieving the color of cone by combining output from both cameras
int PointToPixelNode::get_cone_class(
    std::pair<Eigen::Vector3d, Eigen::Vector3d> pixel_pair,
    std::pair<cv::Mat, cv::Mat> frame_tuple,
    std::pair<cv::Mat, cv::Mat> detection_pair
) {
    return coloring::get_cone_class(
        pixel_pair,
        frame_tuple,
        detection_pair,
        yellow_filter_low,
        yellow_filter_high,
        blue_filter_low,
        blue_filter_high,
        orange_filter_low,
        orange_filter_high,
        confidence_threshold,
        use_yolo == 1
    );
}

// Finds next closest cone
Cone PointToPixelNode::findClosestCone(const std::vector<Cone>& cones) {
    if (cones.empty()) {
        throw std::runtime_error("Empty cone list");
    }
    
    return *std::min_element(cones.begin(), cones.end(),
        [](const Cone& a, const Cone& b) {
            return a.distance < b.distance;
        });
}

// Calculate angle between two cones
double PointToPixelNode::calculateAngle(const Cone& from, const Cone& to) {
    return std::atan2(to.point.y - from.point.y, to.point.x - from.point.x);
}

// Cone ordering function
std::vector<Cone> PointToPixelNode::orderConesByPathDirection(const std::vector<Cone>& unordered_cones) {
    if (unordered_cones.size() <= 1) {
        return unordered_cones;
    }

    std::vector<Cone> ordered_cones;
    std::vector<Cone> remaining_cones = unordered_cones;
    
    // Start with the closest cone to origin
    Cone current_cone = findClosestCone(remaining_cones);
    ordered_cones.push_back(current_cone);
    
    // Remove the first cone from remaining cones
    remaining_cones.erase(
        std::remove_if(remaining_cones.begin(), remaining_cones.end(),
            [&current_cone](const Cone& c) {
                return c.point.x == current_cone.point.x && 
                       c.point.y == current_cone.point.y;
            }), 
        remaining_cones.end());

    double prev_angle = std::atan2(current_cone.point.y, current_cone.point.x);

    while (!remaining_cones.empty()) {
        // Find next best cone based on distance and angle continuation
        auto next_cone_it = std::min_element(remaining_cones.begin(), remaining_cones.end(),
            [&](const Cone& a, const Cone& b) {
                double angle_a = calculateAngle(current_cone, a);
                double angle_b = calculateAngle(current_cone, b);
                
                // Calculate angle differences
                double angle_diff_a = std::abs(angle_a - prev_angle);
                double angle_diff_b = std::abs(angle_b - prev_angle);
                
                // Combine distance and angle criteria
                double score_a = 0.7 * (a.distance / current_cone.distance) + 
                               0.3 * angle_diff_a;
                double score_b = 0.7 * (b.distance / current_cone.distance) + 
                               0.3 * angle_diff_b;
                
                return score_a < score_b;
            });

        current_cone = *next_cone_it;
        ordered_cones.push_back(current_cone);
        prev_angle = calculateAngle(ordered_cones[ordered_cones.size()-2], current_cone);
        remaining_cones.erase(next_cone_it);
    }

    return ordered_cones;
}

// Calculate motion estimate
std::pair<double, double> PointToPixelNode::getMotionEstimate(double velocity, double angle, double dt)
{
    double v_x = velocity * std::cos(angle);
    double v_y = velocity * std::sin(angle);
    return std::make_pair(v_x * dt, v_y * dt);
}

// Topic callback definition
void PointToPixelNode::cone_callback(const interfaces::msg::PPMConeArray::SharedPtr msg)
{
    // Logging Actions
    #if timing
        auto start_time = high_resolution_clock::now();
        int64_t ms_time_since_lidar_2 = (get_clock()->now().nanoseconds() - msg->header.stamp.sec * 1e9 - msg->header.stamp.nanosec) / 1000;
    #endif

    RCLCPP_INFO(get_logger(), "Received message with %zu cones", msg->cone_array.size());

    // Message Definition
    interfaces::msg::ConeArray message = interfaces::msg::ConeArray();
    message.header = msg->header;
    message.controller_receive_time = msg->header.stamp; 
    message.blue_cones = std::vector<geometry_msgs::msg::Point> {};
    message.yellow_cones = std::vector<geometry_msgs::msg::Point> {};
    message.orange_cones = std::vector<geometry_msgs::msg::Point> {};
    message.unknown_color_cones = std::vector<geometry_msgs::msg::Point> {};
    geometry_msgs::msg::Point point_msg;

    // Returns first frame captured after lidar timestamp in the form: (timestamp_l, frame_l, timestamp_r, frame_r)
    std::tuple<uint64_t, cv::Mat, uint64_t, cv::Mat> frame_tuple = get_camera_frame(msg->header.stamp);

    // Motion modeling for both frames
    std::pair<geometry_msgs::msg::TwistStamped::SharedPtr, geometry_msgs::msg::Vector3Stamped::SharedPtr> vel_yaw_l = get_velocity_yaw(std::get<0>(frame_tuple));
    std::pair<geometry_msgs::msg::TwistStamped::SharedPtr, geometry_msgs::msg::Vector3Stamped::SharedPtr> vel_yaw_r = get_velocity_yaw(std::get<2>(frame_tuple));

    // Null Check
    if (vel_yaw_l.first == nullptr || vel_yaw_l.second == nullptr || vel_yaw_r.first == nullptr || vel_yaw_r.second == nullptr)
    {
        RCLCPP_INFO(get_logger(), "Velocity or Yaw is null");
        return;
    }

    geometry_msgs::msg::TwistStamped::SharedPtr velocity_l = vel_yaw_l.first;
    geometry_msgs::msg::Vector3Stamped::SharedPtr yaw_l = vel_yaw_l.second;

    geometry_msgs::msg::TwistStamped::SharedPtr velocity_r = vel_yaw_r.first;
    geometry_msgs::msg::Vector3Stamped::SharedPtr yaw_r = vel_yaw_r.second;


    double dt_l = (std::get<0>(frame_tuple) - msg->header.stamp.sec * 1e9 - msg->header.stamp.nanosec) / 1e9;
    double dt_r = (std::get<2>(frame_tuple) - msg->header.stamp.sec * 1e9 - msg->header.stamp.nanosec) / 1e9;

    RCLCPP_INFO(get_logger(), "left_camera_timestamp: %llu, right_camera_timestamp: %llu", std::get<0>(frame_tuple), std::get<2>(frame_tuple));
    RCLCPP_INFO(get_logger(), "lidar_timestamp: %f", msg->header.stamp.sec * 1e9 + msg->header.stamp.nanosec);

    auto global_dx_l = velocity_l->twist.linear.x * dt_l;
    auto global_dy_l = velocity_l->twist.linear.y * dt_l;
    auto global_dx_r = velocity_r->twist.linear.x * dt_r;
    auto global_dy_r = velocity_r->twist.linear.y * dt_r;

    auto long_l = global_dx_l * std::cos(yaw_l->vector.z * M_PI / 180) + global_dy_l * std::sin(yaw_l->vector.z * M_PI / 180);
    auto lat_l = -global_dx_l * std::sin(yaw_l->vector.z * M_PI / 180) + global_dy_l * std::cos(yaw_l->vector.z * M_PI / 180);
    auto long_r = global_dx_r * std::cos(yaw_r->vector.z * M_PI / 180) + global_dy_r * std::sin(yaw_r->vector.z * M_PI / 180);
    auto lat_r = -global_dx_r * std::sin(yaw_r->vector.z * M_PI / 180) + global_dy_r * std::cos(yaw_r->vector.z * M_PI / 180);

    std::pair<double, double> ds_l = std::make_pair(-lat_l, long_l);
    std::pair<double, double> ds_r = std::make_pair(-lat_r, long_r);

    #if timing
        RCLCPP_INFO(get_logger(), "Time diff L: %f, Time diff R: %f", dt_l, dt_r);
    #endif

    // Calculate velocity 
    // double v_l = std::sqrt(velocity_l->twist.linear.x * velocity_l->twist.linear.x + velocity_l->twist.linear.y * velocity_l->twist.linear.y);
    // double v_r = std::sqrt(velocity_r->twist.linear.x * velocity_r->twist.linear.x + velocity_r->twist.linear.y * velocity_r->twist.linear.y);

    // std::pair<double, double> ds_l = PointToPixelNode::getMotionEstimate(v_l, yaw_l->vector.z, dt_l);
    // std::pair<double, double> ds_r = PointToPixelNode::getMotionEstimate(v_r, yaw_r->vector.z, dt_r);

    #if timing
        auto camera_time = high_resolution_clock::now();
    #endif

    #if use_yolo
        // Process frames with YOLO
        cv::Mat detection_l = coloring::yolo::process_frame(std::get<1>(frame_tuple), net);
        cv::Mat detection_r = coloring::yolo::process_frame(std::get<3>(frame_tuple), net);
        std::pair<cv::Mat, cv::Mat> detection_pair = std::make_pair(detection_l, detection_r);
    #else
        // Initialize empty matrix if not YOLO
        std::pair<cv::Mat, cv::Mat> detection_pair = std::make_pair(cv::Mat(), cv::Mat());
    #endif

    // Timing Variables
    #if timing
        int transform_time = 0;
        int coloring_time = 0;
    #endif

    std::vector<Cone> unordered_yellow_cones;
    std::vector<Cone> unordered_blue_cones;

    // Iterate through all points in /cpp_cones message
    for (size_t i = 0; i < msg->cone_array.size(); i++) {
        #if timing
            auto loop_start = high_resolution_clock::now();
        #endif
          
        // Transform Point
        std::pair<Eigen::Vector3d, Eigen::Vector3d> pixel_pair = transform_point(
            msg->cone_array[i].cone_points[0],
            std::make_pair(ds_l, ds_r),
            std::make_pair(projection_matrix_l, projection_matrix_r)
        );

        #if timing
            // Time for transform
            auto loop_transform = high_resolution_clock::now();
            transform_time = transform_time + std::chrono::duration_cast<std::chrono::microseconds>(loop_transform - loop_start).count();
        #endif

        // CALCULATE CONE DISTANCE TO CAR (PYTHAG)
        // TODO: This cannot be the best way to do this...
        int cone_class = get_cone_class(pixel_pair, std::make_pair(std::get<1>(frame_tuple), std::get<3>(frame_tuple)), detection_pair);

        #if timing
            // Time for coloring
            auto loop_coloring = high_resolution_clock::now();
            coloring_time = coloring_time + std::chrono::duration_cast<std::chrono::microseconds>(loop_coloring - loop_transform).count();
        #endif

        point_msg.x = msg->cone_array[i].cone_points[0].x;
        point_msg.y = msg->cone_array[i].cone_points[0].y;
        point_msg.z = 0.0;

        switch (cone_class) {
            case 0:
                message.yellow_cones.push_back(point_msg);
                break;
            case 1:
                // message.yellow_cones.push_back(point_msg);
                unordered_yellow_cones.push_back(Cone(point_msg));
                break;
            case 2:
                // message.blue_cones.push_back(point_msg);
                unordered_blue_cones.push_back(Cone(point_msg));
                break;
            default:
                message.unknown_color_cones.push_back(point_msg);
                break;
        }
    }

    #if timing
        auto transform_coloring_time = high_resolution_clock::now();
    #endif

    if (!unordered_yellow_cones.empty()) {
        std::vector<Cone> ordered_yellow = orderConesByPathDirection(unordered_yellow_cones);
        for (const auto& cone : ordered_yellow) {
        message.yellow_cones.push_back(cone.point);
        }
    }

    if (!unordered_blue_cones.empty()) {
        std::vector<Cone> ordered_blue = orderConesByPathDirection(unordered_blue_cones);
        for (const auto& cone : ordered_blue) {
            message.blue_cones.push_back(cone.point);
        }
    }

    #if timing
        auto end_ordering_time = high_resolution_clock::now();
        auto ordering_time = std::chrono::duration_cast<std::chrono::microseconds>(end_ordering_time - transform_coloring_time).count();
    #endif

    int cones_published = message.orange_cones.size() + message.yellow_cones.size() + message.blue_cones.size();
    int yellow_cones = message.yellow_cones.size();
    int blue_cones = message.blue_cones.size();
    int orange_cones = message.orange_cones.size();
    int unknown_color_cones = message.unknown_color_cones.size();
    
    RCLCPP_INFO(
        get_logger(), 
        "Transform callback triggered. Published %d cones. %d yellow, %d blue, %d orange, and %d unknown.", 
        cones_published, yellow_cones, blue_cones, orange_cones, unknown_color_cones
    );

    #if timing
        auto end_time = high_resolution_clock::now();
        auto stamp_time = msg->header.stamp;
        auto ms_time_since_lidar = get_clock()->now() - stamp_time;

        RCLCPP_INFO(get_logger(), "Get Camera Frame  %ld microseconds.", std::chrono::duration_cast<std::chrono::microseconds>(camera_time - start_time).count());
        RCLCPP_INFO(get_logger(), "Total Transform and Coloring Time  %ld microseconds.", std::chrono::duration_cast<std::chrono::microseconds>(transform_coloring_time - camera_time).count());
        RCLCPP_INFO(get_logger(), "--Total Transform Time  %ld microseconds.", transform_time);
        RCLCPP_INFO(get_logger(), "--Total Coloring Time  %ld microseconds.", coloring_time);
        RCLCPP_INFO(get_logger(), "--Total Ordering Time  %ld microseconds.", ordering_time);
        auto time_diff = end_time - start_time;
        RCLCPP_INFO(get_logger(), "Total PPM Time %ld microseconds.", std::chrono::duration_cast<std::chrono::microseconds>(time_diff).count());
        RCLCPP_INFO(get_logger(), "Total Time from Lidar  %ld microseconds.", ms_time_since_lidar.nanoseconds() / 1000);
        RCLCPP_INFO(get_logger(), "Total Time from Lidar to start  %ld microseconds.", ms_time_since_lidar_2);
    #endif
    
    cone_pub_->publish(message);
}

#if save_frames
void PointToPixelNode::save_frame(std::pair<uint64_t, cv::Mat> frame_l, std::pair<uint64_t, cv::Mat> frame_r)
{
    std::string l_filename = save_path + std::to_string(frame_l.first) + ".png";
    std::string r_filename = save_path + std::to_string(frame_r.first) + ".png";
    cv::imwrite(l_filename, frame_l.second);
    cv::imwrite(r_filename, frame_r.second);
}
#endif

// Camera Callback (Populates and maintain deque)
void PointToPixelNode::camera_callback()
{
    // Capture and rectify frame from camera l
    std::pair<uint64_t, cv::Mat> frame_l = capture_and_rectify_frame(
        get_logger(),
        cap_l,
        map_left_x_ll,
        map_left_y_ll,
        map_right_x_lr,
        map_right_y_lr,
        true, // left_camera==true
        inner == 1
    );

    // Capture and rectify frame from camera r
    std::pair<uint64_t, cv::Mat> frame_r = capture_and_rectify_frame(
        get_logger(),
        cap_r,
        map_left_x_rl,
        map_left_y_rl,
        map_right_x_rr,
        map_right_y_rr,
        false, // left_camera==false
        inner == 1
    );

    #if save_frames
    if(camera_callback_count % frame_interval == 0)
    {
        save_frame(frame_l, frame_r);
    }    

    camera_callback_count++;
    #endif



    // Deque Management and Updating
    l_img_mutex.lock();
    while (img_deque_l.size() >= max_deque_size) {
        img_deque_l.pop_front();
    }
    img_deque_l.push_back(frame_l);
    l_img_mutex.unlock();

    r_img_mutex.lock();
    while (img_deque_r.size() >= max_deque_size) {
        img_deque_r.pop_front();
    }

    img_deque_r.push_back(frame_r);
    r_img_mutex.unlock();
}

// Launches camera thread
std::thread PointToPixelNode::launch_camera_communication() {
    return std::thread{
        [this]
        {
            while (rclcpp::ok)
            {
                camera_callback();
                // 33 FPS
                rclcpp::sleep_for(std::chrono::milliseconds(30));
            }
        }};
}

void PointToPixelNode::velocity_callback(geometry_msgs::msg::TwistStamped::SharedPtr msg)
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

void PointToPixelNode::yaw_callback(geometry_msgs::msg::Vector3Stamped::SharedPtr msg)
{
    // Deque Management and Updating
    // yaw_mutex.lock();
    while (yaw_deque.size() >= max_deque_size)
    {
        yaw_deque.pop_front();
    }
    yaw_deque.push_back(msg);
}

int main(int argc, char **argv)
{
    rclcpp::init(argc, argv);
    
    // Multithreading for timer callback
    rclcpp::executors::MultiThreadedExecutor executor;
    rclcpp::Node::SharedPtr node = std::make_shared<PointToPixelNode>();
    executor.add_node(node);
    executor.spin();
    
    rclcpp::shutdown();
    return 0;
}