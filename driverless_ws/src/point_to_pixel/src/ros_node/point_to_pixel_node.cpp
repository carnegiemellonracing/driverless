#include "point_to_pixel_node.hpp"

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

    // ---------------------------------------------------------------------------
    //                       INITIALIZATION COMPLETE SEQUENCE
    // ---------------------------------------------------------------------------
    
    // Clear directory
    #if save_frames
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
        std::filesystem::create_directory(save_path);
    }
    catch (const std::exception &e)
    {
        RCLCPP_ERROR(get_logger(), "Failed to delete directory: %s", e.what());
    }
    #endif

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
    #if save_frames
        RCLCPP_INFO(get_logger(), "Saving Frames", save_path);
    #endif


    // Wait for cameras to initialize auto exposure and brightness settings
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


    // Launch camera thread
    launch_camera_communication().detach();

    #if save_frames
    // Launch frame saving thread
    launch_frame_saving().detach();
    #endif
}

// Wrapper function to retrieve closest frame to callback time from both cameras
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

// Wrapper function for retrieving the color of cone by combining output from both cameras
int PointToPixelNode::get_cone_class(
    std::pair<Eigen::Vector3d, Eigen::Vector3d> pixel_pair,
    std::pair<cv::Mat, cv::Mat> frame_pair,
    std::pair<std::vector<cv::Mat> , std::vector<cv::Mat> > detection_pair
) {
    return cones::get_cone_class(
        pixel_pair,
        frame_pair,
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

// Topic callback definition
void PointToPixelNode::cone_callback(const interfaces::msg::PPMConeArray::SharedPtr msg)
{
    // Logging Actions
    #if timing
    auto start_time = high_resolution_clock::now();
    uint64_t ms_time_since_lidar_2 = (get_clock()->now().nanoseconds() - msg->header.stamp.sec * 1e9 - msg->header.stamp.nanosec) / 1e6;
    #endif

    RCLCPP_INFO(get_logger(), "Received message with %zu cones \n", msg->cone_array.size());

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
    if (std::get<1>(frame_tuple).empty() || std::get<3>(frame_tuple).empty())
    {
        RCLCPP_WARN(get_logger(), "No frame, likely an empty frame deque");
        return;
    }
    std::pair<cv::Mat, cv::Mat> frame_pair = std::make_pair(std::get<1>(frame_tuple), std::get<3>(frame_tuple));

    #if timing
    auto camera_time = high_resolution_clock::now();
    #endif

    #if use_yolo

        #if timing
        auto yolo_l_start_time = high_resolution_clock::now();
        #endif

        // Process frames with YOLO
        std::vector<cv::Mat> detection_l = cones::yolo::process_frame(std::get<1>(frame_tuple), net);
        #if timing
        auto yolo_l_end_time = high_resolution_clock::now();
        #endif

        std::vector<cv::Mat> detection_r = cones::yolo::process_frame(std::get<3>(frame_tuple), net);
        #if timing
            auto yolo_r_end_time = high_resolution_clock::now();
            RCLCPP_INFO(get_logger(), "Total yolo time %llu ms", std::chrono::duration_cast<std::chrono::milliseconds>(yolo_r_end_time - yolo_l_start_time).count());
            RCLCPP_INFO(get_logger(), "\t - Left yolo time %llu ms", std::chrono::duration_cast<std::chrono::milliseconds>(yolo_l_end_time - yolo_l_start_time).count());
            RCLCPP_INFO(get_logger(), "\t - Right yolo time %llu ms", std::chrono::duration_cast<std::chrono::milliseconds>(yolo_r_end_time - yolo_l_end_time).count());
        #endif
        
        std::pair<std::vector<cv::Mat> , std::vector<cv::Mat> > detection_pair = std::make_pair(detection_l, detection_r);
    #else
        // Initialize empty matrix if not YOLO
        std::pair<std::vector<cv::Mat> , std::vector<cv::Mat>> detection_pair = std::make_pair(nullptr, nullptr);
    #endif

    #if save_frames
    std::vector<std::pair<cv::Point, cv::Point>> unknown_transformed_pixels, yellow_transformed_pixels, blue_transformed_pixels, orange_transformed_pixels;
    #endif

    // Motion modeling for both frames
    auto [velocity_l_camera_frame, yaw_l_camera_frame] = get_velocity_yaw(get_logger(), &yaw_mutex, &velocity_mutex, velocity_deque, yaw_deque, std::get<0>(frame_tuple));
    auto [velocity_r_camera_frame, yaw_r_camera_frame] = get_velocity_yaw(get_logger(), &yaw_mutex, &velocity_mutex, velocity_deque, yaw_deque, std::get<2>(frame_tuple));

    auto current_lidar_time = msg->header.stamp.sec * 1e9 + msg->header.stamp.nanosec;
    auto [velocity_lidar_frame, yaw_lidar_frame] = get_velocity_yaw(get_logger(), &yaw_mutex, &velocity_mutex, velocity_deque, yaw_deque, current_lidar_time);

    if (velocity_lidar_frame == nullptr || yaw_lidar_frame == nullptr)
    {
        return;
    }

    double dt_l = (std::get<0>(frame_tuple) - current_lidar_time) / 1e9; //::max( 0.0, time_diff_l );
    double dt_r = (std::get<2>(frame_tuple) - current_lidar_time) / 1e9; // time_diffstd::max( 0.0, time_diff_r );

    double average_velocity_l_x = (velocity_l_camera_frame->twist.linear.x + velocity_lidar_frame->twist.linear.x) / 2;
    double average_velocity_l_y = (velocity_l_camera_frame->twist.linear.y + velocity_lidar_frame->twist.linear.y) / 2;
    double average_velocity_r_x = (velocity_r_camera_frame->twist.linear.x + velocity_lidar_frame->twist.linear.x) / 2;
    double average_velocity_r_y = (velocity_r_camera_frame->twist.linear.y + velocity_lidar_frame->twist.linear.y) / 2;

    auto global_dx_l = average_velocity_l_x * dt_l;
    auto global_dy_l = average_velocity_l_y * dt_l;
    auto global_dyaw_l = yaw_l_camera_frame->vector.z - yaw_lidar_frame->vector.z;

    auto global_dx_r = average_velocity_r_x * dt_r;
    auto global_dy_r = average_velocity_r_y * dt_r;
    auto global_dyaw_r = yaw_r_camera_frame->vector.z - yaw_lidar_frame->vector.z;

    double yaw_lidar_rad = yaw_lidar_frame->vector.z * M_PI / 180;

    std::pair<double, double> global_frame_change_l = std::make_pair(global_dx_l, global_dy_l);
    std::pair<double, double> long_lat_l = global_frame_to_local_frame(global_frame_change_l, yaw_lidar_rad);

    std::pair<double, double> global_frame_change_r = std::make_pair(global_dx_r, global_dy_r);
    std::pair<double, double> long_lat_r = global_frame_to_local_frame(global_frame_change_r, yaw_lidar_rad);

    std::vector<cones::Cone> unordered_yellow_cones;
    std::vector<cones::Cone> unordered_blue_cones;

    // Iterate through all points in /cpp_cones message
    for (size_t i = 0; i < msg->cone_array.size(); i++) {
        // Transform Point
        std::pair<Eigen::Vector3d, Eigen::Vector3d> pixel_pair = transform_point(
            get_logger(),
            msg->cone_array[i].cone_points[0],
            {long_lat_l, long_lat_r},
            {global_dyaw_l, global_dyaw_r},
            {projection_matrix_l, projection_matrix_r}
        );

        int cone_class = get_cone_class(pixel_pair, frame_pair, detection_pair);

        point_msg.x = msg->cone_array[i].cone_points[0].x;
        point_msg.y = msg->cone_array[i].cone_points[0].y;
        point_msg.z = 0.0; 

        switch (cone_class) {
            case 0:
                message.yellow_cones.push_back(point_msg);

                #if save_frames
                    orange_transformed_pixels.push_back(
                        std::make_pair(
                            cv::Point(static_cast<int>(pixel_pair.first.head(2)(0)), static_cast<int>(pixel_pair.first.head(2)(1))), 
                            cv::Point(static_cast<int>(pixel_pair.second.head(2)(0)), static_cast<int>(pixel_pair.second.head(2)(1)))
                        )
                    );
                #endif
                break;
            case 1:
                // message.yellow_cones.push_back(point_msg);
                unordered_yellow_cones.push_back(cones::Cone(point_msg));

                #if save_frames
                yellow_transformed_pixels.push_back(
                    std::make_pair(
                        cv::Point(static_cast<int>(pixel_pair.first.head(2)(0)), static_cast<int>(pixel_pair.first.head(2)(1))),
                        cv::Point(static_cast<int>(pixel_pair.second.head(2)(0)), static_cast<int>(pixel_pair.second.head(2)(1)))));
                #endif
                break;
            case 2:
                // message.blue_cones.push_back(point_msg);
                unordered_blue_cones.push_back(cones::Cone(point_msg));

                #if save_frames
                blue_transformed_pixels.push_back(
                    std::make_pair(
                        cv::Point(static_cast<int>(pixel_pair.first.head(2)(0)), static_cast<int>(pixel_pair.first.head(2)(1))),
                        cv::Point(static_cast<int>(pixel_pair.second.head(2)(0)), static_cast<int>(pixel_pair.second.head(2)(1)))));
                #endif
                break;
            default:
                message.unknown_color_cones.push_back(point_msg);

                #if save_frames
                unknown_transformed_pixels.push_back(
                    std::make_pair(
                        cv::Point(static_cast<int>(pixel_pair.first.head(2)(0)), static_cast<int>(pixel_pair.first.head(2)(1))),
                        cv::Point(static_cast<int>(pixel_pair.second.head(2)(0)), static_cast<int>(pixel_pair.second.head(2)(1)))));
                #endif
                break;
        }
    }

    #if timing
    auto end_transform_coloring_time = high_resolution_clock::now();
    auto transform_coloring_time = std::chrono::duration_cast<std::chrono::milliseconds>(end_transform_coloring_time - yolo_r_end_time).count();
    #endif

    // Cone ordering
    if (!unordered_yellow_cones.empty()) {
        std::vector<cones::Cone> ordered_yellow = cones::order_cones(unordered_yellow_cones);
        for (const auto& cone : ordered_yellow) {
            message.yellow_cones.push_back(cone.point);
        }
    }

    if (!unordered_blue_cones.empty()) {
        std::vector<cones::Cone> ordered_blue = cones::order_cones(unordered_blue_cones);
        for (const auto& cone : ordered_blue) {
            message.blue_cones.push_back(cone.point);
        }
    }

    #if timing
    auto end_ordering_time = high_resolution_clock::now();
    auto ordering_time = std::chrono::duration_cast<std::chrono::milliseconds>(end_ordering_time - end_transform_coloring_time).count();
    #endif

    // Frame annotation
    #if save_frames
    if (camera_callback_count == frame_interval) {
        camera_callback_count = 0;
        #if use_yolo
        // Add Yolo Bounding Boxes to frame

        cv::Mat frame_l_canvas = frame_pair.first.clone();
        cv::Mat frame_r_canvas = frame_pair.second.clone();
        float alpha = .3;

        cones::yolo::draw_bounding_boxes(frame_pair.first, frame_l_canvas, detection_l, frame_pair.first.cols, frame_pair.first.rows, confidence_threshold);
        cones::yolo::draw_bounding_boxes(frame_pair.second, frame_r_canvas, detection_pair.second, frame_pair.second.cols, frame_pair.second.rows, confidence_threshold);

        #endif

        // Add transformed points to frame
        for (size_t i = 0; i < yellow_transformed_pixels.size(); i++) {
            // Correctly mapped yellow pixels are green    
            cv::circle(frame_pair.first, yellow_transformed_pixels[i].first, 2, cv::Scalar(0, 255, 0), 5);
            cv::circle(frame_pair.second, yellow_transformed_pixels[i].second, 2, cv::Scalar(0, 255, 0), 5);
        }

        for (size_t i = 0; i < blue_transformed_pixels.size(); i++) {
            // Correctly mapped blue pixels are red    
            cv::circle(frame_pair.first, blue_transformed_pixels[i].first, 2, cv::Scalar(0, 0, 255), 5);
            cv::circle(frame_pair.second, blue_transformed_pixels[i].second, 2, cv::Scalar(0, 0, 255), 5);
        }

        for (size_t i = 0; i < orange_transformed_pixels.size(); i++) {
            cv::circle(frame_pair.first, orange_transformed_pixels[i].first, 2, cv::Scalar(0, 69, 255), 5);
            cv::circle(frame_pair.second, orange_transformed_pixels[i].second, 2, cv::Scalar(0, 69, 255), 5);
        }
        
        for (size_t i = 0; i < unknown_transformed_pixels.size(); i++) {
            cv::circle(frame_pair.first, unknown_transformed_pixels[i].first, 2, cv::Scalar(0, 0, 0), 5);
            cv::circle(frame_pair.second, unknown_transformed_pixels[i].second, 2, cv::Scalar(0, 0, 0), 5);
        }
        
        save_mutex.lock();
        save_queue.push(frame_tuple);
        save_mutex.unlock();
    }
    camera_callback_count += 1;

    #if timing
    auto end_drawing_time = high_resolution_clock::now();
    auto drawing_time = std::chrono::duration_cast<std::chrono::milliseconds>(end_drawing_time - end_ordering_time).count();
    #endif
    #endif
    
    // Publish message and log
    int cones_published = message.orange_cones.size() + message.yellow_cones.size() + message.blue_cones.size();
    int yellow_cones = message.yellow_cones.size();
    int blue_cones = message.blue_cones.size();
    int orange_cones = message.orange_cones.size();
    int unknown_color_cones = message.unknown_color_cones.size();
    
    RCLCPP_INFO(
        get_logger(), 
        "Published %d yellow, %d blue, %d orange, and %d unknown cones.", 
        yellow_cones, blue_cones, orange_cones, unknown_color_cones
    );

    #if timing
        auto end_time = high_resolution_clock::now();
        auto stamp_time = msg->header.stamp.sec * 1e9 + msg->header.stamp.nanosec;
        auto ms_time_since_lidar = (get_clock()->now().nanoseconds() - stamp_time) / 1e6;

        RCLCPP_INFO(get_logger(), "Get Camera Frame  %ld ms.", std::chrono::duration_cast<std::chrono::milliseconds>(camera_time - start_time).count());
        RCLCPP_INFO(get_logger(), "Total Transform and Coloring Time  %ld ms.", transform_coloring_time);
        #if save_frames
        RCLCPP_INFO(get_logger(), "Total Frame Annotation Time  %ld ms.", drawing_time);
        #endif
        RCLCPP_INFO(get_logger(), "Total Cone Ordering Time  %ld ms. \n", ordering_time);
        auto time_diff = end_time - start_time;
        RCLCPP_INFO(get_logger(), "Total Time from Lidar:  %ld ms.", (uint64_t) ms_time_since_lidar);
        RCLCPP_INFO(get_logger(), "Total Time from Lidar to start  %ld ms.", ms_time_since_lidar_2);
        RCLCPP_INFO(get_logger(), "Total PPM Time %ld ms.", std::chrono::duration_cast<std::chrono::milliseconds>(time_diff).count());
    #endif
    RCLCPP_INFO(get_logger(), "----------------------------------------------");
    
    cone_pub_->publish(message);
}

#if save_frames
void PointToPixelNode::save_frame(const rclcpp::Logger &logger, std::tuple<uint64_t, cv::Mat, uint64_t, cv::Mat> frame_tuple)
{
    auto start = std::chrono::high_resolution_clock::now();
    std::string l_filename = save_path + std::to_string(std::get<0>(frame_tuple)) + ".bmp";
    std::string r_filename = save_path + std::to_string(std::get<2>(frame_tuple)) + ".bmp";
    cv::imwrite(l_filename, std::get<1>(frame_tuple));
    cv::imwrite(r_filename, std::get<3>(frame_tuple));
    
    #if timing
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    RCLCPP_INFO(logger, "Saved frame in %ld ms.", duration.count());
    #endif
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

#if save_frames
std::thread PointToPixelNode::launch_frame_saving()
{
    return std::thread{
        [this]
        {
            while (rclcpp::ok)
            {
                save_mutex.lock();
                if(!save_queue.empty()) {
                    save_frame(get_logger(), save_queue.front());
                    save_queue.pop();
                }
                save_mutex.unlock();
                // Save a frame every half second
                rclcpp::sleep_for(std::chrono::milliseconds(1));
            }
        }};
}
#endif

void PointToPixelNode::velocity_callback(geometry_msgs::msg::TwistStamped::SharedPtr msg)
{   
    velocity_mutex.lock();
    // Deque Management and Updating
    while (velocity_deque.size() >= max_deque_size)
    {
        velocity_deque.pop_front();
    }
    velocity_deque.push_back(msg);
    velocity_mutex.unlock();
}

void PointToPixelNode::yaw_callback(geometry_msgs::msg::Vector3Stamped::SharedPtr msg)
{
    yaw_mutex.lock();
    // Deque Management and Updating
    while (yaw_deque.size() >= max_deque_size)
    {
        yaw_deque.pop_front();
    }
    yaw_deque.push_back(msg);
    yaw_mutex.unlock();
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