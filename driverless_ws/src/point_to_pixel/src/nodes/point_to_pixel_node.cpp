#include "point_to_pixel_node.hpp"
#include "../cones/svm/svm_recolour.hpp"

// Constructor definition
PointToPixelNode::PointToPixelNode() : Node("point_to_pixel"),
    params([]() {sl_oc::video::VideoParams p; p.res = sl_oc::video::RESOLUTION::HD1080; p.fps = sl_oc::video::FPS::FPS_30; return p;}()),
    cap_l(sl_oc::video::VideoCapture(params)),
    cap_r(sl_oc::video::VideoCapture(params)),
    left_cam(cap_l, cv::Mat(), cv::Mat(), cv::Mat(), cv::Mat(), 0),
    right_cam(cap_r, cv::Mat(), cv::Mat(), cv::Mat(), cv::Mat(), 2)
{
    // ---------------------------------------------------------------------------
    //                              CAMERA INITIALIZATION
    // ---------------------------------------------------------------------------

    // Initialize cameras
    if (!camera::initialize_camera(left_cam, get_logger())) {
        rclcpp::shutdown(); // Shutdown node if camera initialization fails
        return;
    }
    
    if (!camera::initialize_camera(right_cam, get_logger())) {
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

    // Default Transform
    std::vector<double> param_default(12, 0.0);

    // Confidence Threshold
    declare_parameter("confidence_threshold", 0.25);

    // Load Projection Matrix if inner is set to true, then load lr and rl, else load ll and rr
    #if inner
        declare_parameter("projection_matrix_lr", param_default);
        declare_parameter("projection_matrix_rl", param_default);
        param_l = get_parameter("projection_matrix_lr").as_double_array();
        param_r = get_parameter("projection_matrix_rl").as_double_array();
    #else
        declare_parameter("projection_matrix_ll", param_default);
        declare_parameter("projection_matrix_rr", param_default);
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
    
    #if use_yolo
    net = cones::coloring::yolo::init_model(yolo_model_path);
    if (net.empty()) {
        RCLCPP_ERROR(get_logger(), "Failed to load YOLO model");
        return;
    }
    #endif
    
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
    velocity_options.callback_group = create_callback_group(rclcpp::CallbackGroupType::MutuallyExclusive);
    velocity_sub_ = create_subscription<geometry_msgs::msg::TwistStamped>(
        "/filter/twist", 
        10, 
        [this](const geometry_msgs::msg::TwistStamped::SharedPtr msg) {vel_callback(msg);},
        velocity_options
    );

    // auto yaw_callback_group_ = create_callback_group(rclcpp::CallbackGroupType::MutuallyExclusive);
    rclcpp::SubscriptionOptions yaw_options;
    yaw_options.callback_group = create_callback_group(rclcpp::CallbackGroupType::MutuallyExclusive);
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
    rclcpp::sleep_for(std::chrono::seconds(3));

    // Capture and rectify frames for calibration
    camera::capture_freezes(get_logger(), left_cam, right_cam, l_img_mutex, r_img_mutex, img_deque_l, img_deque_r, inner == 1);
    
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
    std::pair<uint64_t, cv::Mat> closestFrame_l = camera::find_closest_frame(img_deque_l, callbackTime, get_logger());
    l_img_mutex.unlock();

    r_img_mutex.lock();
    std::pair<uint64_t, cv::Mat> closestFrame_r = camera::find_closest_frame(img_deque_r, callbackTime, get_logger());
    r_img_mutex.unlock();

    return std::make_tuple(closestFrame_l.first, closestFrame_l.second, closestFrame_r.first, closestFrame_r.second);
}

// Implementation of capture_freezes method
void PointToPixelNode::capture_freezes()
{
    // Use the camera namespace's capture_freezes function
    camera::capture_freezes(
        get_logger(),
        left_cam,
        right_cam,
        l_img_mutex,
        r_img_mutex,
        img_deque_l,
        img_deque_r,
        inner == 1
    );
}

// Camera Callback (Populates and maintain deque)
void PointToPixelNode::camera_callback()
{
    // Capture and rectify frame from left camera
    std::pair<uint64_t, cv::Mat> frame_l = camera::capture_and_rectify_frame(
        get_logger(),
        left_cam,
        true, // left_camera==true
        inner == 1
    );

    // Capture and rectify frame from right camera
    std::pair<uint64_t, cv::Mat> frame_r = camera::capture_and_rectify_frame(
        get_logger(),
        right_cam,
        false, // right_camera==false
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

// Wrapper function for retrieving the color of cone by combining output from both cameras
int PointToPixelNode::get_cone_class(
    std::pair<Eigen::Vector3d, Eigen::Vector3d> pixel_pair,
    std::pair<cv::Mat, cv::Mat> frame_pair,
    std::pair<std::vector<cv::Mat>, std::vector<cv::Mat>> detection_pair
) {
    #if use_yolo
    return cones::coloring::yolo::get_cone_class(
        pixel_pair,
        frame_pair,
        detection_pair,
        yellow_filter_low,
        yellow_filter_high,
        blue_filter_low,
        blue_filter_high,
        orange_filter_low,
        orange_filter_high,
        confidence_threshold
    );
    #else
    return cones::coloring::hsv::get_cone_class(
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
    );
    #endif
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
    geometry_msgs::msg::Point point_msg;

    // Returns first frame captured after lidar timestamp in the form: (timestamp_l, frame_l, timestamp_r, frame_r)
    std::tuple<uint64_t, cv::Mat, uint64_t, cv::Mat> frame_tuple = get_camera_frame(msg->header.stamp);
    std::pair<cv::Mat, cv::Mat> frame_pair = {std::get<1>(frame_tuple), std::get<3>(frame_tuple)};

    // Check that frame is not empty
    if (frame_pair.first.empty() || frame_pair.second.empty())
    {
        RCLCPP_WARN(get_logger(), "No frame, likely an empty frame deque");
        return;
    }

    #if timing
    auto camera_time = high_resolution_clock::now();
    uint64_t ms_lidar_camera_diff_l = (std::get<0>(frame_tuple) - msg->header.stamp.sec * 1e9 + msg->header.stamp.nanosec) / 1e6;
    uint64_t ms_lidar_camera_diff_r = (std::get<2>(frame_tuple) - msg->header.stamp.sec * 1e9 + msg->header.stamp.nanosec) / 1e6;
    #endif

    // Motion modeling for both frames
    auto [vel_l_camera_frame, yaw_l_camera_frame] = transform::get_vel_yaw(get_logger(), yaw_mutex, vel_mutex, vel_deque, yaw_deque, std::get<0>(frame_tuple));
    auto [vel_r_camera_frame, yaw_r_camera_frame] = transform::get_vel_yaw(get_logger(), yaw_mutex, vel_mutex, vel_deque, yaw_deque, std::get<2>(frame_tuple));

    auto current_lidar_time = msg->header.stamp.sec * 1e9 + msg->header.stamp.nanosec;
    auto [vel_lidar_frame, yaw_lidar_frame] = transform::get_vel_yaw(get_logger(), yaw_mutex, vel_mutex, vel_deque, yaw_deque, current_lidar_time);

    if (vel_lidar_frame == nullptr || yaw_lidar_frame == nullptr)
    {
        RCLCPP_ERROR(get_logger(), "Could not get velocity and yaw from lidar");
        return;
    }

    double dt_l = (std::get<0>(frame_tuple) > current_lidar_time) ? (std::get<0>(frame_tuple) - current_lidar_time) / 1e9 : 0.0;
    double avg_vel_l_x = (vel_l_camera_frame->twist.linear.x + vel_lidar_frame->twist.linear.x) / 2;
    double avg_vel_l_y = (vel_l_camera_frame->twist.linear.y + vel_lidar_frame->twist.linear.y) / 2;
    auto global_dx_l = avg_vel_l_x * dt_l;
    auto global_dy_l = avg_vel_l_y * dt_l;
    std::pair<double, double> global_frame_change_l = {global_dx_l, global_dy_l};
    auto global_dyaw_l = yaw_l_camera_frame->vector.z - yaw_lidar_frame->vector.z;

    double dt_r = (std::get<2>(frame_tuple) > current_lidar_time) ? (std::get<2>(frame_tuple) - current_lidar_time) / 1e9 : 0.0;
    double avg_vel_r_x = (vel_r_camera_frame->twist.linear.x + vel_lidar_frame->twist.linear.x) / 2;
    double avg_vel_r_y = (vel_r_camera_frame->twist.linear.y + vel_lidar_frame->twist.linear.y) / 2;
    auto global_dx_r = avg_vel_r_x * dt_r;
    auto global_dy_r = avg_vel_r_y * dt_r;
    auto global_dyaw_r = yaw_r_camera_frame->vector.z - yaw_lidar_frame->vector.z;
    std::pair<double, double> global_frame_change_r = {global_dx_r, global_dy_r};

    double yaw_lidar_rad = yaw_lidar_frame->vector.z * M_PI / 180;

    std::pair<double, double> long_lat_l = transform::global_frame_to_local_frame(global_frame_change_l, yaw_lidar_rad);
    std::pair<double, double> long_lat_r = transform::global_frame_to_local_frame(global_frame_change_r, yaw_lidar_rad);

    // Process frames with YOLO
    #if use_yolo
    #if timing
    auto yolo_l_start_time = high_resolution_clock::now();
    #endif // use_yolo

    // Get YOLO detection outputs
    std::vector<cv::Mat> detection_l = cones::coloring::yolo::process_frame(std::get<1>(frame_tuple), net);
    #if timing
    auto yolo_l_end_time = high_resolution_clock::now();
    #endif // timing

    std::vector<cv::Mat> detection_r = cones::coloring::yolo::process_frame(std::get<3>(frame_tuple), net);
    #if timing
    auto yolo_r_end_time = high_resolution_clock::now();
    #endif // timing
    
    std::pair<std::vector<cv::Mat>, std::vector<cv::Mat>> detection_pair = {detection_l, detection_r};
    #else
    // Initialize empty matrix if not YOLO
    std::pair<std::vector<cv::Mat>, std::vector<cv::Mat>> detection_pair = {std::vector<cv::Mat>(), std::vector<cv::Mat>()};
    #endif // use_yolo

    // Declare point and cone vectors
    cones::TrackBounds unordered;
    #if save_frames
    std::vector<std::pair<cv::Point, cv::Point>> unknown_transformed_pixels, yellow_transformed_pixels, blue_transformed_pixels, orange_transformed_pixels;
    #endif // save_frames

    // Iterate through all points in /cpp_cones message
    for (size_t i = 0; i < msg->cone_array.size(); i++) {
        // Motion model lidar point then transform to camera space
        std::pair<Eigen::Vector3d, Eigen::Vector3d> pixel_pair = transform::transform_point(
            get_logger(),
            msg->cone_array[i].cone_points[0],
            {long_lat_l, long_lat_r},
            {global_dyaw_l, global_dyaw_r},
            {projection_matrix_l, projection_matrix_r}
        );

        // Get cone class of lidar point by combining output from both cameras
        int cone_class = get_cone_class(pixel_pair, frame_pair, detection_pair);

        point_msg.x = msg->cone_array[i].cone_points[0].x;
        point_msg.y = msg->cone_array[i].cone_points[0].y;
        point_msg.z = 0.0; 

        switch (cone_class) {
            case 0:
                message.yellow_cones.push_back(point_msg);

                #if save_frames
                    orange_transformed_pixels.emplace_back(
                        cv::Point(static_cast<int>(pixel_pair.first.head(2)(0)), static_cast<int>(pixel_pair.first.head(2)(1))), 
                        cv::Point(static_cast<int>(pixel_pair.second.head(2)(0)), static_cast<int>(pixel_pair.second.head(2)(1)))
                    );
                #endif
                break;
            case 1:
                // message.yellow_cones.push_back(point_msg);
                unordered.yellow.push_back(cones::Cone(point_msg));

                #if save_frames
                yellow_transformed_pixels.emplace_back(
                    cv::Point(static_cast<int>(pixel_pair.first.head(2)(0)), static_cast<int>(pixel_pair.first.head(2)(1))),
                    cv::Point(static_cast<int>(pixel_pair.second.head(2)(0)), static_cast<int>(pixel_pair.second.head(2)(1)))
                );
                #endif
                break;
            case 2:
                // message.blue_cones.push_back(point_msg);
                unordered.blue.push_back(cones::Cone(point_msg));

                #if save_frames
                blue_transformed_pixels.emplace_back(
                    cv::Point(static_cast<int>(pixel_pair.first.head(2)(0)), static_cast<int>(pixel_pair.first.head(2)(1))),
                    cv::Point(static_cast<int>(pixel_pair.second.head(2)(0)), static_cast<int>(pixel_pair.second.head(2)(1)))
                );
                #endif
                break;
            default:
                message.unknown_color_cones.push_back(point_msg);

                #if save_frames
                unknown_transformed_pixels.emplace_back(
                    cv::Point(static_cast<int>(pixel_pair.first.head(2)(0)), static_cast<int>(pixel_pair.first.head(2)(1))),
                    cv::Point(static_cast<int>(pixel_pair.second.head(2)(0)), static_cast<int>(pixel_pair.second.head(2)(1)))
                );
                #endif
                break;
        }
    }

    #if timing
    auto end_transform_coloring_time = high_resolution_clock::now();
    auto transform_coloring_time = std::chrono::duration_cast<std::chrono::milliseconds>(end_transform_coloring_time - yolo_r_end_time).count();
    #endif

    // Recolouring
    unordered = cones::recolouring::recolour_cones(unordered, svm_C);

    // Cone ordering
    // cones::TrackBounds ordered;
    
    if (!unordered.yellow.empty()) {
        // ordered.yellow = cones::order_cones(unordered.yellow, max_distance_threshold);
        for (const auto& cone : unordered.yellow) {
            message.yellow_cones.push_back(cone.point);
        }
    }

    if (!unordered.blue.empty()) {
        // ordered.blue = cones::order_cones(unordered.blue, max_distance_threshold);
        for (const auto& cone : unordered.blue) {
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

        cones::coloring::yolo::draw_bounding_boxes(frame_pair.first, frame_l_canvas, detection_l, frame_pair.first.cols, frame_pair.first.rows, confidence_threshold);
        cones::coloring::yolo::draw_bounding_boxes(frame_pair.second, frame_r_canvas, detection_pair.second, frame_pair.second.cols, frame_pair.second.rows, confidence_threshold);

        #endif // use_yolo

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
    #endif // timing
    #endif // save_frames

    #if timing
    auto end_time = high_resolution_clock::now();
    auto stamp_time = msg->header.stamp.sec * 1e9 + msg->header.stamp.nanosec;
    auto ms_time_since_lidar = (get_clock()->now().nanoseconds() - stamp_time) / 1e6;


    RCLCPP_INFO(get_logger(), "Total yolo time %llu ms", std::chrono::duration_cast<std::chrono::milliseconds>(yolo_r_end_time - yolo_l_start_time).count());
    RCLCPP_INFO(get_logger(), "\t - Left yolo time %llu ms | Right yolo time %llu ms", 
                std::chrono::duration_cast<std::chrono::milliseconds>(yolo_l_end_time - yolo_l_start_time).count(), 
                std::chrono::duration_cast<std::chrono::milliseconds>(yolo_r_end_time - yolo_l_end_time).count());
    RCLCPP_INFO(get_logger(), "Camera Diff L: %llu ms | Camera Diff R: %llu ms", ms_lidar_camera_diff_l, ms_lidar_camera_diff_r);
    RCLCPP_INFO(get_logger(), "Get Camera Time  %ld ms.", std::chrono::duration_cast<std::chrono::milliseconds>(camera_time - start_time).count());
    RCLCPP_INFO(get_logger(), "Total Transform and Coloring Time  %ld ms.", transform_coloring_time);
    #if save_frames
    RCLCPP_INFO(get_logger(), "Total Frame Annotation Time  %ld ms.", drawing_time);
    #endif // save_frames
    RCLCPP_INFO(get_logger(), "Total Cone Ordering Time  %ld ms. \n", ordering_time);
    auto time_diff = end_time - start_time;
    RCLCPP_INFO(get_logger(), "Total PPM Time %ld ms.", std::chrono::duration_cast<std::chrono::milliseconds>(time_diff).count());
    RCLCPP_INFO(get_logger(), "Time from Lidar to start  %ld ms.", ms_time_since_lidar_2);
    RCLCPP_INFO(get_logger(), "Time from Lidar:  %ld ms.", (uint64_t) ms_time_since_lidar);
    #endif // timing

    // Publish message and log
    int cones_published = message.orange_cones.size() + message.yellow_cones.size() + message.blue_cones.size() + message.unknown_color_cones.size();
    int yellow_cones = message.yellow_cones.size();
    int blue_cones = message.blue_cones.size();
    int orange_cones = message.orange_cones.size();
    int unknown_color_cones = message.unknown_color_cones.size();
    
    RCLCPP_INFO(get_logger(), "Published %d total cones.", cones_published);
    RCLCPP_INFO(get_logger(), "%d yellow cones, %d blue cones, %d orange cones, %d unknown color cones.\n",
    yellow_cones, blue_cones, orange_cones, unknown_color_cones);

    RCLCPP_INFO(get_logger(), "==============END=OF=CALLBACK==============");
    
    cone_pub_->publish(message);
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
                rclcpp::sleep_for(std::chrono::milliseconds(1));
            }
        }};
}
#endif

void PointToPixelNode::vel_callback(geometry_msgs::msg::TwistStamped::SharedPtr msg)
{   
    vel_mutex.lock();
    // Deque Management and Updating
    while (vel_deque.size() >= max_deque_size)
    {
        vel_deque.pop_front();
    }
    vel_deque.push_back(msg);
    vel_mutex.unlock();
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