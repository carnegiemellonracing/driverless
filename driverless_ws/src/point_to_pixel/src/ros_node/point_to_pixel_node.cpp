#include "point_to_pixel_node.hpp"

// Standard Imports
#include <deque>
#include <memory>
#include <chrono>

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

    // Initialize Empty IMG Deque
    img_deque_l = {};
    img_deque_r = {};

    // Boolean parameter that determines whether we use the inner two zed cameras or the outer two zed cameras
    declare_parameter("inner", false);

    // Projection matrix that takes LiDAR points to pixels
    std::vector<double> param_default(12, 1.0f); 

    declare_parameter("projection_matrix_l", param_default);
    declare_parameter("projection_matrix_r", param_default);

    // Threshold that determines whether it reports the color on a cone or not
    declare_parameter("confidence_threshold", 0.05);

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

    // Load inner boolean parameter
    bool inner = get_parameter("camera_inner").as_bool();

    // Load Projection Matrix if inner is set to true, then load lr and rl, else load ll and rr
    std::vector<double> param_l, param_r;
    
    if (inner) {
        param_l = get_parameter("projection_matrix_lr").as_double_array();
        param_r = get_parameter("projection_matrix_rl").as_double_array();
    } else {
        param_l = get_parameter("projection_matrix_ll").as_double_array();
        param_r = get_parameter("projection_matrix_rr").as_double_array();
    }
    
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
    publisher_ = create_publisher<interfaces::msg::ConeArray>("/perc_cones", 10);
    
    // Subscriber that reads the input topic that contains an array of cone_point arrays from LiDAR stack
    subscriber_ = create_subscription<interfaces::msg::PPMConeArray>(
        "/cpp_cones", 
        10, 
        [this](const interfaces::msg::PPMConeArray::SharedPtr msg) {topic_callback(msg);}
    );

    // Camera Callback (25 fps)
    camera_timer_ = create_wall_timer(
        std::chrono::milliseconds(40),
        [this](){camera_callback();}
    );

    // ---------------------------------------------------------------------------
    //                       INITIALIZATION COMPLETE SEQUENCE
    // ---------------------------------------------------------------------------
    
    // Capture and rectify frames for calibration
    cv::Mat frame_l = capture_and_rectify_frame(
        cap_l,
        map_left_x_ll,
        map_left_y_ll,
        map_right_x_lr,
        map_right_y_lr,
        true, // left_camera==true;
        inner
    );

    if (frame_l.empty()) {
        RCLCPP_ERROR(get_logger(), "Failed to capture frame from left camera.");
    };

    cv::Mat frame_r = capture_and_rectify_frame(
        cap_r,
        map_left_x_rl,
        map_left_y_rl,
        map_right_x_rr,
        map_right_y_rr,
        false, // left_camera==false;
        inner
    );

    if (frame_r.empty()) {
        RCLCPP_ERROR(get_logger(), "Failed to capture frame from right camera.");
    };

    // Save freeze images
    cv::imwrite("src/point_to_pixel/config/freeze_l.png", frame_l);
    cv::imwrite("src/point_to_pixel/config/freeze_r.png", frame_r);

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
std::pair<cv::Mat, cv::Mat> PointToPixelNode::get_camera_frame(rclcpp::Time callbackTime)
{
    // Get closest frame from each camera
    cv::Mat closestFrame_l = find_closest_frame(img_deque_l, callbackTime, get_logger());
    cv::Mat closestFrame_r = find_closest_frame(img_deque_r, callbackTime, get_logger());
    
    return std::make_pair(closestFrame_l, closestFrame_r);
}

// Get color of cone by combining output from both cameras
int PointToPixelNode::get_cone_class(
    std::pair<Eigen::Vector3d, Eigen::Vector3d> pixel_pair,
    std::pair<cv::Mat, cv::Mat> frame_pair,
    std::pair<cv::Mat, cv::Mat> detection_pair
) {
    return coloring::get_cone_class(
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
void PointToPixelNode::topic_callback(const interfaces::msg::PPMConeArray::SharedPtr msg)
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
    message.orig_data_stamp = msg->header.stamp; 
    message.blue_cones = std::vector<geometry_msgs::msg::Point> {};
    message.yellow_cones = std::vector<geometry_msgs::msg::Point> {};
    message.orange_cones = std::vector<geometry_msgs::msg::Point> {};
    message.unknown_color_cones = std::vector<geometry_msgs::msg::Point> {};
    geometry_msgs::msg::Point point_msg;

    // Retrieve Camera Frame
    std::pair<cv::Mat, cv::Mat> frame_pair = get_camera_frame(msg->header.stamp);

    #if timing
        auto camera_time = high_resolution_clock::now();
    #endif

    #if use_yolo
        // Process frames with YOLO
        cv::Mat detection_l = coloring::yolo::process_frame(frame_pair.first, net);
        cv::Mat detection_r = coloring::yolo::process_frame(frame_pair.second, net);
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

    // Iterate through all points in /cpp_cones message
    for (int i = 0; i < msg->cone_array.size(); i++) {
        #if timing
            auto loop_start = high_resolution_clock::now();
        #endif
          
        // Transform Point
        std::pair<Eigen::Vector3d, Eigen::Vector3d> pixel_pair = transform_point(
            msg->cone_array[i].cone_points[0],
            projection_matrix_l,
            projection_matrix_r
        );

        #if timing
            // Time for transform
            auto loop_transform = high_resolution_clock::now();
            transform_time = transform_time + std::chrono::duration_cast<std::chrono::microseconds>(loop_transform - loop_start).count();
        #endif

        int cone_class = get_cone_class(pixel_pair, frame_pair, detection_pair);

        #if timing
            // Time for coloring
            auto loop_coloring = high_resolution_clock::now();
            coloring_time = coloring_time + std::chrono::duration_cast<std::chrono::microseconds>(loop_coloring - loop_transform).count();
        #endif

        point_msg.x = msg->cone_array[i].cone_points[0].x;
        point_msg.y = msg->cone_array[i].cone_points[0].y;
        point_msg.z = 0.0;
      
        #if viz
            RCLCPP_INFO(get_logger(), "Cone: Color %d, 2D[ l: (%lf, %lf) | r: (%lf, %lf) ] from 3D[ (%lf, %lfl, %lf)",
                    cone_class, pixel_pair.first[0], pixel_pair.first[1], pixel_pair.second[0], pixel_pair.second[1],
                    msg->cone_array[i].cone_points[0].x, msg->cone_array[i].cone_points[0].y, msg->cone_array[i].cone_points[0].z);
        #endif

        switch (cone_class) {
            case 0:
                message.orange_cones.push_back(point_msg);
                break;
            case 1:
                message.yellow_cones.push_back(point_msg);
                break;
            case 2:
                message.blue_cones.push_back(point_msg);
                break;
            default:
                message.unknown_color_cones.push_back(point_msg);
                break;
        }
    }

    #if timing
        auto transform_coloring_time = high_resolution_clock::now();
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
        auto time_diff = end_time - start_time;
        RCLCPP_INFO(get_logger(), "Total PPM Time %ld microseconds.", std::chrono::duration_cast<std::chrono::microseconds>(time_diff).count());
        RCLCPP_INFO(get_logger(), "Total Time from Lidar  %ld microseconds.", ms_time_since_lidar.nanoseconds() / 1000);
        RCLCPP_INFO(get_logger(), "Total Time from Lidar to start  %ld microseconds.", ms_time_since_lidar_2);
    #endif
    
    publisher_->publish(message);
}

// Camera Callback (Populates and maintain deque)
void PointToPixelNode::camera_callback()
{
    // Capture and rectify frame from camera l
    cv::Mat frameBGR_l = capture_and_rectify_frame(
        cap_l,
        map_left_x_ll,
        map_left_y_ll,
        map_right_x_lr,
        map_right_y_lr,
        true, // left_camera==true
        inner
    );

    rclcpp::Time time_l = get_clock()->now();

    // Capture and rectify frame from camera r
    cv::Mat frameBGR_r = capture_and_rectify_frame(
        cap_r,
        map_left_x_rl,
        map_left_y_rl,
        map_right_x_rr,
        map_right_y_rr,
        false, // left_camera==false
        inner
    );
    rclcpp::Time time_r = get_clock()->now();

    // Deque Management and Updating
    while (img_deque_l.size() >= max_deque_size) {
        img_deque_l.pop_front();
    }

    while (img_deque_r.size() >= max_deque_size) {
        img_deque_r.pop_front();
    }

    img_deque_l.push_back(std::make_pair(time_l, frameBGR_l));
    img_deque_r.push_back(std::make_pair(time_r, frameBGR_r));
}

int main(int argc, char** argv)
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