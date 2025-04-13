#include "camera.hpp"
#include <iostream>

std::pair<uint64_t, cv::Mat> find_closest_frame(
    const std::deque<std::pair<uint64_t, cv::Mat>> &img_deque,
    const rclcpp::Time &callbackTime,
    const rclcpp::Logger &logger
) {

    // Check if deque empty
    if (img_deque.empty()) {
        RCLCPP_ERROR(logger, "Image deque is empty! Cannot find matching frame.");
        return std::make_pair(uint64_t(0), cv::Mat());
    }

    // Iterate through deque to find the closest frame by timestamp
    for (const auto &frame : img_deque) {
        if (frame.first >= callbackTime.nanoseconds()) {
            return frame;
        }
    }

    return img_deque.back();

    RCLCPP_ERROR(logger, "Callback time out of range! Cannot find matching frame.");
    return std::make_pair(uint64_t(0), cv::Mat());
}

bool initialize_camera(
    sl_oc::video::VideoCapture& cap,
    int device_id,
    cv::Mat& map_left_x,
    cv::Mat& map_left_y,
    cv::Mat& map_right_x,
    cv::Mat& map_right_y,
    const rclcpp::Logger& logger
) {
    // Initialize video capture
    if (!cap.initializeVideo(device_id)) {
        RCLCPP_ERROR(logger, "Cannot open camera %d video capture", device_id);
        return false;
    }
    
    RCLCPP_INFO(logger, "Connected to ZED camera %d. %s", device_id, cap.getDeviceName().c_str());

    // Retrieve calibration file from Stereolabs server
    int sn = cap.getSerialNumber();
    std::string calibration_file;
    unsigned int serial_number = sn;

    // Download camera calibration file
    if (!sl_oc::tools::downloadCalibrationFile(serial_number, calibration_file)) {
        RCLCPP_ERROR(logger, "Could not load calibration file from Stereolabs servers for Camera %d", device_id);
        return false;
    }

    // Get Frame size
    int w, h;
    cap.getFrameSize(w, h);
    cv::Mat cameraMatrix_left, cameraMatrix_right;

    // Initialize calibration
    sl_oc::tools::initCalibration(
        calibration_file, 
        cv::Size(w / 2, h),
        map_left_x, map_left_y, 
        map_right_x, map_right_y,
        cameraMatrix_left, cameraMatrix_right
    );

    // Set auto exposure and brightness
    cap.setAECAGC(true);
    cap.setAutoWhiteBalance(true);

    RCLCPP_INFO(logger, "ZED Camera %d Ready. %s", device_id, cap.getDeviceName().c_str());
    
    return true;
}

std::pair<uint64_t, cv::Mat> capture_and_rectify_frame(
    const rclcpp::Logger &logger,
    sl_oc::video::VideoCapture& cap,
    const cv::Mat& map_left_x,
    const cv::Mat& map_left_y,
    const cv::Mat& map_right_x,
    const cv::Mat& map_right_y,
    bool left_camera,
    bool use_inner_lens
) {
    // Capture the frame
    // uint64_t systime = std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::high_resolution_clock::now().time_since_epoch()).count();
    const sl_oc::video::Frame frame = cap.getLastFrame();

    cv::Mat frameBGR, raw, rect;
    cv::Rect index; 
    cv::Mat map_x;
    cv::Mat map_y;

    // Set the side of the frame to capture
    if (left_camera) {
        if (use_inner_lens) {
            index = cv::Rect(frame.width / 2, 0, frame.width / 2, frame.height); // Right side of the frame
            map_x = map_right_x;
            map_y = map_right_y;
        } else {
            index = cv::Rect(0, 0, frame.width / 2, frame.height); // Left side of the frame
            map_x = map_left_x;
            map_y = map_left_y;
        }
    } else {
        if (use_inner_lens) {
            index = cv::Rect(0, 0, frame.width / 2, frame.height); // Left side of the frame
            map_x = map_left_x;
            map_y = map_left_y;
        } else {
            index = cv::Rect(frame.width / 2, 0, frame.width / 2, frame.height); // Right side of the frame
            map_x = map_right_x;
            map_y = map_right_y;
        }
    }

    if (frame.data != nullptr) {
        cv::Mat frameYUV = cv::Mat(frame.height, frame.width, CV_8UC2, frame.data);
        cv::cvtColor(frameYUV, frameBGR, cv::COLOR_YUV2BGR_YUYV);
        
        // Extract relevant part of the image from side-by-side
        raw = frameBGR(cv::Rect(index));
        
        // Apply rectification
        cv::remap(raw, rect, map_x, map_y, cv::INTER_LINEAR);
        
        return std::make_pair(frame.timestamp, rect);
    } else {
        return std::make_pair(frame.timestamp, cv::Mat());
    }
}