#include "camera.hpp"
#include <cmath>

cv::Mat find_closest_frame(
    const std::deque<std::pair<rclcpp::Time, cv::Mat>>& img_deque,
    const rclcpp::Time& callbackTime,
    const rclcpp::Logger& logger
) {
    // Initialize variables
    int64_t bestDiff = INT64_MAX;
    cv::Mat closestFrame;

    // Check if deque empty
    if (img_deque.empty()) {
        RCLCPP_ERROR(logger, "Image deque is empty! Cannot find matching frame.");
        return cv::Mat();
    }

    // Iterate through deque to find the closest frame by timestamp
    for (const auto &frame : img_deque) {
        int64_t timeDiff = std::abs(frame.first.nanoseconds() - callbackTime.nanoseconds());

        if (timeDiff < bestDiff) {
            closestFrame = frame.second;
            bestDiff = timeDiff;
        }
    }

    return closestFrame;
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
    // cap.setAutoWhiteBalance(true);

    RCLCPP_INFO(logger, "ZED Camera %d Ready. %s", device_id, cap.getDeviceName().c_str());
    
    return true;
}

cv::Mat capture_and_rectify_frame(
    sl_oc::video::VideoCapture& cap,
    const cv::Mat& map_left_x,
    const cv::Mat& map_left_y,
    const cv::Mat& map_right_x,
    const cv::Mat& map_right_y,
    bool use_inner_lens,
    const rclcpp::Logger& logger
) {
    // Capture the frame
    const sl_oc::video::Frame frame = cap.getLastFrame();
    cv::Mat frameBGR, left_raw, left_rect;
    
    if (frame.data != nullptr) {
        cv::Mat frameYUV = cv::Mat(frame.height, frame.width, CV_8UC2, frame.data);
        cv::cvtColor(frameYUV, frameBGR, cv::COLOR_YUV2BGR_YUYV);
        
        // Extract left image from side-by-side
        left_raw = frameBGR(cv::Rect(0, 0, frameBGR.cols / 2, frameBGR.rows));
        
        // Apply rectification
        cv::remap(left_raw, left_rect, map_left_x, map_left_y, cv::INTER_LINEAR);
        
        return left_rect;
    } else {
        RCLCPP_ERROR(logger, "Failed to capture frame from camera.");
        return cv::Mat();
    }
}