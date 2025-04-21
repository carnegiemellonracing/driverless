#include "camera.hpp"
#include <iostream>

namespace camera {
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

        // If we didn't find a frame with timestamp >= callbackTime, return the most recent frame
        return img_deque.back();
    }

    bool initialize_camera(
        Camera &cam,
        const rclcpp::Logger &logger
    ) {
        // Initialize video capture
        if (!cam.cap.initializeVideo(cam.device_id)) {
            RCLCPP_ERROR(logger, "Cannot open camera %d video capture", cam.device_id);
            return false;
        }
        
        RCLCPP_INFO(logger, "Connected to ZED camera %d. %s", cam.device_id, cam.cap.getDeviceName().c_str());

        // Retrieve calibration file from Stereolabs server
        int sn = cam.cap.getSerialNumber();
        std::string calibration_file;
        unsigned int serial_number = sn;

        // Download camera calibration file
        if (!sl_oc::tools::downloadCalibrationFile(serial_number, calibration_file)) {
            RCLCPP_ERROR(logger, "Could not load calibration file from Stereolabs servers for Camera %d", cam.device_id);
            return false;
        }

        // Get Frame size
        int w, h;
        cam.cap.getFrameSize(w, h);
        cv::Mat cameraMatrix_left, cameraMatrix_right;

        // Initialize calibration
        sl_oc::tools::initCalibration(
            calibration_file, 
            cv::Size(w / 2, h),
            cam.map_left_x, cam.map_left_y, 
            cam.map_right_x, cam.map_right_y,
            cameraMatrix_left, cameraMatrix_right
        );

        // Set auto exposure and brightness
        cam.cap.setAECAGC(true);
        cam.cap.setAutoWhiteBalance(true);

        RCLCPP_INFO(logger, "ZED Camera %d Ready. %s", cam.device_id, cam.cap.getDeviceName().c_str());
        
        return true;
    }

    std::pair<uint64_t, cv::Mat> capture_and_rectify_frame(
        const rclcpp::Logger &logger,
        const Camera &cam,
        bool left_camera,
        bool use_inner_lens
    ) {
        // Capture the frame
        const sl_oc::video::Frame frame = cam.cap.getLastFrame();

        cv::Mat frameBGR, raw, rect;
        cv::Rect index; 
        cv::Mat map_x;
        cv::Mat map_y;

        // Set the side of the frame to capture
        if (left_camera) {
            if (use_inner_lens) {
                index = cv::Rect(frame.width / 2, 0, frame.width / 2, frame.height); // Right side of the frame
                map_x = cam.map_right_x;
                map_y = cam.map_right_y;
            } else {
                index = cv::Rect(0, 0, frame.width / 2, frame.height); // Left side of the frame
                map_x = cam.map_left_x;
                map_y = cam.map_left_y;
            }
        } else {
            if (use_inner_lens) {
                index = cv::Rect(0, 0, frame.width / 2, frame.height); // Left side of the frame
                map_x = cam.map_left_x;
                map_y = cam.map_left_y;
            } else {
                index = cv::Rect(frame.width / 2, 0, frame.width / 2, frame.height); // Right side of the frame
                map_x = cam.map_right_x;
                map_y = cam.map_right_y;
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

    void capture_freezes(
        const rclcpp::Logger &logger,
        const Camera &left_cam,
        const Camera &right_cam,
        std::mutex &l_img_mutex,
        std::mutex &r_img_mutex,
        std::deque<std::pair<uint64_t, cv::Mat>> &img_deque_l,
        std::deque<std::pair<uint64_t, cv::Mat>> &img_deque_r,
        bool use_inner_lens
    ) {
        // Capture and rectify frames for calibration from left camera
        std::pair<uint64_t, cv::Mat> frame_ll = capture_and_rectify_frame(
            logger,
            left_cam,
            true,  // left_camera==true
            false  // outer lens
        );

        if (frame_ll.second.empty()) {
            RCLCPP_ERROR(logger, "Failed to capture frame from left camera left frame.");
        }

        std::pair<uint64_t, cv::Mat> frame_lr = capture_and_rectify_frame(
            logger,
            left_cam,
            true,  // left_camera==true
            true   // inner lens
        );

        if (frame_lr.second.empty()) {
            RCLCPP_ERROR(logger, "Failed to capture frame from left camera right frame.");
        }

        // Capture and rectify frames for calibration from right camera
        std::pair<uint64_t, cv::Mat> frame_rr = capture_and_rectify_frame(
            logger,
            right_cam,
            false,  // right_camera==false
            false   // outer lens
        );

        if (frame_rr.second.empty()) {
            RCLCPP_ERROR(logger, "Failed to capture frame from right camera right frame.");
        }

        std::pair<uint64_t, cv::Mat> frame_rl = capture_and_rectify_frame(
            logger,
            right_cam,
            false,  // right_camera==false
            true    // inner lens
        );

        if (frame_rl.second.empty()) {
            RCLCPP_ERROR(logger, "Failed to capture frame from right camera left frame.");
        }

        // Save freeze images
        cv::imwrite("src/point_to_pixel/config/freeze_ll.png", frame_ll.second);
        cv::imwrite("src/point_to_pixel/config/freeze_lr.png", frame_lr.second);
        cv::imwrite("src/point_to_pixel/config/freeze_rr.png", frame_rr.second);
        cv::imwrite("src/point_to_pixel/config/freeze_rl.png", frame_rl.second);

        // Update image deque 
        l_img_mutex.lock();
        if (use_inner_lens) {
            img_deque_l.push_back(std::make_pair(frame_lr.first, frame_lr.second));
        } else {
            img_deque_l.push_back(std::make_pair(frame_ll.first, frame_ll.second));
        }
        l_img_mutex.unlock();

        r_img_mutex.lock();
        if (use_inner_lens) {
            img_deque_r.push_back(std::make_pair(frame_rl.first, frame_rl.second));
        } else {
            img_deque_r.push_back(std::make_pair(frame_rr.first, frame_rr.second));
        }
        r_img_mutex.unlock();
    }
}