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
        // Get list of video devices
        std::map<int, std::pair<uint16_t, uint16_t>> device_info;
        
        // Check first 5 video devices
        for (int device_id = 0; device_id < 10; device_id++) {
            try {
                std::string sysfs_path = "/sys/class/video4linux/video" + std::to_string(device_id);
                if (!std::filesystem::exists(sysfs_path)) continue; 
                
                std::filesystem::path current_path = std::filesystem::path(sysfs_path) / "device";
                if (!std::filesystem::exists(current_path)) continue;
                
                // Find USB info using vendor and product IDs
                bool found_usb = false;
                for (int i = 0; i < 4 && !found_usb; i++) {
                    if (current_path.string().find("/usb") != std::string::npos) {
                        std::filesystem::path vendor_path = current_path / "idVendor";
                        std::filesystem::path product_path = current_path / "idProduct";
                        
                        if (std::filesystem::exists(vendor_path) && std::filesystem::exists(product_path)) {
                            std::ifstream vendor_file(vendor_path);
                            std::ifstream product_file(product_path);
                            std::string vendor_id, product_id;
                            
                            // Check if file streams are valid and if reads were successful
                            if (vendor_file >> vendor_id && product_file >> product_id) {
                                uint16_t vid = std::stoi(vendor_id, nullptr, 16);
                                uint16_t pid = std::stoi(product_id, nullptr, 16);
                                
                                // Save device info
                                device_info[device_id] = {vid, pid};
                                
                                // Check for ZED cameras
                                if (vid == ZED_VENDOR_ID && (pid == ZED_PRODUCT_ID || pid == ZED2_PRODUCT_ID)) {
                                    found_usb = true;
                                }
                            }
                        }
                        break;
                    }
                    
                    // Move back to parent directory
                    std::filesystem::path parent_path = current_path / "..";
                    if (!std::filesystem::exists(parent_path)) break;
                    
                    current_path = std::filesystem::canonical(parent_path);
                }
            } catch (const std::exception& e) {
                // Gracefully skip if there is error
                continue;
            }
        }
        
        // Lookup ZED device ids
        int zed_device_id = -1;   // Original ZED (left camera)
        int zed2_device_id = -1;  // ZED 2 (right camera)
        
        for (const auto& [dev_id, ids] : device_info) {
            uint16_t vid = ids.first;
            uint16_t pid = ids.second;
            
            if (vid == ZED_VENDOR_ID) {
                if (pid == ZED_PRODUCT_ID && dev_id % 2 == 0) { // Ensure camera ids are either 0 or 2
                    zed_device_id = dev_id;
                    RCLCPP_INFO(logger, "Found ZED (left camera) at /dev/video%d", dev_id);
                } else if (pid == ZED2_PRODUCT_ID && dev_id % 2 == 0) { // Ensure camera ids are either 0 or 2
                    zed2_device_id = dev_id;
                    RCLCPP_INFO(logger, "Found ZED 2 (right camera) at /dev/video%d", dev_id);
                }
            }
        }
        
        // Early return if either camera doesn't exist
        if (zed_device_id == -1 || zed2_device_id == -1) {
            RCLCPP_ERROR(logger, "Could not find both ZED cameras. ZED: %d, ZED 2: %d", 
                    zed_device_id, zed2_device_id);
            return false;
        }
        
        // Check if assigned device ID matches ZED device id
        int correct_device_id;
        if (cam.device_id == 0) {
            // This should be the left camera (ZED)
            correct_device_id = zed_device_id;
        } else {
            // This should be the right camera (ZED 2)
            correct_device_id = zed2_device_id;
        }
        
        // If incorrect, reassign correct device ID to cam object, but log first.
        if (correct_device_id != cam.device_id) {
            RCLCPP_INFO(logger, "Reassigning camera %d to use device ID %d",
                    cam.device_id, correct_device_id);
        }
        
        // Initialize video capture with the correct device ID
        if (!cam.cap.initializeVideo(correct_device_id)) {
            RCLCPP_ERROR(logger, "Cannot open camera %d video capture", correct_device_id);
            return false;
        }
        if (cam.device_id == 0) {
        RCLCPP_INFO(logger, "Connected to left ZED camera. %s", 
                  cam.cap.getDeviceName().c_str());
        } else {
            RCLCPP_INFO(logger, "Connected to right ZED 2 camera. %s", 
                        cam.cap.getDeviceName().c_str());
        }
        
        // Assign correct device ID
        cam.device_id = correct_device_id;

        // Camera Rectification
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

        RCLCPP_INFO(logger, "ZED Camera %d Ready. %s \n", cam.device_id, cam.cap.getDeviceName().c_str());
        
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
            
            if (map_x.empty() || map_y.empty()) {
                RCLCPP_ERROR(logger, "Mapping matrices are empty for camera.");
                return std::make_pair(frame.timestamp, raw.clone());
            }

            // Type check for rectification matrices
            cv::Mat map_x_float, map_y_float;
    
            if (map_x.type() != CV_32F) {
                map_x.convertTo(map_x_float, CV_32F);
            } else {
                map_x_float = map_x;
            }
            
            if (map_y.type() != CV_32F) {
                map_y.convertTo(map_y_float, CV_32F);
            } else {
                map_y_float = map_y;
            }


            // Apply rectification
            cv::remap(raw, rect, map_x_float, map_y_float, cv::INTER_LINEAR);
            
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