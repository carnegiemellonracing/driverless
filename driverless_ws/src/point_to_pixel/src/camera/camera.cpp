#include "camera.hpp"
#include <iostream>

namespace camera {
    stamped_frame Camera::find_closest_frame(
        const rclcpp::Time &callbackTime,
        const rclcpp::Logger &logger
    ) {
        img_mutex.lock();
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
        stamped_frame last_frame = img_deque.back();
        img_mutex.unlock();

        return last_frame;
    }

    bool Camera::initialize_camera(
        const rclcpp::Logger &logger
    ) {
        // Camera reasignment 
        std::map<int, std::pair<uint16_t, uint16_t>> device_info;
        
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
        if (this->device_id == 0) {
            // This should be the left camera (ZED)
            correct_device_id = zed_device_id;
        } else {
            // This should be the right camera (ZED 2)
            correct_device_id = zed2_device_id;
        }
        
        // If incorrect, reassign correct device ID to cam object, but log first.
        if (correct_device_id != this->device_id) {
            RCLCPP_INFO(logger, "Reassigning camera %d to use device ID %d",
                    this->device_id, correct_device_id);
        }
        
        // Initialize video capture with the correct device ID
        if (!this->cap.initializeVideo(correct_device_id)) {
            RCLCPP_ERROR(logger, "Cannot open camera %d video capture", correct_device_id);
            return false;
        }
        if (this->device_id == 0) {
        RCLCPP_INFO(logger, "Connected to left ZED camera. %s", 
                  this->cap.getDeviceName().c_str());
        } else {
            RCLCPP_INFO(logger, "Connected to right ZED 2 camera. %s", 
                        this->cap.getDeviceName().c_str());
        }
        
        // Assign correct device ID
        this->device_id = correct_device_id;

        // Camera Rectification
        int sn = this->cap.getSerialNumber();
        std::string calibration_file;
        unsigned int serial_number = sn;

        // Download camera calibration file
        if (!sl_oc::tools::downloadCalibrationFile(serial_number, calibration_file)) {
            RCLCPP_ERROR(logger, "Could not load calibration file from Stereolabs servers for Camera %d", this->device_id);
            return false;
        }

        // Get Frame size
        int w, h;
        this->cap.getFrameSize(w, h);
        cv::Mat cameraMatrix_left, cameraMatrix_right;

        // Initialize calibration
        sl_oc::tools::initCalibration(
            calibration_file, 
            cv::Size(w / 2, h),
            this->map_left_x, this->map_left_y, 
            this->map_right_x, this->map_right_y,
            cameraMatrix_left, cameraMatrix_right
        );

        // Set auto exposure and brightness
        this->cap.setAECAGC(true);
        this->cap.setAutoWhiteBalance(true);

        RCLCPP_INFO(logger, "ZED Camera %d Ready. %s \n", this->device_id, this->cap.getDeviceName().c_str());
        
        return true;
    }

    stamped_frame Camera::capture_and_rectify_frame(
        const rclcpp::Logger &logger,
        bool is_left_camera,
        bool use_inner_lens)
    {
        // Capture the frame
        const sl_oc::video::Frame frame = this->cap.getLastFrame();

        cv::Mat frameBGR, raw, rect;
        cv::Rect index; 
        cv::Mat map_x;
        cv::Mat map_y;

        // Set the side of the frame to capture
        if (is_left_camera) {
            if (use_inner_lens) {
                index = cv::Rect(frame.width / 2, 0, frame.width / 2, frame.height); // Right side of the frame
                map_x = this->map_right_x;
                map_y = this->map_right_y;
            } else {
                index = cv::Rect(0, 0, frame.width / 2, frame.height); // Left side of the frame
                map_x = this->map_left_x;
                map_y = this->map_left_y;
            }
        } else {
            if (use_inner_lens) {
                index = cv::Rect(0, 0, frame.width / 2, frame.height); // Left side of the frame
                map_x = this->map_left_x;
                map_y = this->map_left_y;
            } else {
                index = cv::Rect(frame.width / 2, 0, frame.width / 2, frame.height); // Right side of the frame
                map_x = this->map_right_x;
                map_y = this->map_right_y;
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

    void Camera::capture_freezes(
        const rclcpp::Logger &logger,
        bool is_left_camera,
        bool use_inner_lens
    ) {
        // Capture and rectify frames for calibration from left camera
        stamped_frame frame_l = capture_and_rectify_frame(
            logger,
            is_left_camera, 
            false  // outer lens
        );

        stamped_frame frame_r = capture_and_rectify_frame(
            logger,
            is_left_camera,
            true   // inner lens
        );

        if (frame_l.second.empty()) {
            RCLCPP_ERROR(logger, "Failed to capture frame from %s camera left frame." , is_left_camera ? "left" : "right");
        } else if (frame_r.second.empty()) {
            RCLCPP_ERROR(logger, "Failed to capture frame from %s camera right frame.", is_left_camera ? "left" : "right");
        } else {
            std::string camera_char = is_left_camera ? "l" : "r";
            std::string freeze_l_path = save_path + camera_char + "l" + ".bmp";
            std::string freeze_r_path = save_path + camera_char + "r" + ".bmp";

            cv::imwrite(freeze_l_path, frame_l.second);
            cv::imwrite(freeze_r_path, frame_r.second);
        }

        // Update image deque 
        img_mutex.lock();
        if(use_inner_lens) {
            img_deque.push_back(std::make_pair(frame_l.first, frame_l.second));
        } else {
            img_deque.push_back(std::make_pair(frame_r.first, frame_r.second));
        }
        img_mutex.unlock();
    }


    void Camera::update_deque(stamped_frame new_frame, int max_deque_size) {
        img_mutex.lock();
        while (img_deque.size() >= max_deque_size) {
            img_deque.pop_front();
        }
        img_deque.push_back(new_frame);
        img_mutex.unlock();
    }
}