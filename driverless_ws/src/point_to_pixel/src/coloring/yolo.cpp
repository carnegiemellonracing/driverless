#include "yolo.hpp"
#include <vector>

namespace coloring {
    namespace yolo {
        std::pair<int, double> get_color(
            Eigen::Vector3d& pixel,
            cv::Mat detection,
            int cols,
            int rows,
            double confidence_threshold
        ) {
            static constexpr int yolo_h = 640;
            static constexpr int yolo_w = 640;
            
            int x = static_cast<int>(pixel(0));
            int y = static_cast<int>(pixel(1));

            // Post Processing
            const float x_factor = cols / static_cast<float>(yolo_w);
            const float y_factor = rows / static_cast<float>(yolo_h);

            // Loop through all detection
            for (int i = 0; i < detection.rows; ++i) {
                double confidence = detection.at<double>(i, 4);

                if (confidence >= confidence_threshold) {
                    // Get bounding box coordinates
                    float cx = detection.at<float>(i, 0) * x_factor;
                    float cy = detection.at<float>(i, 1) * y_factor;
                    float width = detection.at<float>(i, 2) * x_factor;
                    float height = detection.at<float>(i, 3) * y_factor;
                    
                    // Calculate the bounding box corners
                    float left = cx - width/2;
                    float top = cy - height/2;
                    float right = cx + width/2;
                    float bottom = cy + height/2;
                    
                    // If pixel is inside the bounding box
                    if (left <= x && x <= right && top <= y && y <= bottom) {
                        // Find the highest class score
                        double max_class_score = 0;
                        int class_id = -1;
                        
                        // Assuming class scores start after index 5
                        for (int j = 5; j < detection.cols; ++j) {
                            double class_score = detection.at<float>(i, j);
                            if (class_score > max_class_score) {
                                max_class_score = class_score;
                                class_id = j - 5;  // Adjust index to get the actual class ID
                            }
                        }
                        
                        // Map class_id to cone color
                        int cone_color = -1;
                        if (class_id >= 0) {
                            switch(class_id) {
                                case 0: cone_color = 0; break;  // Orange cone
                                case 1: cone_color = 1; break;  // Yellow cone
                                case 2: cone_color = 2; break;  // Blue cone
                                default: cone_color = -1; break; // Unknown
                            }
                            return std::make_pair(cone_color, confidence);
                        }
                    }
                }
            }
            
            // No detection 
            return std::make_pair(-1, 0.0);
        }

        cv::dnn::Net init_model(const std::string& model_path) {
            cv::dnn::Net net = cv::dnn::readNetFromONNX(model_path);
            
            // Set CUDA On if available
            if (!net.empty()) {
                net.setPreferableBackend(cv::dnn::DNN_BACKEND_CUDA);
                net.setPreferableTarget(cv::dnn::DNN_TARGET_CUDA);
            }
            
            return net;
        }

        cv::Mat process_frame(
            const cv::Mat& frame,
            cv::dnn::Net& net,
            int yolo_width,
            int yolo_height
        ) {
            // Create a blob from the image
            cv::Mat blob;
            cv::dnn::blobFromImage(
                frame, 
                blob, 
                1 / 255.0, 
                cv::Size(yolo_width, yolo_height), 
                cv::Scalar(), 
                true, 
                false
            );
            
            // Forward pass through the network
            net.setInput(blob);
            std::vector<cv::Mat> outputs;
            net.forward(outputs, net.getUnconnectedOutLayersNames());
            
            // Return the detection results
            return outputs[0];
        }
    }
}