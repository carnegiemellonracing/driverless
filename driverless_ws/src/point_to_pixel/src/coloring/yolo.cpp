#include "yolo.hpp"
#include <vector>

namespace coloring {
    namespace yolo {
        std::pair<int, double> get_color(
            Eigen::Vector3d& pixel,
            float *detection,
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

            // Loop through all detections (assuming num_detections * 85 values per detection)
            const int num_detections = 25200;  // Typically, this will be the number of predictions (e.g., 25200 for a 640x640 input)
            const int detection_size = 10;  // Number of values per detection (e.g., 4 bbox values + 1 confidence + 5 class scores)

            for (int i = 0; i < num_detections; ++i) {
                // Get confidence value
                float confidence = detection[i * detection_size + 4];

                if (confidence >= confidence_threshold) {
                    // Get bounding box coordinates (cx, cy, w, h)
                    float cx = detection[i * detection_size + 0] * x_factor;
                    float cy = detection[i * detection_size + 1] * y_factor;
                    float width = detection[i * detection_size + 2] * x_factor;
                    float height = detection[i * detection_size + 3] * y_factor;

                    // Calculate the bounding box corners
                    float left = cx - width / 2;
                    float top = cy - height / 2;
                    float right = cx + width / 2;
                    float bottom = cy + height / 2;

                    // If pixel is inside the bounding box
                    if (left <= x && x <= right && top <= y && y <= bottom) {
                        // Find the highest class score
                        double max_class_score = 0;
                        int class_id = -1;

                        // Assuming class scores start after index 5 (since indices 0-4 are bbox and confidence)
                        for (int j = 5; j < detection_size; ++j) {
                            float class_score = detection[i * detection_size + j];
                            if (class_score > max_class_score) {
                                max_class_score = class_score;
                                class_id = j - 5;  // Adjust index to get the actual class ID
                            }
                        }

                        // Map class_id to cone color
                        int cone_color = -1;
                        if (class_id >= 0) {
                            switch (class_id) {
                                case 0: cone_color = 2; break;  // Orange cone
                                case 1: cone_color = 1; break;  // Yellow cone
                                case 2: cone_color = 0; break;  // Blue cone
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

        std::pair<int, double> draw_bounding_boxes(
            cv::Mat& frame,
            float *detection,
            int cols,
            int rows,
            double confidence_threshold
        ) {
            static constexpr int yolo_h = 640;
            static constexpr int yolo_w = 640;

            // Initialize vectors to hold respective outputs while unwrapping detections.
            std::vector<int> class_ids;
            std::vector<float> confidences;
            std::vector<cv::Rect> boxes;
            // Resizing factor.
            float x_factor = cols / yolo_w;
            float y_factor = rows / yolo_h;
            const int dimensions = 10;
            // 25200 for default size 640.
            const int rows_ = 25200;
            // Iterate through 25200 detections.
            for (int i = 0; i < rows; ++i)
            {
                float confidence = detection[4];
                // Discard bad detections and continue.
                if (confidence >= confidence_threshold)
                {
                    float * classes_scores = detection + 5;
                    // Create a 1x85 Mat and store class scores of 80 classes.
                    cv::Mat scores(1, 4, CV_32FC1, classes_scores);
                    // Perform minMaxLoc and acquire the index of best class  score.
                    cv::Point class_id;
                    double max_class_score;
                    cv::minMaxLoc(scores, 0, &max_class_score, 0, &class_id);
                    // Continue if the class score is above the threshold.
                    if (max_class_score > confidence_threshold)
                    {
                        // Store class ID and confidence in the pre-defined respective vectors.
                        confidences.push_back(confidence);
                        class_ids.push_back(class_id.x);
                        // Center.
                        float cx = detection[0];
                        float cy = detection[1];
                        // Box dimension.
                        float w = detection[2];
                        float h = detection[3];
                        // Bounding box coordinates.
                        int left = int((cx * x_factor - 0.5 * w)); // * x_factor);
                        int top = int((cy*y_factor - 0.5 * h)); // * y_factor);
                        int width = int(w * x_factor);
                        int height = int(h * y_factor);
                        // Store good detections in the boxes vector.
                        cv::rectangle(frame, cv::Rect(left, top, width, height), cv::Scalar(0, 255, 0), cv::FILLED);
                    }
                }
                // Jump to the next row.
                detection += dimensions;
            }

            // No detection
            return std::make_pair(-1, 0.0);
        }

        // std::pair<int, double> get_color(
        //     Eigen::Vector3d& pixel,
        //     float *detection,
        //     int cols,
        //     int rows,
        //     double confidence_threshold
        // ) {
        //     static constexpr int yolo_h = 640;
        //     static constexpr int yolo_w = 640;
            
        //     int x = static_cast<int>(pixel(0));
        //     int y = static_cast<int>(pixel(1));

        //     // Post Processing
        //     const float x_factor = cols / static_cast<float>(yolo_w);
        //     const float y_factor = rows / static_cast<float>(yolo_h);

        //     // Loop through all detection
        //     for (int i = 0; i < detection.rows; ++i) {
        //         double confidence = detection.at<double>(i, 4);

        //         if (confidence >= confidence_threshold) {
        //             // Get bounding box coordinates
        //             float cx = detection.at<float>(i, 0) * x_factor;
        //             float cy = detection.at<float>(i, 1) * y_factor;
        //             float width = detection.at<float>(i, 2) * x_factor;
        //             float height = detection.at<float>(i, 3) * y_factor;
                    
        //             // Calculate the bounding box corners
        //             float left = cx - width/2;
        //             float top = cy - height/2;
        //             float right = cx + width/2;
        //             float bottom = cy + height/2;
                    
        //             // If pixel is inside the bounding box
        //             if (left <= x && x <= right && top <= y && y <= bottom) {
        //                 // Find the highest class score
        //                 double max_class_score = 0;
        //                 int class_id = -1;
                        
        //                 // Assuming class scores start after index 5
        //                 for (int j = 5; j < detection.cols; ++j) {
        //                     double class_score = detection.at<float>(i, j);
        //                     if (class_score > max_class_score) {
        //                         max_class_score = class_score;
        //                         class_id = j - 5;  // Adjust index to get the actual class ID
        //                     }
        //                 }
                        
        //                 // Map class_id to cone color
        //                 int cone_color = -1;
        //                 if (class_id >= 0) {
        //                     switch(class_id) {
        //                         case 0: cone_color = 0; break;  // Orange cone
        //                         case 1: cone_color = 1; break;  // Yellow cone
        //                         case 2: cone_color = 2; break;  // Blue cone
        //                         default: cone_color = -1; break; // Unknown
        //                     }
        //                     return std::make_pair(cone_color, confidence);
        //                 }
        //             }
        //         }
        //     }
            
        //     // No detection 
        //     return std::make_pair(-1, 0.0);
        // }

        cv::dnn::Net init_model(const std::string& model_path) {
            cv::dnn::Net net = cv::dnn::readNetFromONNX(model_path);
            
            // Set CUDA On if available
            if (!net.empty()) {
                net.setPreferableBackend(cv::dnn::DNN_BACKEND_CUDA);
                net.setPreferableTarget(cv::dnn::DNN_TARGET_CUDA);
            } else {
                std::cerr << "Failed to load YOLO model" << std::endl;
            }
            
            return net;
        }

        float *process_frame(
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

            std::cout << "Output dimensions: ";
            for (int i = 0; i < outputs[0].dims; ++i) {
                std::cout << outputs[0].size[i] << " ";
            }
            std::cout << std::endl;

            // Return the detection results
            return (float*)outputs[0].data;
        }
    }
}