#include "yolo.hpp"
#include <vector>

namespace coloring {
    namespace yolo {
        std::pair<int, double> get_color(
            Eigen::Vector3d& pixel,
            std::vector<cv::Mat> detections,
            int cols,
            int rows,
            double confidence_threshold
        ) {
            float* data = (float*)detections[0].data;
            int num_detections = detections[0].size[1];  // 25200
            int attributes = detections[0].size[2];      // 10

            float x_scale = static_cast<float>(cols) / 640.0f;
            float y_scale = static_cast<float>(rows) / 640.0f;
            
            for (int i = 0; i < num_detections; i++) {
                float confidence = data[i * attributes + 4];
                if (confidence > confidence_threshold) {
                    float cx = data[i * attributes];
                    float cy = data[i * attributes + 1];
                    float w = data[i * attributes + 2];
                    float h = data[i * attributes + 3];

                    // Convert center coordinates to top-left corner (x, y)
                    int x = static_cast<int>((cx - w / 2) * x_scale);
                    int y = static_cast<int>((cy - h / 2) * y_scale);
                    int width = static_cast<int>(w * x_scale);
                    int height = static_cast<int>(h * y_scale);

                    // If pixel is inside the bounding box color it
                    if (pixel(0) > x && pixel(0) < x + width && pixel(1) > y && pixel(1) < y + height) {
                        
                        int cone_class;
                        float conf = 0.0f;

                        for (int j = 0; j < 5; j++) {
                            if (data[i * attributes + j + 5] > conf) {
                                cone_class = j;
                                conf = data[i * attributes + j + 5];
                            };
                        }

                        cv::Scalar color;

                        switch (cone_class) {
                            case 0:
                                return std::make_pair(2, conf);  // Blue
                            case 4:
                                return std::make_pair(1, conf);  // Yellow
                            case 2:
                                return std::make_pair(0, conf);  // Orange
                            case 3:
                                return std::make_pair(0, conf);  // Big Orange
                            default:
                                return std::make_pair(-1, 0.0);  // Unknown
                        }
                    }
                }
            }

            // No detection
            return std::make_pair(-1, 0.0);
        }

        

        void draw_bounding_boxes(cv::Mat& frame, cv::Mat& canvas, std::vector<cv::Mat> detections, int rows, int cols, double confidence_threshold) {
            float alpha = .3f;
            float* data = (float*)detections[0].data;
            int num_detections = detections[0].size[1];  // 25200
            int attributes = detections[0].size[2];      // 10

            float x_scale = static_cast<float>(rows) / 640.0f;
            float y_scale = static_cast<float>(cols) / 640.0f;

            for (int i = 0; i < num_detections; i++) {
                float confidence = data[i * attributes + 4];
                if (confidence > confidence_threshold) {
                    float cx = data[i * attributes];
                    float cy = data[i * attributes + 1];
                    float w = data[i * attributes + 2];
                    float h = data[i * attributes + 3];

                    // Convert center coordinates to top-left corner (x, y)
                    int x = static_cast<int>((cx - w / 2) * x_scale);
                    int y = static_cast<int>((cy - h / 2) * y_scale);
                    int width = static_cast<int>(w * x_scale);
                    int height = static_cast<int>(h * y_scale);
                    
                    int cone_class;
                    float conf = 0.0f;

                    for (int j = 0; j < 5; j++) {
                        if (data[i * attributes + j + 5] > conf) {
                            cone_class = j;
                            conf = data[i * attributes + j + 5];
                        };
                    }

                    cv::Scalar color;
                    switch (cone_class) {
                        case 0:
                            color = cv::Scalar(255, 0, 0);  // Blue
                            break;
                        case 4:
                            color = cv::Scalar(0, 255, 255);  // Yellow
                            break;
                        case 2:
                            color = cv::Scalar(0, 69, 255);  // Orange
                            break;
                        case 3:
                            color = cv::Scalar(0, 69, 255);  // Green
                            break;
                        default:
                            color = cv::Scalar(0, 255, 255);  // Unknown
                            break;
                    }

                    // Draw bounding box (assuming class color is handled elsewhere)
                    cv::rectangle(frame, cv::Rect(x, y, width, height), color, 2);
                }
            }

            cv::addWeighted(canvas, alpha, frame, 1-alpha, 0, frame);
        }

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

        std::vector<cv::Mat> process_frame(
            const cv::Mat& frame,
            cv::dnn::Net& net,
            int yolo_width,
            int yolo_height
        ) {
            // Prepare input
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
            
            // Run inference
            net.setInput(blob);
            std::vector<cv::Mat> outputs;
            net.forward(outputs, net.getUnconnectedOutLayersNames());

            // Return the detection results
            return outputs;
        }
    }
}