#include "../include/yolo.hpp"
#include <vector>

namespace point_to_pixel {
namespace coloring {
namespace yolo {
    int get_cone_class(
        std::pair<Eigen::Vector3d, Eigen::Vector3d> pixel_pair,
        std::pair<cv::Mat, cv::Mat> frame_pair,
        std::pair<std::vector<cv::Mat>, std::vector<cv::Mat>> detection_pair,
        const cv::Scalar& yellow_filter_low,
        const cv::Scalar& yellow_filter_high,
        const cv::Scalar& blue_filter_low,
        const cv::Scalar& blue_filter_high,
        const cv::Scalar& orange_filter_low,
        const cv::Scalar& orange_filter_high,
        double confidence_threshold
    ) {
        // Declare the pixels in left (l) and right (r) camera space
        // CONE_CLASS [-1, 0, 1, 2], CONFIDENCE [0<--->1]
        std::pair<int, double> pixel_l;
        std::pair<int, double> pixel_r;

        // Identify the color at the transformed image pixel
        pixel_l = coloring::yolo::get_color(
            pixel_pair.first, 
            detection_pair.first, 
            frame_pair.first.cols, 
            frame_pair.first.rows,
            confidence_threshold
        );
        
        pixel_r = coloring::yolo::get_color(
            pixel_pair.second, 
            detection_pair.second, 
            frame_pair.second.cols, 
            frame_pair.second.rows,
            confidence_threshold
        );

        // Logic for handling detection results
        // Return l if r did not detect color
        if (pixel_l.first != -1 && pixel_r.first == -1) return pixel_l.first;
        // Return r if l did not detect color
        else if (pixel_l.first == -1 && pixel_r.first != -1) return pixel_r.first;
        // Return result with highest confidence if both detect color
        else if (pixel_l.first != -1 && pixel_r.first != -1) {
            return (pixel_l.second > pixel_r.second) ? pixel_l.first : pixel_r.first;
        }
        else return -1;
    }

    float depth_to_box_height(float depth) {
        // Map depth to box height
        return (15.6f + 198.0f / depth);
    }

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

        float prob = 0.0f; // Store higher probability here
        std::vector<std::tuple<double, int, double>> boxes; // Vector of (dist from center, cone class)
        
        // Loop through detections
        for (int i = 0; i < num_detections; i++) {
            float confidence = data[i * attributes + 4];

            // If YOLO is more confident than threshold...
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

                // If pixel is inside the bounding box add color to the vector of all boxes pixel is in
                if (pixel(0) > x && pixel(0) < x + width && pixel(1) > y && pixel(1) < y + height) {
                    int c_c;

                    for (int j = 0; j < 5; j++) {
                        if (data[i * attributes + j + 5] > prob) {
                            c_c = j;
                            prob = data[i * attributes + j + 5];
                        };
                    }

                    float dist_from_center = std::sqrt((pixel(0) - cx * x_scale) *(pixel(0) - cx * x_scale) + 
                                                    (pixel(1) - cy * y_scale) * (pixel(1) - cy * y_scale));
                    float expected_box_height = depth_to_box_height(pixel(2));
                    float box_height_diff = std::abs(height - expected_box_height);
                    #if box_heuristic == 0
                    boxes.emplace_back(prob, c_c, prob);
                    #endif
                    #if box_heuristic == 1
                    boxes.emplace_back(dist_from_center, c_c, prob);
                    #endif
                    #if box_heuristic == 2
                    boxes.emplace_back(box_height_diff, c_c, prob);
                    #endif
                }
            }
        }
        
        if (boxes.size() > 0) {

            // Find closest box
            double smallest = std::get<0>(boxes[0]) + 1.0;
            int cone_class = -1;
            double prob = 0.0;
            for (int i = 0; i < boxes.size(); i++) {
                if (std::get<0>(boxes[i]) < smallest) {
                    cone_class = std::get<1>(boxes[i]);
                    prob = std::get<2>(boxes[i]);
                }
            }
            switch (cone_class) {
                case 0:
                    return {2, prob}; // Blue
                case 4:
                    return {1, prob}; // Yellow
                case 2:
                    return {0, prob}; // Orange
                case 3:
                    return {0, prob}; // Big Orange
                default:
                    return {-1, 0.0};  // Unknown
            }
        }
        
        // No detection
        return {-1, 0.0};
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
                cv::rectangle(canvas, cv::Rect(x, y, width, height), color, cv::FILLED);
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

        float* data = (float*)outputs[0].data;
        int num_detections = outputs[0].size[1];  // 25200
        int attributes = outputs[0].size[2];      // 10

        // Return the detection results
        return outputs;
    }
} // namespace yolo
} // namespace coloring
} // namespace point_to_pixel