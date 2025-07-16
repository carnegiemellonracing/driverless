#include "yolo_predictor.hpp"
#include <iostream>
#include <algorithm>
#include <cmath>
#include <tuple>

namespace point_to_pixel
{
    ConeClass YoloPredictor::predict_color(
        std::pair<Eigen::Vector3d, Eigen::Vector3d> pixel_pair,
        std::pair<cv::Mat, cv::Mat> frame_pair,
        double confidence_threshold)
    {
        // Declare the pixels in left (l) and right (r) camera space
        std::pair<ConeClass, double> pixel_l;
        std::pair<ConeClass, double> pixel_r;

        // Identify the color at the transformed image pixel
        pixel_l = get_color(
            pixel_pair.first,
            detections_l,
            frame_pair.first.cols,
            frame_pair.first.rows,
            confidence_threshold);

        pixel_r = get_color(
            pixel_pair.second,
            detections_r,
            frame_pair.second.cols,
            frame_pair.second.rows,
            confidence_threshold);

        // Logic for handling detection results
        // Return l if r did not detect color
        if (pixel_l.first != ConeClass::UNKNOWN && pixel_r.first == ConeClass::UNKNOWN)
        {
            return pixel_l.first;
        }
        // Return r if l did not detect color
        else if (pixel_l.first == ConeClass::UNKNOWN && pixel_r.first != ConeClass::UNKNOWN)
        {
            return pixel_r.first;
        }
        // Return result with highest confidence if both detect color
        else if (pixel_l.first != ConeClass::UNKNOWN && pixel_r.first != ConeClass::UNKNOWN)
        {
            return (pixel_l.second > pixel_r.second) ? pixel_l.first : pixel_r.first;
        }
        else
        {
            return ConeClass::UNKNOWN;
        }
    }

    std::pair<ConeClass, double> YoloPredictor::get_color(
        Eigen::Vector3d &pixel,
        std::vector<cv::Mat> detections,
        int cols,
        int rows,
        double confidence_threshold)
    {
        if (detections.empty())
        {
            return {ConeClass::UNKNOWN, 0.0};
        }

        float *data = (float *)detections[0].data;
        int num_detections = detections[0].size[1]; // 25200
        int attributes = detections[0].size[2];     // 10

        float x_scale = static_cast<float>(cols) / 640.0f;
        float y_scale = static_cast<float>(rows) / 640.0f;

        float prob = 0.0f;
        std::vector<std::tuple<double, int, double>> boxes; // Vector of (metric, cone class, confidence)

        // Loop through detections
        for (int i = 0; i < num_detections; i++)
        {
            float confidence = data[i * attributes + 4];

            // If YOLO is more confident than threshold...
            if (confidence > confidence_threshold)
            {
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
                if (pixel(0) > x && pixel(0) < x + width && pixel(1) > y && pixel(1) < y + height)
                {
                    int c_c = 0;
                    float max_class_prob = 0.0f;

                    for (int j = 0; j < 5; j++)
                    {
                        if (data[i * attributes + j + 5] > max_class_prob)
                        {
                            c_c = j;
                            max_class_prob = data[i * attributes + j + 5];
                        }
                    }

                    float expected_box_height = depth_to_box_height(pixel(2));
                    float box_height_diff = std::abs(height - expected_box_height);

                    // Use box height heuristic - smaller difference is better
                    boxes.emplace_back(box_height_diff, c_c, max_class_prob);
                }
            }
        }

        if (boxes.size() > 0)
        {
            // Find box with smallest height difference (best match for expected size)
            double best_metric = std::get<0>(boxes[0]);
            int cone_class = std::get<1>(boxes[0]);
            double confidence = std::get<2>(boxes[0]);

            for (const auto &box : boxes)
            {
                if (std::get<0>(box) < best_metric)
                {
                    best_metric = std::get<0>(box);
                    cone_class = std::get<1>(box);
                    confidence = std::get<2>(box);
                }
            }

            // Convert YOLO class to ConeClass enum
            switch (cone_class)
            {
            case 0:
                return {ConeClass::BLUE, confidence};
            case 4:
                return {ConeClass::YELLOW, confidence};
            case 2:
                return {ConeClass::ORANGE, confidence};
            case 3:
                return {ConeClass::ORANGE, confidence}; // Big Orange
            default:
                return {ConeClass::UNKNOWN, 0.0};
            }
        }

        // No detection
        return {ConeClass::UNKNOWN, 0.0};
    }

    float YoloPredictor::depth_to_box_height(float depth)
    {
        // Map depth to box height
        return (15.6f + 198.0f / depth);
    }

    cv::dnn::Net YoloPredictor::init_model(const std::string &model_path)
    {
        cv::dnn::Net net = cv::dnn::readNetFromONNX(model_path);

        // Set CUDA if available
        if (!net.empty())
        {
            net.setPreferableBackend(cv::dnn::DNN_BACKEND_CUDA);
            net.setPreferableTarget(cv::dnn::DNN_TARGET_CUDA);
        }
        else
        {
            std::cerr << "Failed to load YOLO model from: " << model_path << std::endl;
        }

        return net;
    }

    void YoloPredictor::process_frame(const cv::Mat &frame, bool is_left_frame)
    {
        // Prepare input
        cv::Mat blob;
        cv::dnn::blobFromImage(
            frame,
            blob,
            1 / 255.0,
            cv::Size(yolo_width_, yolo_height_),
            cv::Scalar(),
            true,
            false);

        // Run inference
        net_.setInput(blob);
        std::vector<cv::Mat> outputs;
        net_.forward(outputs, net_.getUnconnectedOutLayersNames());

        if (is_left_frame) detections_l = outputs;
        else detections_r = outputs;
    }

    void YoloPredictor::draw_bounding_boxes(
        cv::Mat &frame,
        cv::Mat &canvas,
        std::vector<cv::Mat> detections,
        int rows,
        int cols,
        double confidence_threshold)
    {
        if (detections.empty())
            return;

        float alpha = 0.3f;
        float *data = (float *)detections[0].data;
        int num_detections = detections[0].size[1]; // 25200
        int attributes = detections[0].size[2];     // 10

        float x_scale = static_cast<float>(cols) / 640.0f;
        float y_scale = static_cast<float>(rows) / 640.0f;

        for (int i = 0; i < num_detections; i++)
        {
            float confidence = data[i * attributes + 4];
            if (confidence > confidence_threshold)
            {
                float cx = data[i * attributes];
                float cy = data[i * attributes + 1];
                float w = data[i * attributes + 2];
                float h = data[i * attributes + 3];

                // Convert center coordinates to top-left corner (x, y)
                int x = static_cast<int>((cx - w / 2) * x_scale);
                int y = static_cast<int>((cy - h / 2) * y_scale);
                int width = static_cast<int>(w * x_scale);
                int height = static_cast<int>(h * y_scale);

                int cone_class = 0;
                float conf = 0.0f;

                for (int j = 0; j < 5; j++)
                {
                    if (data[i * attributes + j + 5] > conf)
                    {
                        cone_class = j;
                        conf = data[i * attributes + j + 5];
                    }
                }

                cv::Scalar color;
                switch (cone_class)
                {
                case 0:
                    color = cv::Scalar(255, 0, 0); // Blue
                    break;
                case 4:
                    color = cv::Scalar(0, 255, 255); // Yellow
                    break;
                case 2:
                    color = cv::Scalar(0, 69, 255); // Orange
                    break;
                case 3:
                    color = cv::Scalar(0, 69, 255); // Big Orange
                    break;
                default:
                    color = cv::Scalar(0, 255, 255); // Unknown
                    break;
                }

                // Draw bounding box
                cv::rectangle(canvas, cv::Rect(x, y, width, height), color, cv::FILLED);
                cv::rectangle(frame, cv::Rect(x, y, width, height), color, 2);
            }
        }

        cv::addWeighted(canvas, alpha, frame, 1 - alpha, 0, frame);
    }
} // namespace point_to_pixel