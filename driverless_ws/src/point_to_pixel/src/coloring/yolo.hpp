#pragma once

#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>
#include <Eigen/Dense>

namespace coloring {
    namespace yolo {
        /**
         * @brief Uses YOLO to determine cone color
         * 
         * @param pixel The pixel location to check
         * @param detection YOLO detection output
         * @param cols Image columns
         * @param rows Image rows
         * @param confidence_threshold Minimum confidence to report a detection
         * @return std::pair<int, double> Color ID and confidence
         */
        std::pair<int, double> get_color(
            Eigen::Vector3d& pixel,
            cv::Mat detection,
            int cols,
            int rows,
            double confidence_threshold
        );

        /**
         * @brief Initialize the YOLO model
         * 
         * @param model_path Path to the ONNX model file
         * @return cv::dnn::Net Initialized neural network
         */
        cv::dnn::Net init_model(const std::string& model_path);

        /**
         * @brief Process frame with YOLO
         * 
         * @param frame Input frame
         * @param net Neural network
         * @param yolo_width Width for YOLO input
         * @param yolo_height Height for YOLO input
         * @return cv::Mat Detection results
         */
        cv::Mat process_frame(
            const cv::Mat& frame,
            cv::dnn::Net& net,
            int yolo_width = 640,
            int yolo_height = 640
        );
    }
}