#pragma once

#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>
#include <Eigen/Dense>

namespace cones {
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
            std::vector<cv::Mat> detection,
            int cols,
            int rows,
            double confidence_threshold
        );
        
        /**
         * @brief Labels image frames with bounding boxes. Works with references.
         * 
         * @param frame image frame 
         * @param canvas clone of image frame. Gets merged to image frame to have a see-through box
         * @param detections output of YOLO model
         * @param cols Image columns
         * @param rows Image rows
         * @param confidence_threshold Use bounding box if YOLO conf is higher than threshold
         * 
         * @return Nothing, directly applies bounding boxes to the input images.
         */
        void draw_bounding_boxes(
            cv::Mat& frame,
            cv::Mat& canvas,
            std::vector<cv::Mat> detections,
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
         * @return float *detection results
         */
        std::vector<cv::Mat> process_frame(
            const cv::Mat& frame,
            cv::dnn::Net& net,
            int yolo_width = 640,
            int yolo_height = 640
        );
    }
}