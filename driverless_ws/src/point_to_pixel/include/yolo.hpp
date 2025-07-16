#pragma once

#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>
#include <Eigen/Dense>

#define box_heuristic 2 // 0: none, 1: distance from center, 2: depth

namespace point_to_pixel {
namespace coloring {
namespace yolo {
            /**
             * @brief Determines cone class from pixel pairs across cameras
             * 
             * @param pixel_pair Pixel coordinates in both cameras
             * @param frame_pair Frames from both cameras
             * @param detection_pair YOLO detection results (if using YOLO)
             * @param yellow_filter_low Lower HSV bound for yellow detection
             * @param yellow_filter_high Upper HSV bound for yellow detection  
             * @param blue_filter_low Lower HSV bound for blue detection
             * @param blue_filter_high Upper HSV bound for blue detection
             * @param orange_filter_low Lower HSV bound for orange detection  
             * @param orange_filter_high Upper HSV bound for orange detection
             * @param confidence_threshold Minimum confidence to report a color
             * @return int Cone class (-1=unknown, 0=orange, 1=yellow, 2=blue)
             */
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
            );


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
} // namespace yolo
} // namespace coloring
} // namespace point_to_pixel