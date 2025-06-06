#pragma once

#include <opencv2/opencv.hpp>
#include <Eigen/Dense>

namespace cones {
    namespace coloring {
        namespace hsv {
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
             * @brief Uses HSV color filtering to determine cone color
             * 
             * @param pixel The pixel location to check
             * @param image The image to analyze
             * @param yellow_filter_low Lower HSV bound for yellow detection
             * @param yellow_filter_high Upper HSV bound for yellow detection
             * @param blue_filter_low Lower HSV bound for blue detection
             * @param blue_filter_high Upper HSV bound for blue detection
             * @param orange_filter_low Lower HSV bound for orange detection
             * @param orange_filter_high Upper HSV bound for orange detection
             * @param confidence_threshold Minimum confidence to report a color
             * @return std::pair<int, double> Color ID (-1, 0=orange, 1=yellow, 2=blue) and confidence
             */
            std::pair<int, double> get_color(
                Eigen::Vector3d& pixel,
                cv::Mat image,
                const cv::Scalar& yellow_filter_low,
                const cv::Scalar& yellow_filter_high,
                const cv::Scalar& blue_filter_low,
                const cv::Scalar& blue_filter_high,
                const cv::Scalar& orange_filter_low,
                const cv::Scalar& orange_filter_high,
                double confidence_threshold
            );
        }
    }
}