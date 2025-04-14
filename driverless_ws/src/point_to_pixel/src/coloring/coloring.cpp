#include "coloring.hpp"

namespace coloring {
    int get_cone_class(
        std::pair<Eigen::Vector3d, Eigen::Vector3d> pixel_pair,
        std::pair<cv::Mat, cv::Mat> frame_pair,
        std::pair<cv::Mat, cv::Mat> detection_pair,
        const cv::Scalar& yellow_filter_low,
        const cv::Scalar& yellow_filter_high,
        const cv::Scalar& blue_filter_low,
        const cv::Scalar& blue_filter_high,
        const cv::Scalar& orange_filter_low,
        const cv::Scalar& orange_filter_high,
        double confidence_threshold,
        bool use_yolo
    ) {
        // Declare the pixels in left (l) and right (r) camera space
        // CONE_CLASS [-1, 0, 1, 2], CONFIDENCE [0<--->1]
        std::pair<int, double> pixel_l;
        std::pair<int, double> pixel_r;

        // Identify the color at the transformed image pixel
        if (use_yolo) {
            pixel_l = yolo::get_color(
                pixel_pair.first, 
                detection_pair.first, 
                frame_pair.first.cols, 
                frame_pair.first.rows,
                confidence_threshold
            );
            
            pixel_r = yolo::get_color(
                pixel_pair.second, 
                detection_pair.second, 
                frame_pair.second.cols, 
                frame_pair.second.rows,
                confidence_threshold
            );
        } else {
            pixel_l = hsv::get_color(
                pixel_pair.first, 
                frame_pair.first,
                yellow_filter_low,
                yellow_filter_high,
                blue_filter_low,
                blue_filter_high,
                orange_filter_low,
                orange_filter_high,
                confidence_threshold
            );
            
            pixel_r = hsv::get_color(
                pixel_pair.second, 
                frame_pair.second,
                yellow_filter_low,
                yellow_filter_high,
                blue_filter_low,
                blue_filter_high,
                orange_filter_low,
                orange_filter_high,
                confidence_threshold
            );
        }

        // Logic for handling detection results
        // Return l if r did not detect color
        if (pixel_l.first != -1 && pixel_r.first == -1) return pixel_l.first;
        // Return r if l did not detect color
        else if (pixel_l.first == -1 && pixel_r.first != -1) return pixel_r.first;
        // Return result with highest confidence if both detect color
        else if (pixel_l.first != -1 && pixel_r.first != -1) {
            if (pixel_l.second > pixel_r.second) return pixel_l.first;
            else return pixel_r.first;
        }
        else return -1;
    }
}