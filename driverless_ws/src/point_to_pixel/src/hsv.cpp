#include "../include/hsv.hpp"

namespace point_to_pixel {
    namespace coloring {
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

        pixel_l = coloring::hsv::get_color(
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
        
        pixel_r = coloring::hsv::get_color(
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

        namespace hsv {
            std::pair<int, double> get_color(
                Eigen::Vector3d& pixel,
                cv::Mat img,
                const cv::Scalar& yellow_filter_low,
                const cv::Scalar& yellow_filter_high,
                const cv::Scalar& blue_filter_low,
                const cv::Scalar& blue_filter_high,
                const cv::Scalar& orange_filter_low,
                const cv::Scalar& orange_filter_high,
                double confidence_threshold
            ) {
                // Ratio of color in relation to all other colors
                const double RATIO_THRESHOLD = 1.5;
                const double NOMINAL_SIDE_LENGTH = 25;
                const double SCALING_FACTOR = 1.0;

                // Find real side length (MATTS AN ECON MAJOR) by dividing by scaling factor (distance)
                int real_side_length = NOMINAL_SIDE_LENGTH; // / (SCALING_FACTOR * pixel(2));

                int x = static_cast<int>(pixel(0));
                int y = static_cast<int>(pixel(1));
                int height = img.rows;
                int width = img.cols;
                int x_min = std::max(0, x - real_side_length);
                int x_max = std::min(width, x + real_side_length);
                int y_min = std::max(0, y - real_side_length);
                int y_max = std::min(height, y + real_side_length);

                // Transformed point out of frame
                if (x_min >= x_max || y_min >= y_max) {
                    return std::make_pair(-1, 1.0);
                }
                
                // Extract ROI and convert to HSV
                cv::Mat roi = img(cv::Range(y_min, y_max), cv::Range(x_min, x_max));
                cv::Mat hsv_roi;
                cv::cvtColor(roi, hsv_roi, cv::COLOR_BGR2HSV);
                
                // Define HSV color ranges
                std::pair<cv::Scalar, cv::Scalar> yellow_range = {yellow_filter_low, yellow_filter_high};
                std::pair<cv::Scalar, cv::Scalar> blue_range = {blue_filter_low, blue_filter_high};
                std::pair<cv::Scalar, cv::Scalar> orange_range = {orange_filter_low, orange_filter_high};
                
                // Create color masks
                cv::Mat yellow_mask = cv::Mat::zeros(hsv_roi.size(), CV_8UC1);
                cv::Mat blue_mask = cv::Mat::zeros(hsv_roi.size(), CV_8UC1);
                cv::Mat orange_mask = cv::Mat::zeros(hsv_roi.size(), CV_8UC1);
                cv::Mat temp_mask;
                
                // Apply color masks
                cv::inRange(hsv_roi, yellow_range.first, yellow_range.second, temp_mask);
                cv::bitwise_or(yellow_mask, temp_mask, yellow_mask);
                
                cv::inRange(hsv_roi, blue_range.first, blue_range.second, temp_mask);
                cv::bitwise_or(blue_mask, temp_mask, blue_mask);
                
                cv::inRange(hsv_roi, orange_range.first, orange_range.second, temp_mask);
                cv::bitwise_or(orange_mask, temp_mask, orange_mask);
                
                // Calculate color percentages
                double total_pixels = (y_max - y_min) * (x_max - x_min);
                double yellow_pixels = cv::countNonZero(yellow_mask);
                double blue_pixels = cv::countNonZero(blue_mask);
                double orange_pixels = cv::countNonZero(orange_mask);
                double yellow_percentage = yellow_pixels / total_pixels;
                double blue_percentage = blue_pixels / total_pixels;
                double orange_percentage = orange_pixels / total_pixels;

                // Determine cone color
                if (orange_percentage > confidence_threshold && 
                    orange_percentage > std::max(yellow_percentage, blue_percentage) * RATIO_THRESHOLD) {
                    return std::make_pair(0, orange_percentage);
                } else if (yellow_percentage > confidence_threshold && 
                        yellow_percentage > std::max(blue_percentage, orange_percentage) * RATIO_THRESHOLD) {
                    return std::make_pair(1, yellow_percentage);
                } else if (blue_percentage > confidence_threshold && 
                        blue_percentage > std::max(yellow_percentage, orange_percentage) * RATIO_THRESHOLD) {
                    return std::make_pair(2, blue_percentage);
                }
                
                return std::make_pair(-1, 1.0);
            }
        }
    }
}