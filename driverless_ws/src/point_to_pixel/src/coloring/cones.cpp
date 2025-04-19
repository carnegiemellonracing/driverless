#include "cones.hpp"

namespace cones {
    struct Cone {
        geometry_msgs::msg::Point point;
        double distance;
        Cone(const geometry_msgs::msg::Point& p) : point(p) {
            distance = std::sqrt(p.x * p.x + p.y * p.y);
        }
    };

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

    Cone findClosestCone(const std::vector<Cone>& cones) {
        if (cones.empty()) {
            throw std::runtime_error("Empty cone list");
        }
        
        return *std::min_element(cones.begin(), cones.end(),
            [](const Cone& a, const Cone& b) {
                return a.distance < b.distance;
            });
    }

    double calculateAngle(const Cone& from, const Cone& to) {
        return std::atan2(to.point.y - from.point.y, to.point.x - from.point.x);
    }

    std::vector<Cone> orderConesByPathDirection(const std::vector<Cone>& unordered_cones) {
        if (unordered_cones.size() <= 1) {
            return unordered_cones;
        }

        std::vector<Cone> ordered_cones;
        std::vector<Cone> remaining_cones = unordered_cones;
        
        // Start with the closest cone to origin
        Cone current_cone = findClosestCone(remaining_cones);
        ordered_cones.push_back(current_cone);
        
        // Remove the first cone from remaining cones
        remaining_cones.erase(
            std::remove_if(remaining_cones.begin(), remaining_cones.end(),
                [&current_cone](const Cone& c) {
                    return c.point.x == current_cone.point.x && 
                        c.point.y == current_cone.point.y;
                }), 
            remaining_cones.end());

        double prev_angle = std::atan2(current_cone.point.y, current_cone.point.x);

        while (!remaining_cones.empty()) {
            // Find next best cone based on distance and angle continuation
            auto next_cone_it = std::min_element(remaining_cones.begin(), remaining_cones.end(),
                [&](const Cone& a, const Cone& b) {
                    double angle_a = calculateAngle(current_cone, a);
                    double angle_b = calculateAngle(current_cone, b);
                    
                    // Calculate angle differences
                    double angle_diff_a = std::abs(angle_a - prev_angle);
                    double angle_diff_b = std::abs(angle_b - prev_angle);
                    
                    // Combine distance and angle criteria
                    double score_a = 0.7 * (a.distance / current_cone.distance) + 
                                0.3 * angle_diff_a;
                    double score_b = 0.7 * (b.distance / current_cone.distance) + 
                                0.3 * angle_diff_b;
                    
                    return score_a < score_b;
                });

            current_cone = *next_cone_it;
            ordered_cones.push_back(current_cone);
            prev_angle = calculateAngle(ordered_cones[ordered_cones.size()-2], current_cone);
            remaining_cones.erase(next_cone_it);
        }

        return ordered_cones;
    }
}