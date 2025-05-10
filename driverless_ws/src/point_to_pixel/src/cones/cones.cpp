#include "cones.hpp"

namespace cones {
    Cone find_closest_cone(const Cones& cones) {
        if (cones.empty()) {
            throw std::runtime_error("Empty cone list");
        }
        
        return *std::min_element(cones.begin(), cones.end(),
            [](const Cone& a, const Cone& b) {
                return a.distance < b.distance;
            });
    }

    double calculate_angle(const Cone& from, const Cone& to) {
        return std::atan2(to.point.y - from.point.y, to.point.x - from.point.x);
    }

    Cones order_cones(const Cones& unordered_cones, double max_distance_threshold) {
        if (unordered_cones.size() <= 1) {
            return unordered_cones;
        }

        Cones ordered_cones;
        Cones remaining_cones = unordered_cones;
        
        // Start with the closest cone to origin
        Cone current_cone = find_closest_cone(remaining_cones);
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
                    double angle_a = calculate_angle(current_cone, a);
                    double angle_b = calculate_angle(current_cone, b);
                    
                    // Calculate angle differences
                    double angle_diff_a = std::abs(angle_a - prev_angle);
                    double angle_diff_b = std::abs(angle_b - prev_angle);
                    
                    // Combine distance and angle criteria
                    double score_a = std::pow(std::abs(a.point.x - current_cone.point.x), 2) + std::pow(std::abs(a.point.y - current_cone.point.y), 2);
                    double score_b = std::pow(std::abs(b.point.x - current_cone.point.x), 2) + std::pow(std::abs(b.point.y - current_cone.point.y), 2);
                    // double score_a = 0.7 * (a.distance / current_cone.distance) + 
                    //             0.3 * angle_diff_a;
                    // double score_b = 0.7 * (b.distance / current_cone.distance) + 
                    //             0.3 * angle_diff_b;
                    
                    return score_a < score_b;
                });

            auto next_cone = *next_cone_it;

            // Check if next cone is more than {max_threshold} meters away
            auto bw_cone_dist = std::sqrt(std::pow(std::abs(next_cone.point.x - current_cone.point.x), 2) + std::pow(std::abs(next_cone.point.y - current_cone.point.y), 2));
            if (bw_cone_dist > max_distance_threshold) {
                break;
            }

            current_cone = next_cone;
            ordered_cones.push_back(current_cone);
            prev_angle = calculate_angle(ordered_cones[ordered_cones.size()-2], current_cone);
            remaining_cones.erase(next_cone_it);
        }

        return ordered_cones;
    }
    
    // Helper function to convert TrackBounds to XY training data
    std::pair<std::vector<std::vector<double>>, std::vector<double>> cones_to_xy(const TrackBounds &track_bounds) {
        std::vector<std::vector<double>> X;
        std::vector<double> y;
        
        // Process yellow cones (label 0.0)
        for (const auto& cone : track_bounds.yellow) {
            X.push_back({cone.point.x, cone.point.y});
            y.push_back(0.0);  // Yellow label
        }
        
        // Process blue cones (label 1.0)
        for (const auto& cone : track_bounds.blue) {
            X.push_back({cone.point.x, cone.point.y});
            y.push_back(1.0);  // Blue label
        }
        
        return {X, y};
    }
    
    // Add dummy cones
    void supplement_cones(TrackBounds &track_bounds) {

        geometry_msgs::msg::Point yellow_cone;
        yellow_cone.x = 2.0;
        yellow_cone.y = 1.0;
        yellow_cone.z = 0;

        track_bounds.yellow.push_back(yellow_cone);

        geometry_msgs::msg::Point blue_cone;
        blue_cone.x = -2.0;
        blue_cone.y = 1.0;
        blue_cone.z = 0;
        
        track_bounds.blue.push_back(blue_cone);
    }
    
    // Adds a ring of circles around a cone
    void augment_cones_circle(TrackBounds &track_bounds, int degrees, double radius) {
        // Convert angle from degrees to radians
        double angle_radians = degrees * (M_PI / 180.0);
        
        // Create vector of angles around the circle
        std::vector<double> angles;
        for (double angle = 0; angle < 2 * M_PI; angle += angle_radians) {
            angles.push_back(angle);
        }
        
        // Augment blue cones
        std::vector<Cone> blue_extra;
        for (const auto& cone : track_bounds.blue) {
            for (const auto& angle : angles) {
                // Create a new cone rotated around the circle
                double new_x = cone.point.x + radius * std::cos(angle);
                double new_y = cone.point.y + radius * std::sin(angle);
                geometry_msgs::msg::Point new_blue;
                new_blue.x = new_x;
                new_blue.y = new_y;
                new_blue.z = cone.point.z;
                blue_extra.push_back(new_blue);
            }
        }
        
        // Augment yellow cones
        std::vector<Cone> yellow_extra;
        for (const auto& cone : track_bounds.yellow) {
            for (const auto& angle : angles) {
                // Create a new cone rotated around the circle
                double new_x = cone.point.x + radius * std::cos(angle);
                double new_y = cone.point.y + radius * std::sin(angle);
                geometry_msgs::msg::Point new_yellow;
                new_yellow.x = new_x;
                new_yellow.y = new_y;
                new_yellow.z = cone.point.z;
                yellow_extra.push_back(new_yellow);
            }
        }
        
        // Add augmented cones to the original lists
        track_bounds.blue.insert(track_bounds.blue.end(), blue_extra.begin(), blue_extra.end());
        track_bounds.yellow.insert(track_bounds.yellow.end(), yellow_extra.begin(), yellow_extra.end());
    }
}